from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
import typer
from mcp_types import ImageContent, TextContent
from PIL import Image

from fast_agent.cli.constants import normalize_convenience_flag_args
from fast_agent.cli.runtime.request_builders import validate_image_gen_conflicts
from fast_agent.config import ImageGenSettings, Settings
from fast_agent.core.agent_card_loader import load_agent_cards
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.tool_result_metadata import get_tool_result_media_preview
from fast_agent.ptc.vocabulary import is_builtin_tool
from fast_agent.tools import image_gen

if TYPE_CHECKING:
    from fast_agent.tools.execution_environment import ShellEnvironment

from fast_agent.tools.image_gen import (
    EDIT_IMAGE_TOOL_NAME,
    GENERATE_IMAGE_TOOL_NAME,
    ImageGenRuntime,
    build_image_gen_tools,
    resolve_image_gen_model,
)


def _png_bytes(
    size: tuple[int, int] = (32, 24),
    color: tuple[int, int, int] = (200, 40, 40),
) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(
    size: tuple[int, int] = (32, 24),
    color: tuple[int, int, int] = (40, 80, 220),
) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _patch_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> None:
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(image_gen.httpx, "AsyncClient", fake_async_client)


def test_resolve_image_gen_model_matches_main_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_gen, "_resolve_backend_endpoint", lambda *_a, **_kw: ("https://x", {})
    )

    codex = resolve_image_gen_model(None, main_provider=Provider.CODEX_RESPONSES)
    assert codex.backend == "codexresponses"
    assert codex.model == "gpt-image-2.6-flare"

    openai = resolve_image_gen_model(None, main_provider=Provider.OPENAI)
    assert openai.backend == "openai"
    assert openai.model == "gpt-image-2.6-flare"

    xai = resolve_image_gen_model(None, main_provider=Provider.XAI)
    assert xai.backend == "xai"
    assert xai.model == "grok-imagine-image-2.0"


def test_resolve_image_gen_model_fallback_precedence_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(image_gen, "get_codex_access_token", lambda: None)
    monkeypatch.setattr(image_gen, "get_xai_access_token", lambda: None)
    monkeypatch.setattr(
        image_gen.ProviderKeyManager,
        "_provider_specific_fallback_key",
        staticmethod(lambda _p: None),
    )

    resolved = resolve_image_gen_model(None, settings=Settings(), main_provider=Provider.ANTHROPIC)
    assert resolved.spec == "xai.grok-imagine-image-2.0"

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ProviderKeyError):
        resolve_image_gen_model(None, settings=Settings(), main_provider=Provider.ANTHROPIC)


def test_resolve_image_gen_model_shorthands_bare_names_and_query_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_gen, "_resolve_backend_endpoint", lambda *_a, **_kw: ("https://x", {})
    )

    r1 = resolve_image_gen_model("xai?resolution=2k&quality=medium")
    assert r1.spec == "xai.grok-imagine-image-2.0"
    assert r1.resolution == "2k"
    assert r1.default_quality == "medium"

    r2 = resolve_image_gen_model(
        "gpt-image-2.6-sunburst?complexity=high",
        main_provider=Provider.OPENAI,
    )
    assert r2.spec == "openai.gpt-image-2.6-sunburst"
    assert r2.complexity == "high"

    r3 = resolve_image_gen_model("grok-imagine-image")
    assert r3.spec == "xai.grok-imagine-image"
    assert not r3.supports_quality


def test_tool_schemas_are_provider_tailored_and_marked_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        image_gen, "_resolve_backend_endpoint", lambda *_a, **_kw: ("https://x", {})
    )

    oai_tools = {t.name: t for t in build_image_gen_tools(resolve_image_gen_model("codex"))}
    assert set(oai_tools) == {GENERATE_IMAGE_TOOL_NAME, EDIT_IMAGE_TOOL_NAME}
    assert all(is_builtin_tool(t) for t in oai_tools.values())
    assert "transparent" in oai_tools[GENERATE_IMAGE_TOOL_NAME].input_schema["properties"]
    assert "mask" in oai_tools[EDIT_IMAGE_TOOL_NAME].input_schema["properties"]
    assert oai_tools[GENERATE_IMAGE_TOOL_NAME].input_schema["properties"]["aspect_ratio"][
        "enum"
    ] == [
        "auto",
        "1:1",
        "3:2",
        "2:3",
    ]

    xai_tools = {t.name: t for t in build_image_gen_tools(resolve_image_gen_model("xai"))}
    assert "transparent" not in xai_tools[GENERATE_IMAGE_TOOL_NAME].input_schema["properties"]
    assert "mask" not in xai_tools[EDIT_IMAGE_TOOL_NAME].input_schema["properties"]
    assert (
        "16:9"
        in xai_tools[GENERATE_IMAGE_TOOL_NAME].input_schema["properties"]["aspect_ratio"]["enum"]
    )
    assert xai_tools[EDIT_IMAGE_TOOL_NAME].input_schema["properties"]["images"]["maxItems"] == 5

    xai_v1_tools = {
        t.name: t for t in build_image_gen_tools(resolve_image_gen_model("xai.grok-imagine-image"))
    }
    assert "quality" not in xai_v1_tools[GENERATE_IMAGE_TOOL_NAME].input_schema["properties"]


@pytest.mark.asyncio
async def test_codex_generate_edit_and_text_only_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png = _png_bytes((64, 48))
    b64_png = base64.b64encode(png).decode("ascii")
    captured: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured.append((request.url.path, payload))
        return httpx.Response(
            200,
            json={"data": [{"b64_json": b64_png, "revised_prompt": "refined"}]},
        )

    _patch_mock_transport(monkeypatch, handler)
    monkeypatch.setattr(
        image_gen,
        "_resolve_backend_endpoint",
        lambda *_a, **_kw: ("https://chatgpt.com/backend-api/codex", {"Authorization": "Bearer t"}),
    )

    settings = Settings(image_gen=ImageGenSettings(output_dir=str(tmp_path / "out")))
    resolved = resolve_image_gen_model("codexresponses.gpt-image-2.6-flare", settings=settings)

    rt = ImageGenRuntime(
        resolved, settings=settings, working_dir=tmp_path, supports_vision=lambda: True
    )
    res = await rt.call_tool(
        GENERATE_IMAGE_TOOL_NAME,
        {"prompt": "Red Fox Logo", "aspect_ratio": "3:2", "quality": "high", "transparent": True},
    )
    assert not res.is_error
    assert len(res.content) == 2
    assert isinstance(res.content[0], ImageContent)
    assert isinstance(res.content[1], TextContent)
    assert "Revised prompt: refined" in res.content[1].text
    assert captured[0][0].endswith("/images/generations")
    assert captured[0][1]["size"] == "1536x1024"
    assert captured[0][1]["background"] == "transparent"
    assert res.structured_content is not None
    saved = Path(res.structured_content["images"][0]["path"])
    assert saved.is_file()

    edit_res = await rt.call_tool(
        EDIT_IMAGE_TOOL_NAME,
        {"images": [str(saved)], "prompt": "Make it blue", "output_path": str(saved)},
    )
    assert not edit_res.is_error
    assert captured[1][0].endswith("/images/edits")
    assert captured[1][1]["images"][0]["image_url"].startswith("data:image/png;base64,")
    assert edit_res.structured_content is not None
    assert Path(edit_res.structured_content["images"][0]["path"]) == saved

    rt_text = ImageGenRuntime(
        resolved, settings=settings, working_dir=tmp_path, supports_vision=lambda: False
    )
    res_text = await rt_text.call_tool(GENERATE_IMAGE_TOOL_NAME, {"prompt": "Icon"})
    assert not res_text.is_error
    assert len(res_text.content) == 1
    assert isinstance(res_text.content[0], TextContent)
    preview = get_tool_result_media_preview(res_text)
    assert preview is not None and len(preview) == 1


@pytest.mark.asyncio
async def test_xai_multi_image_edit_and_extension_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jpg = _jpeg_bytes((40, 40))
    b64_jpg = base64.b64encode(jpg).decode("ascii")
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"data": [{"b64_json": b64_jpg, "mime_type": "image/jpeg"}]},
        )

    _patch_mock_transport(monkeypatch, handler)
    monkeypatch.setattr(
        image_gen,
        "_resolve_backend_endpoint",
        lambda *_a, **_kw: ("https://api.x.ai/v1", {"Authorization": "Bearer t"}),
    )

    src1 = tmp_path / "a.png"
    src2 = tmp_path / "b.png"
    src1.write_bytes(_png_bytes())
    src2.write_bytes(_png_bytes())

    resolved = resolve_image_gen_model("xai.grok-imagine-image-2.0?resolution=2k")
    rt = ImageGenRuntime(resolved, working_dir=tmp_path)
    res = await rt.call_tool(
        EDIT_IMAGE_TOOL_NAME,
        {
            "images": [str(src1), str(src2)],
            "prompt": "Combine both",
            "aspect_ratio": "16:9",
            "output_path": "combined.png",
        },
    )
    assert not res.is_error
    assert len(captured[0]["images"]) == 2
    assert captured[0]["resolution"] == "2k"
    assert captured[0]["aspect_ratio"] == "16:9"
    assert res.structured_content is not None
    saved = Path(res.structured_content["images"][0]["path"])
    assert saved.name == "combined.jpg"
    assert saved.is_file()
    last_block = res.content[-1]
    assert isinstance(last_block, TextContent)
    assert "saved with .jpg extension" in last_block.text


@pytest.mark.asyncio
async def test_svg_inline_and_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svg_bytes = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
    b64_svg = base64.b64encode(svg_bytes).decode("ascii")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": b64_svg}]})

    _patch_mock_transport(monkeypatch, handler)
    monkeypatch.setattr(
        image_gen,
        "_resolve_backend_endpoint",
        lambda *_a, **_kw: ("https://chatgpt.com/backend-api/codex", {"Authorization": "Bearer t"}),
    )

    resolved = resolve_image_gen_model("codexresponses.gpt-image-2.6-flare")
    rt = ImageGenRuntime(resolved, working_dir=tmp_path)

    svg_res = await rt.call_tool(
        GENERATE_IMAGE_TOOL_NAME,
        {"prompt": "Simple icon", "output_path": "icon.svg"},
    )
    assert not svg_res.is_error
    assert len(svg_res.content) == 1
    assert isinstance(svg_res.content[0], TextContent)
    assert "<svg" in svg_res.content[0].text
    assert (tmp_path / "icon.svg").is_file()

    err_res = await rt.call_tool(
        GENERATE_IMAGE_TOOL_NAME,
        {"prompt": "Logo", "transparent": True, "output_path": "logo.jpg"},
    )
    assert err_res.is_error
    assert isinstance(err_res.content[0], TextContent)
    assert "JPEG" in err_res.content[0].text


def test_cli_flag_normalization_conflicts_and_agent_card(tmp_path: Path) -> None:
    args = ["--model", "sonnet", "--image-gen=xai.grok-imagine-image-2.0"]
    normalize_convenience_flag_args(args)
    assert args == ["--model", "sonnet", "--image-gen-model", "xai.grok-imagine-image-2.0"]

    args_empty = ["--image-gen="]
    normalize_convenience_flag_args(args_empty)
    assert args_empty == ["--image-gen"]

    with pytest.raises(
        typer.BadParameter, match="Cannot combine --image-gen-model with --no-image-gen"
    ):
        validate_image_gen_conflicts(image_gen=False, image_gen_model="xai")

    card_path = tmp_path / "designer.md"
    card_path.write_text(
        "---\nname: designer\nmodel: sonnet\nimage_gen: xai.grok-imagine-image-2.0\n---\nDesign.\n"
    )
    cards = load_agent_cards(card_path)
    assert len(cards) == 1
    assert cards[0].agent_data["config"].image_gen == "xai.grok-imagine-image-2.0"


def test_no_home_fallback_jpeg_extension_and_rgba_preview_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.setattr(
        image_gen, "_resolve_backend_endpoint", lambda *_a, **_kw: ("https://x", {})
    )
    settings = Settings()
    settings._fast_agent_no_home = True

    resolved = resolve_image_gen_model("codex", settings=settings)
    rt = ImageGenRuntime(resolved, settings=settings, working_dir=tmp_path)
    out_dir = rt._default_output_dir()
    assert out_dir.name == "fast-agent-images"

    paths = rt._resolve_output_paths(
        output_path="photo.jpeg",
        count=1,
        ext=".jpg",
        prompt="photo",
    )
    assert paths == [tmp_path / "photo.jpeg"]

    noisy_rgba = Image.frombytes("RGBA", (1536, 1024), os.urandom(1536 * 1024 * 4))
    buf = BytesIO()
    noisy_rgba.save(buf, format="PNG")
    raw_png = buf.getvalue()
    assert len(raw_png) > image_gen.DEFAULT_PREVIEW_MAX_BYTES

    preview = image_gen._build_preview_block(raw_png, "image/png")
    if preview is not None:
        assert len(base64.b64decode(preview.data)) <= image_gen.DEFAULT_PREVIEW_MAX_BYTES


def test_mcp_agent_image_gen_allows_local_shell_and_rejects_remote_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.mcp_agent import McpAgent
    from fast_agent.context import Context
    from fast_agent.core.exceptions import AgentConfigError
    from fast_agent.core.logging.logger import get_logger
    from fast_agent.llm.internal.passthrough import PassthroughLLM
    from fast_agent.tools.local_shell_executor import LocalEnvironment

    monkeypatch.setattr(
        image_gen, "_resolve_backend_endpoint", lambda *_a, **_kw: ("https://x", {})
    )
    settings = Settings()
    context = Context(config=settings)
    llm = PassthroughLLM(context=context)

    local_agent = McpAgent(
        AgentConfig(name="local_img", shell=True, image_gen="openai.gpt-image-2.6-flare"),
        context=context,
        shell_environment=LocalEnvironment(
            logger=get_logger(__name__),
            working_directory=tmp_path,
        ),
    )
    local_agent._configure_image_gen_runtime(llm)
    tool_names = [t.name for t in local_agent._image_gen_runtime_tools()]
    assert tool_names == ["generate_image", "edit_image"]

    class _RemoteEnvironment:
        pass

    remote_agent = McpAgent(
        AgentConfig(name="remote_img", image_gen="openai.gpt-image-2.6-flare"),
        context=context,
        shell_environment=cast("ShellEnvironment", _RemoteEnvironment()),
    )
    with pytest.raises(AgentConfigError, match="only supported in local environments"):
        remote_agent._configure_image_gen_runtime(llm)


@pytest.mark.asyncio
async def test_image_gen_emits_stage_progress_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.event_progress import ProgressAction

    monkeypatch.setattr(
        image_gen, "_resolve_backend_endpoint", lambda *_a, **_kw: ("https://x", {})
    )
    resolved = resolve_image_gen_model("codexresponses.gpt-image-2.6-flare", settings=Settings())
    rt = ImageGenRuntime(resolved, working_dir=tmp_path, agent_name="artist")

    emitted: list[dict[str, Any]] = []

    def _capture_info(_msg: str, **kwargs: Any) -> None:
        data = kwargs.get("data")
        if isinstance(data, dict):
            emitted.append(data)

    monkeypatch.setattr(image_gen.logger, "info", _capture_info)

    async def _fake_generate(**_kw: Any) -> list[image_gen._GeneratedImageData]:
        return [image_gen._GeneratedImageData(data=_png_bytes(), mime_type="image/png")]

    monkeypatch.setattr(rt, "_call_openai_generate", _fake_generate)

    result = await rt.call_tool(
        "generate_image",
        {"prompt": "a lighthouse"},
        tool_use_id="call_1",
    )
    assert not result.is_error
    assert [e["progress_action"] for e in emitted] == [
        ProgressAction.CALLING_TOOL,
        ProgressAction.TOOL_PROGRESS,
        ProgressAction.TOOL_PROGRESS,
    ]
    assert emitted[0]["agent_name"] == "artist"
    assert emitted[0]["tool_use_id"] == "call_1"
    assert "Generating with gpt-image-2.6-flare" in emitted[0]["details"]
    assert "Saving 1 image" in emitted[1]["details"]
    assert emitted[2]["tool_terminal"] is True
    assert emitted[2]["tool_state"] == "completed"

    emitted.clear()

    async def _failing_generate(**_kw: Any) -> list[image_gen._GeneratedImageData]:
        raise RuntimeError("boom")

    monkeypatch.setattr(rt, "_call_openai_generate", _failing_generate)

    err = await rt.call_tool("generate_image", {"prompt": "fail"}, tool_use_id="call_2")
    assert err.is_error
    assert len(emitted) == 2
    assert emitted[0]["progress_action"] == ProgressAction.CALLING_TOOL
    assert emitted[1]["tool_terminal"] is True
    assert emitted[1]["tool_state"] == "failed"

    emitted.clear()

    async def _cancelled_generate(**_kw: Any) -> list[image_gen._GeneratedImageData]:
        raise asyncio.CancelledError()

    monkeypatch.setattr(rt, "_call_openai_generate", _cancelled_generate)
    with pytest.raises(asyncio.CancelledError):
        await rt.call_tool("generate_image", {"prompt": "cancel"}, tool_use_id="call_3")
    assert len(emitted) == 2
    assert emitted[0]["progress_action"] == ProgressAction.CALLING_TOOL
    assert emitted[1]["tool_terminal"] is True
    assert emitted[1]["tool_state"] == "cancelled"


@pytest.mark.asyncio
async def test_image_gen_ptc_stub_tool_handler_working_dir_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.mcp_agent import McpAgent
    from fast_agent.context import Context
    from fast_agent.core.logging.logger import get_logger
    from fast_agent.llm.internal.passthrough import PassthroughLLM
    from fast_agent.llm.provider_key_manager import ProviderKeyManager
    from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler
    from fast_agent.ptc.stubs import build_functions, render_stubs
    from fast_agent.tools.local_shell_executor import LocalEnvironment

    # 1. PTC stub guidance + xAI output_path description
    xai_resolved = resolve_image_gen_model(
        "xai.grok-imagine-image-2.0",
        settings=Settings(),
        validate_credentials=False,
    )
    xai_tools = build_image_gen_tools(xai_resolved)
    xai_out_desc = xai_tools[0].input_schema["properties"]["output_path"]["description"]
    assert "default .png" not in xai_out_desc
    assert "Provider chooses" in xai_out_desc
    ptc_stub = render_stubs(build_functions(xai_tools))
    assert "structured_content" in ptc_stub

    # 2. Expired Codex OAuth in ProviderKeyManager falls back cleanly to OPENAI_API_KEY
    def _flaky_key(provider_name: str, _cfg: dict[str, Any]) -> str | None:
        if provider_name == "codexresponses":
            raise RuntimeError("expired refresh token")
        if provider_name == "openai":
            return "sk-test"
        return None

    monkeypatch.setattr(ProviderKeyManager, "get_optional_api_key", staticmethod(_flaky_key))
    auto = resolve_image_gen_model(
        None,
        settings=Settings(),
        main_provider=None,
        validate_credentials=False,
    )
    assert auto.backend == "openai"

    # 3. Live LocalEnvironment.working_directory() is used over config.cwd
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        image_gen, "_resolve_backend_endpoint", lambda *_a, **_kw: ("https://x", {})
    )
    context = Context(config=Settings())
    llm = PassthroughLLM(context=context)
    env = LocalEnvironment(logger=get_logger(__name__), working_directory=workspace)
    agent = McpAgent(
        AgentConfig(name="ws_img", cwd=tmp_path, image_gen="openai.gpt-image-2.6-flare"),
        context=context,
        shell_environment=env,
    )
    agent._configure_image_gen_runtime(llm)
    assert agent._image_gen_runtime is not None
    assert agent._image_gen_runtime._resolve_input_path("circle.png") == workspace / "circle.png"

    # 4. ToolExecutionHandler receives start / progress / complete (ACP visibility)
    events: list[tuple[str, Any]] = []

    class _FakeHandler(NoOpToolExecutionHandler):
        async def on_tool_start(
            self,
            tool_name: str,
            server_name: str,
            arguments: dict[str, Any] | None,
            tool_use_id: str | None = None,
        ) -> str:
            events.append(("start", (tool_name, server_name, tool_use_id)))
            return "acp_call_1"

        async def on_tool_progress(
            self,
            tool_call_id: str,
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            events.append(("progress", (tool_call_id, progress, total, message)))

        async def on_tool_complete(
            self,
            tool_call_id: str,
            success: bool,
            content: Any,
            error: str | None,
        ) -> None:
            events.append(("complete", (tool_call_id, success, error)))

    async def _fake_gen(**_kw: Any) -> list[image_gen._GeneratedImageData]:
        return [image_gen._GeneratedImageData(data=_png_bytes(), mime_type="image/png")]

    monkeypatch.setattr(agent._image_gen_runtime, "_call_openai_generate", _fake_gen)
    res = await agent._image_gen_runtime.call_tool(
        "generate_image",
        {"prompt": "star", "output_path": "star.png"},
        tool_use_id="tu_1",
        tool_handler=_FakeHandler(),
    )
    assert not res.is_error
    assert (workspace / "star.png").exists()
    assert [e[0] for e in events] == ["start", "progress", "progress", "complete"]
    assert events[-1] == ("complete", ("acp_call_1", True, None))

    # 5. Pre-commit _validate_llm_attachment rejects invalid image_gen before session/LLM commit
    bad_agent = McpAgent(
        AgentConfig(name="bad_img", image_gen="unsupported.model"),
        context=context,
    )
    with pytest.raises(ValueError, match="Unrecognized image generation model"):
        bad_agent._validate_llm_attachment(llm, for_new_session=True)


def test_copilot_fallback_chain_and_startup_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.llm.provider_key_manager import ProviderKeyManager
    from fast_agent.ui.prompt import input_startup

    # Copilot main model falls back automatically: codexresponses -> openai -> xai
    for available, expected_backend, expected_model in [
        ({"codexresponses", "openai", "xai"}, "codexresponses", "gpt-image-2.6-flare"),
        ({"openai", "xai"}, "openai", "gpt-image-2.6-flare"),
        ({"xai"}, "xai", "grok-imagine-image-2.0"),
    ]:
        monkeypatch.setattr(
            ProviderKeyManager,
            "get_optional_api_key",
            staticmethod(lambda name, _cfg, avail=available: "tok" if name in avail else None),
        )
        resolved = resolve_image_gen_model(
            None,
            settings=Settings(),
            main_provider=Provider.COPILOT,
            validate_credentials=False,
        )
        assert resolved.backend == expected_backend
        assert resolved.model == expected_model

    # Startup banner shows image generation status + resolved model spec
    printed: list[str] = []
    monkeypatch.setattr(
        input_startup, "rich_print", lambda msg="", *_a, **_kw: printed.append(str(msg))
    )

    class _AgentWithImageGen:
        image_gen_model = "codexresponses.gpt-image-2.6-flare"

    class _AgentWithoutImageGen:
        image_gen_model = None

    class _Provider:
        def __init__(self, agents: dict[str, Any]) -> None:
            self._agents = agents

        def _agent(self, name: str) -> Any:
            return self._agents[name]

    provider = cast("Any", _Provider({"on": _AgentWithImageGen(), "off": _AgentWithoutImageGen()}))
    input_startup.show_image_gen_startup(agent_name="on", agent_provider=provider)
    input_startup.show_image_gen_startup(agent_name="off", agent_provider=provider)

    assert len(printed) == 1
    assert "Image generation enabled" in printed[0]
    assert "codexresponses.gpt-image-2.6-flare" in printed[0]


def test_resolve_google_nano_banana_specs_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.llm.provider_key_manager import ProviderKeyManager

    for spec, expected_model, expected_res in [
        ("google", "gemini-nano-banana-2.1", None),
        ("gemini", "gemini-nano-banana-2.1", None),
        ("nano-banana", "gemini-nano-banana-2.1", None),
        ("google.gemini-nano-banana-2.1", "gemini-nano-banana-2.1", None),
        ("gemini-nano-banana-2.1?resolution=4k", "gemini-nano-banana-2.1", "4k"),
        ("gemini-3.1-flash-image?resolution=2k", "gemini-3.1-flash-image", "2k"),
    ]:
        resolved = resolve_image_gen_model(
            spec,
            settings=Settings(),
            validate_credentials=False,
        )
        assert resolved.backend == "google"
        assert resolved.model == expected_model
        assert resolved.resolution == expected_res

    # Auto-selects Google Nano Banana when main provider is Google
    auto_google = resolve_image_gen_model(
        None,
        settings=Settings(),
        main_provider=Provider.GOOGLE,
        validate_credentials=False,
    )
    assert auto_google.backend == "google"
    assert auto_google.model == "gemini-nano-banana-2.1"

    # Falls back to Google when only GOOGLE_API_KEY is available
    monkeypatch.setattr(
        ProviderKeyManager,
        "get_optional_api_key",
        staticmethod(lambda name, _cfg: "gkey" if name == "google" else None),
    )
    monkeypatch.setattr(image_gen, "_has_codex_credentials", lambda _s: False)
    fallback = resolve_image_gen_model(
        None,
        settings=Settings(),
        main_provider=Provider.ANTHROPIC,
        validate_credentials=False,
    )
    assert fallback.backend == "google"
    assert fallback.model == "gemini-nano-banana-2.1"

    # 4k resolution is rejected for xAI Imagine
    with pytest.raises(ValueError, match="Unsupported image resolution '4k'"):
        resolve_image_gen_model(
            "xai.grok-imagine-image-2.0?resolution=4k",
            settings=Settings(),
            validate_credentials=False,
        )

    # Tool schemas reflect Google aspect ratios and 14-image edit limit
    tools = {t.name: t for t in build_image_gen_tools(auto_google)}
    gen_props = tools["generate_image"].input_schema["properties"]
    assert gen_props["aspect_ratio"]["enum"] == list(image_gen.GOOGLE_ASPECT_RATIOS)
    assert "quality" not in gen_props
    assert "transparent" not in gen_props
    edit_props = tools["edit_image"].input_schema["properties"]
    assert edit_props["images"]["maxItems"] == image_gen.MAX_GOOGLE_EDIT_IMAGES


@pytest.mark.asyncio
async def test_google_nano_banana_generate_and_edit_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64

    import httpx

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    img_bytes = _png_bytes((32, 32))
    img_b64 = base64.b64encode(img_bytes).decode("ascii")
    captured: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append((str(request.url), body, dict(request.headers)))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Generated banana."},
                                {"inlineData": {"mimeType": "image/png", "data": img_b64}},
                            ]
                        }
                    }
                ]
            },
        )

    real_async_client = httpx.AsyncClient

    def _mock_client(*_args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_async_client(transport=httpx.MockTransport(_handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client)

    resolved = resolve_image_gen_model(
        "google.gemini-nano-banana-2.1?resolution=2k",
        settings=Settings(),
        validate_credentials=True,
    )
    runtime = ImageGenRuntime(
        resolved=resolved,
        settings=Settings(),
        working_dir=tmp_path,
    )

    gen_res = await runtime.call_tool(
        "generate_image",
        {
            "prompt": "a bright yellow banana",
            "aspect_ratio": "16:9",
            "n": 2,
            "output_path": "banana.png",
        },
    )
    assert not gen_res.is_error
    assert len(captured) == 2
    url, body, headers = captured[0]
    assert url.endswith("/models/gemini-nano-banana-2.1:generateContent")
    assert headers["x-goog-api-key"] == "test-google-key"
    assert body["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9", "imageSize": "2K"}
    assert (tmp_path / "banana-1.png").exists()
    assert (tmp_path / "banana-2.png").exists()

    edit_res = await runtime.call_tool(
        "edit_image",
        {
            "images": [str(tmp_path / "banana-1.png")],
            "prompt": "make it neon green",
            "output_path": "banana_green.png",
        },
    )
    assert not edit_res.is_error
    assert len(captured) == 3
    edit_parts = captured[2][1]["contents"][0]["parts"]
    assert "inlineData" in edit_parts[0]
    assert edit_parts[0]["inlineData"]["mimeType"] == "image/png"
    assert edit_parts[1] == {"text": "make it neon green"}
    assert (tmp_path / "banana_green.png").exists()
