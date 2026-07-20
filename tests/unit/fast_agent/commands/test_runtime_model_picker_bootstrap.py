"""
Testing notes:

- This module owns runtime bootstrap behavior around when the model picker is
  shown, how initial model selection is resolved, and how last-used model state
  is persisted.
- Prefer seam tests through the bootstrap helpers over rebuilding picker or
  settings internals by hand.
- Exact picker option construction belongs in ui/test_model_picker*.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.cli.runtime.agent_setup import (
    _agent_config_defines_startup_model,
    _attach_cli_servers_to_selected_agent,
    _explicit_agent_cards_define_startup_model,
    _generic_model_prompt_default,
    _last_used_model_reference,
    _load_request_settings,
    _persist_model_picker_last_used_selection,
    _resolve_model_picker_initial_selection,
    _resolve_model_without_hardcoded_default,
    _select_model_from_picker,
    _select_startup_model_if_needed,
    _should_prompt_for_model_picker,
    _split_requested_models,
    run_agent_request,
)
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.config import Settings
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker import ModelPickerResult
from fast_agent.ui.model_picker_common import (
    LLAMACPP_PROVIDER_KEY,
    ProviderActivation,
    normalize_generic_model_spec,
)


def _picker_result(
    *,
    provider: str = "overlays",
    selected_model: str = "haikutiny",
    resolved_model: str | None = "haikutiny",
) -> ModelPickerResult:
    return ModelPickerResult(
        provider=provider,
        provider_available=True,
        selected_model=selected_model,
        resolved_model=resolved_model,
        source="curated",
        refer_to_docs=False,
        activation_action=None,
    )


def _make_request(
    *,
    config_path: str | None = None,
    message: str | None = None,
    prompt_file: str | None = None,
    agent_cards: list[str] | None = None,
    card_tools: list[str] | None = None,
    resume: str | None = None,
) -> AgentRunRequest:
    return AgentRunRequest(
        name="test",
        instruction="instruction",
        config_path=config_path,
        server_list=None,
        agent_cards=agent_cards,
        card_tools=card_tools,
        model=None,
        message=message,
        prompt_file=prompt_file,
        result_file=None,
        resume=resume,
        url_servers=None,
        stdio_servers=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        home=None,
        no_home=False,
        force_smart=False,
        shell_runtime=False,
        no_shell=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )


def test_should_prompt_for_model_picker_in_interactive_tty_startup() -> None:
    request = _make_request(message=None, prompt_file=None)

    assert _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


def test_should_not_prompt_for_model_picker_when_message_mode() -> None:
    request = _make_request(message="hello")

    assert not _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


def test_should_not_prompt_for_model_picker_when_prompt_file_mode() -> None:
    request = _make_request(prompt_file="prompt.txt")

    assert not _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


def test_should_prompt_for_model_picker_when_cards_present() -> None:
    request = _make_request(agent_cards=["cards/"])

    assert _should_prompt_for_model_picker(
        request,
        stdin_is_tty=True,
        stdout_is_tty=True,
    )


@pytest.mark.asyncio
async def test_startup_model_selection_skipped_when_resuming(monkeypatch) -> None:
    # On --resume the session snapshot owns the model; the startup picker must
    # not run (its selection would be overridden by hydration anyway). Instead
    # a distinctive model source is returned so the startup status notice names
    # session resumption as the source.
    async def _fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("model picker must not run during resume")

    monkeypatch.setattr(
        "fast_agent.cli.runtime.agent_setup._select_model_from_picker", _fail
    )
    request = _make_request(resume="__latest__")

    assert await _select_startup_model_if_needed(request) == "session resumption"
    assert request.model is None


@pytest.mark.asyncio
async def test_startup_model_selection_skipped_when_resuming_named_session(
    monkeypatch,
) -> None:
    async def _fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("model picker must not run during resume")

    monkeypatch.setattr(
        "fast_agent.cli.runtime.agent_setup._select_model_from_picker", _fail
    )
    request = _make_request(resume="session-123")

    assert await _select_startup_model_if_needed(request) == "session resumption"
    assert request.model is None


def test_attach_cli_servers_prefers_typed_default_agent_config() -> None:
    default_config = AgentConfig("primary", default=True, servers=["existing"])
    fallback_config = AgentConfig("fallback", servers=[])
    fast = SimpleNamespace(
        agents={
            "fallback": {"config": fallback_config},
            "primary": {"config": default_config},
        }
    )
    request = _make_request()
    request.server_list = ["existing", "from-cli"]

    _attach_cli_servers_to_selected_agent(fast, request)

    assert default_config.servers == ["existing", "from-cli"]
    assert fallback_config.servers == []


def test_attach_cli_servers_prefers_explicit_agent_over_default() -> None:
    default_config = AgentConfig("primary", default=True, servers=[])
    explicit_config = AgentConfig("target", servers=["existing"])
    fast = SimpleNamespace(
        agents={
            "primary": {"config": default_config},
            "target": {"config": explicit_config},
        }
    )
    request = _make_request()
    request.agent_name = "target"
    request.server_list = ["existing", "from-cli"]

    _attach_cli_servers_to_selected_agent(fast, request)

    assert default_config.servers == []
    assert explicit_config.servers == ["existing", "from-cli"]


def test_attach_cli_servers_skips_tool_only_fallback_agent() -> None:
    tool_config = AgentConfig("tool", tool_only=True, servers=[])
    runnable_config = AgentConfig("runnable", servers=[])
    fast = SimpleNamespace(
        agents={
            "tool": {"config": tool_config, "tool_only": True},
            "runnable": {"config": runnable_config},
        }
    )
    request = _make_request()
    request.server_list = ["from-cli"]

    _attach_cli_servers_to_selected_agent(fast, request)

    assert tool_config.servers == []
    assert runnable_config.servers == ["from-cli"]


def test_explicit_remote_agent_card_model_suppresses_startup_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.io import source_resolver

    def fake_read_text_source(source: str, *, label: str) -> str:
        assert source == "hf://buckets/evalstate/demo-bucket/ai-news-summary-card.md"
        assert label == "AgentCard URL"
        return "\n".join(
            [
                "---",
                "type: agent",
                "name: ai_news_summary",
                "model: passthrough",
                "---",
                "Summarize news.",
                "",
            ]
        )

    monkeypatch.setattr(source_resolver, "read_text_source", fake_read_text_source)
    request = _make_request(
        agent_cards=["hf://buckets/evalstate/demo-bucket/ai-news-summary-card.md"]
    )

    assert _explicit_agent_cards_define_startup_model(request) is True


def test_explicit_remote_agent_card_without_model_keeps_startup_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.io import source_resolver

    def fake_read_text_source(source: str, *, label: str) -> str:
        assert source == "hf://buckets/evalstate/demo-bucket/ai-news-summary-card.md"
        assert label == "AgentCard URL"
        return "\n".join(
            [
                "---",
                "type: agent",
                "name: ai_news_summary",
                "---",
                "Summarize news.",
                "",
            ]
        )

    monkeypatch.setattr(source_resolver, "read_text_source", fake_read_text_source)
    request = _make_request(
        agent_cards=["hf://buckets/evalstate/demo-bucket/ai-news-summary-card.md"]
    )

    assert _explicit_agent_cards_define_startup_model(request) is False


@pytest.mark.parametrize(
    ("model_references", "expected"),
    [
        (None, False),
        ({"system": {"fast": "passthrough"}}, True),
    ],
)
def test_explicit_remote_agent_card_model_reference_only_suppresses_when_available(
    monkeypatch: pytest.MonkeyPatch,
    model_references: dict[str, dict[str, str]] | None,
    expected: bool,
) -> None:
    from fast_agent.io import source_resolver

    def fake_read_text_source(source: str, *, label: str) -> str:
        assert source == "hf://buckets/evalstate/demo-bucket/ai-news-summary-card.md"
        assert label == "AgentCard URL"
        return "\n".join(
            [
                "---",
                "type: agent",
                "name: ai_news_summary",
                "model: $system.fast",
                "---",
                "Summarize news.",
                "",
            ]
        )

    monkeypatch.setattr(source_resolver, "read_text_source", fake_read_text_source)
    request = _make_request(
        agent_cards=["hf://buckets/evalstate/demo-bucket/ai-news-summary-card.md"]
    )

    assert (
        _explicit_agent_cards_define_startup_model(
            request,
            model_references=model_references,
        )
        is expected
    )


def test_resolve_model_without_hardcoded_default_returns_none_without_sources() -> None:
    previous = os.environ.pop("FAST_AGENT_MODEL", None)
    try:
        resolved_model = _resolve_model_without_hardcoded_default(
            model=None,
            config_default_model=None,
            model_references=None,
        )
    finally:
        if previous is not None:
            os.environ["FAST_AGENT_MODEL"] = previous

    assert resolved_model.model is None
    assert resolved_model.source is None


def test_resolve_model_without_hardcoded_default_prefers_config_default() -> None:
    previous = os.environ.pop("FAST_AGENT_MODEL", None)
    try:
        resolved_model = _resolve_model_without_hardcoded_default(
            model=None,
            config_default_model="openai.gpt-4.1-mini",
            model_references=None,
        )
    finally:
        if previous is not None:
            os.environ["FAST_AGENT_MODEL"] = previous

    assert resolved_model.model == "openai.gpt-4.1-mini"
    assert resolved_model.source == "config file"


def test_resolve_model_without_hardcoded_default_uses_environment_variable() -> None:
    previous = os.environ.get("FAST_AGENT_MODEL")
    os.environ["FAST_AGENT_MODEL"] = "responses.gpt-5-mini"
    try:
        resolved_model = _resolve_model_without_hardcoded_default(
            model=None,
            config_default_model=None,
            model_references=None,
        )
    finally:
        if previous is not None:
            os.environ["FAST_AGENT_MODEL"] = previous
        else:
            os.environ.pop("FAST_AGENT_MODEL", None)

    assert resolved_model.model == "responses.gpt-5-mini"
    assert resolved_model.source == "environment variable FAST_AGENT_MODEL"


@pytest.mark.asyncio
async def test_interactive_startup_prompts_when_system_default_is_only_last_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent import config as config_module

    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        (
            'default_model: "$system.default"\n'
            "model_references:\n"
            "  system:\n"
            "    last_used: claude-haiku-4-5\n"
        ),
        encoding="utf-8",
    )
    request = _make_request(config_path=str(config_path))
    captured: dict[str, object] = {}

    async def fake_select_model_from_picker(*args, **kwargs):
        del args
        captured.update(kwargs)
        return "gpt-4.1-mini"

    old_settings = config_module._settings
    previous_fast_agent_model = os.environ.pop("FAST_AGENT_MODEL", None)
    try:
        config_module._settings = None
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr(
            "fast_agent.cli.runtime.agent_setup._select_model_from_picker",
            fake_select_model_from_picker,
        )

        assert await _select_startup_model_if_needed(request) == "model picker"
    finally:
        if previous_fast_agent_model is not None:
            os.environ["FAST_AGENT_MODEL"] = previous_fast_agent_model
        config_module._settings = old_settings

    assert request.model == "gpt-4.1-mini"
    assert captured["initial_model_spec"] == "claude-haiku-4-5"
    assert captured["initial_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_interactive_startup_uses_explicit_system_default_without_picker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent import config as config_module

    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        (
            'default_model: "$system.default"\n'
            "model_references:\n"
            "  system:\n"
            "    default: claude-sonnet-4-5\n"
            "    last_used: claude-haiku-4-5\n"
        ),
        encoding="utf-8",
    )
    request = _make_request(config_path=str(config_path))

    async def fail_select_model_from_picker(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model picker must not run")

    old_settings = config_module._settings
    previous_fast_agent_model = os.environ.pop("FAST_AGENT_MODEL", None)
    try:
        config_module._settings = None
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr(
            "fast_agent.cli.runtime.agent_setup._select_model_from_picker",
            fail_select_model_from_picker,
        )

        assert await _select_startup_model_if_needed(request) is None
    finally:
        if previous_fast_agent_model is not None:
            os.environ["FAST_AGENT_MODEL"] = previous_fast_agent_model
        config_module._settings = old_settings

    assert request.model is None


@pytest.mark.asyncio
async def test_select_model_from_picker_preserves_overlay_token_when_resolved_model_is_present(
    monkeypatch,
) -> None:
    request = _make_request()

    async def fake_run_model_picker_async(**kwargs):
        del kwargs
        return _picker_result()

    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )

    selected = await _select_model_from_picker(request, config_payload={})

    assert selected == "haikutiny"


@pytest.mark.asyncio
async def test_select_model_from_picker_refreshes_expired_oauth_provider(
    monkeypatch,
) -> None:
    request = _make_request()
    refreshes = 0
    logins = 0

    async def fake_run_model_picker_async(**kwargs):
        del kwargs
        return ModelPickerResult(
            provider=Provider.XAI.config_name,
            provider_available=False,
            selected_model="xai.grok-4",
            resolved_model="xai.grok-4",
            source="curated",
            refer_to_docs=False,
            activation_action=ProviderActivation(Provider.XAI),
        )

    def login() -> None:
        nonlocal logins
        logins += 1

    def access_token() -> str:
        nonlocal refreshes
        refreshes += 1
        return "refreshed-token"

    handler = SimpleNamespace(
        display_name="xAI",
        status=lambda: {"present": True, "expired": True},
        access_token=access_token,
        login=login,
    )
    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )
    monkeypatch.setattr(
        "fast_agent.auth.providers.get_oauth_provider",
        lambda provider: handler,
    )

    selected = await _select_model_from_picker(request, config_payload={})

    assert selected == "xai.grok-4"
    assert refreshes == 1
    assert logins == 0


@pytest.mark.asyncio
async def test_select_model_from_picker_reauthenticates_after_refresh_is_cleared(
    monkeypatch,
) -> None:
    request = _make_request()
    logins = 0

    async def fake_run_model_picker_async(**kwargs):
        del kwargs
        return ModelPickerResult(
            provider=Provider.XAI.config_name,
            provider_available=False,
            selected_model="xai.grok-4",
            resolved_model="xai.grok-4",
            source="curated",
            refer_to_docs=False,
            activation_action=ProviderActivation(Provider.XAI),
        )

    def login() -> None:
        nonlocal logins
        logins += 1

    handler = SimpleNamespace(
        display_name="xAI",
        status=lambda: {"present": True, "expired": True},
        access_token=lambda: None,
        login=login,
    )
    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )
    monkeypatch.setattr(
        "fast_agent.auth.providers.get_oauth_provider",
        lambda provider: handler,
    )

    selected = await _select_model_from_picker(request, config_payload={})

    assert selected == "xai.grok-4"
    assert logins == 1


@pytest.mark.asyncio
async def test_select_model_from_picker_passes_config_start_path(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "project" / "fast-agent.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("default_model: haikutiny\n", encoding="utf-8")
    request = _make_request(config_path=str(config_path))
    captured_kwargs: dict[str, object] = {}

    async def fake_run_model_picker_async(**kwargs):
        captured_kwargs.update(kwargs)
        return _picker_result()

    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )

    selected = await _select_model_from_picker(request, config_payload={})

    assert selected == "haikutiny"
    assert captured_kwargs["start_path"] == config_path.parent


@pytest.mark.asyncio
async def test_select_model_from_picker_can_import_llamacpp_overlay(monkeypatch) -> None:
    request = _make_request()
    captured_import_kwargs: dict[str, object] = {}

    async def fake_run_model_picker_async(**kwargs):
        del kwargs
        return _picker_result(
            provider=LLAMACPP_PROVIDER_KEY,
            selected_model="llamacpp.__import__",
            resolved_model=None,
        )

    async def fake_import_llamacpp_overlay_from_default_url(**kwargs):
        captured_import_kwargs.update(kwargs)
        return "llamacpp-qwen"

    monkeypatch.setattr(
        "fast_agent.ui.model_picker.run_model_picker_async",
        fake_run_model_picker_async,
    )
    monkeypatch.setattr(
        "fast_agent.cli.commands.model.import_llamacpp_overlay_from_default_url",
        fake_import_llamacpp_overlay_from_default_url,
    )

    selected = await _select_model_from_picker(request, config_payload={})

    assert selected == "llamacpp-qwen"
    assert isinstance(captured_import_kwargs["start_path"], Path)
    assert captured_import_kwargs["home"] is not None


def test_normalize_generic_model_spec_adds_generic_prefix_when_missing() -> None:
    assert normalize_generic_model_spec("llama3.2") == "generic.llama3.2"


def test_normalize_generic_model_spec_preserves_explicit_provider_prefix() -> None:
    assert normalize_generic_model_spec("generic.llama3.2:latest") == "generic.llama3.2:latest"
    assert normalize_generic_model_spec("openai/gpt-4.1") == "openai/gpt-4.1"


def test_normalize_generic_model_spec_returns_none_for_blank_input() -> None:
    assert normalize_generic_model_spec("   ") is None


def test_generic_model_prompt_default_strips_generic_prefix() -> None:
    assert _generic_model_prompt_default("generic.llama3.2") == "llama3.2"


@pytest.mark.parametrize(
    ("initial_model_spec", "expected"),
    [
        ("   ", "llama3.2"),
        ("  generic.qwen3  ", "qwen3"),
    ],
)
def test_generic_model_prompt_default_normalizes_initial_model_spec(
    initial_model_spec: str,
    expected: str,
) -> None:
    assert _generic_model_prompt_default(initial_model_spec) == expected


def test_generic_model_prompt_default_ignores_non_generic_provider_spec() -> None:
    assert _generic_model_prompt_default("responses.gpt-5-mini") == "llama3.2"


def test_resolve_model_picker_initial_selection_uses_last_used_alias() -> None:
    initial_selection = _resolve_model_picker_initial_selection(
        settings=Settings(
            model_references={
                "system": {
                    "last_used": "claude-haiku-4-5",
                }
            }
        )
    )

    assert initial_selection.provider == "anthropic"
    assert initial_selection.model_spec == "claude-haiku-4-5"


def test_resolve_model_picker_initial_selection_uses_vertex_group_for_anthropic_vertex() -> None:
    initial_selection = _resolve_model_picker_initial_selection(
        settings=Settings(
            model_references={
                "system": {
                    "last_used": "anthropic-vertex.claude-sonnet-4-6",
                }
            }
        )
    )

    assert initial_selection.provider == Provider.ANTHROPIC_VERTEX.config_name
    assert initial_selection.model_spec == "anthropic-vertex.claude-sonnet-4-6"


def test_resolve_model_picker_initial_selection_preserves_overlay_alias(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    overlays_dir = home / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "haikutiny.yaml").write_text(
        (
            "name: haikutiny\n"
            "provider: anthropic\n"
            "model: claude-haiku-4-5\n"
            "metadata:\n"
            "  context_window: 8192\n"
            "  max_output_tokens: 1024\n"
        ),
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        os.chdir(tmp_path)
        initial_selection = _resolve_model_picker_initial_selection(
            settings=Settings(
                home=str(home),
                model_references={
                    "system": {
                        "last_used": "haikutiny",
                    }
                },
            )
        )
    finally:
        os.chdir(previous_cwd)
        if previous_home is not None:
            os.environ["FAST_AGENT_HOME"] = previous_home

    assert initial_selection.provider == "overlays"
    assert initial_selection.model_spec == "haikutiny"


def test_resolve_model_picker_initial_selection_uses_config_relative_overlay_dir(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    project_dir = workspace / "project"
    home = project_dir / ".fast-agent"
    config_path = project_dir / "fast-agent.yaml"
    overlays_dir = home / "model-overlays"
    overlays_dir.mkdir(parents=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("home: .fast-agent\n", encoding="utf-8")
    (overlays_dir / "haikutiny.yaml").write_text(
        ("name: haikutiny\nprovider: anthropic\nmodel: claude-haiku-4-5\n"),
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        os.chdir(tmp_path)
        settings = Settings(
            home=".fast-agent",
            model_references={"system": {"last_used": "haikutiny"}},
        )
        settings._config_file = str(config_path.resolve())

        initial_selection = _resolve_model_picker_initial_selection(settings=settings)
    finally:
        os.chdir(previous_cwd)
        if previous_home is not None:
            os.environ["FAST_AGENT_HOME"] = previous_home

    assert initial_selection.provider == "overlays"
    assert initial_selection.model_spec == "haikutiny"


def test_load_request_settings_refreshes_stale_cached_settings(tmp_path: Path) -> None:
    from fast_agent import config as config_module

    workspace = tmp_path / "workspace"
    home = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    home.mkdir(parents=True)
    (home / "fast-agent.yaml").write_text(
        "model_references:\n  system:\n    last_used: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    old_settings = config_module._settings
    previous_cwd = Path.cwd()
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        config_module._settings = Settings(
            model_references={"system": {"last_used": "claude-haiku-4-5"}}
        )
        os.chdir(workspace)
        settings = _load_request_settings(_make_request())
    finally:
        os.chdir(previous_cwd)
        if previous_home is not None:
            os.environ["FAST_AGENT_HOME"] = previous_home
        config_module._settings = old_settings

    assert settings.model_references["system"]["last_used"] == "gpt-4.1-mini"
    assert settings._config_file == str((home / "fast-agent.yaml").resolve())


def test_agent_config_defines_startup_model_normalizes_model_value() -> None:
    assert _agent_config_defines_startup_model(
        SimpleNamespace(model="  passthrough  "),
        model_references=None,
    )
    assert not _agent_config_defines_startup_model(
        SimpleNamespace(model="   "),
        model_references=None,
    )


def test_agent_config_does_not_treat_unpinned_system_default_as_interactive_model() -> None:
    assert not _agent_config_defines_startup_model(
        SimpleNamespace(model="$system.default"),
        model_references={"system": {"last_used": "claude-haiku-4-5"}},
        system_default_requires_explicit=True,
    )
    assert _agent_config_defines_startup_model(
        SimpleNamespace(model="$system.default"),
        model_references={
            "system": {
                "default": "claude-sonnet-4-5",
                "last_used": "claude-haiku-4-5",
            }
        },
        system_default_requires_explicit=True,
    )


def test_last_used_model_reference_normalizes_blank_values() -> None:
    assert _last_used_model_reference({"system": {"last_used": "  gpt-4.1-mini  "}}) == (
        "gpt-4.1-mini"
    )
    assert _last_used_model_reference({"system": {"last_used": "   "}}) is None


def test_split_requested_models_normalizes_comma_separated_values() -> None:
    assert _split_requested_models(" opus, , sonnet ,") == ["opus", "sonnet"]


def test_persist_model_picker_last_used_selection_writes_env_overlay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    home = workspace / ".fast-agent"
    home.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        request = _make_request()
        request.home = home
        settings = Settings(home=str(home))

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)

    assert persisted is True

    with open(home / "fast-agent.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"
    assert settings.model_references["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_uses_request_home(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    home = workspace / ".custom-env"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        request = _make_request()
        request.home = home
        settings = Settings()

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)

    assert persisted is True

    with open(home / "fast-agent.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_uses_runtime_cwd_env_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "nested"
    workspace.mkdir(parents=True)
    nested.mkdir(parents=True)
    (workspace / "fast-agent.yaml").write_text("default_model: null\n", encoding="utf-8")

    previous_cwd = Path.cwd()
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        os.chdir(nested)
        request = _make_request()
        settings = Settings()

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)
        if previous_home is not None:
            os.environ["FAST_AGENT_HOME"] = previous_home

    assert persisted is True
    assert not (workspace / ".fast-agent" / "fast-agent.yaml").exists()

    with open(nested / ".fast-agent" / "fast-agent.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_creates_env_overlay_on_first_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    home = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        request = _make_request()
        request.home = home
        settings = Settings(home=str(home))

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)

    assert persisted is True

    with open(home / "fast-agent.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_updates_loaded_env_overlay_in_place(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    home = workspace / ".fast-agent"
    config_path = home / "fast-agent.yaml"
    workspace.mkdir(parents=True)
    home.mkdir(parents=True)
    config_path.write_text(
        "model_references:\n  system:\n    last_used: google.gemini-3.1-pro-preview\n",
        encoding="utf-8",
    )

    previous_cwd = Path.cwd()
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        os.chdir(workspace)
        request = _make_request()
        settings = _load_request_settings(request)

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)
        if previous_home is not None:
            os.environ["FAST_AGENT_HOME"] = previous_home

    assert persisted is True
    assert not (home / ".fast-agent" / "fast-agent.yaml").exists()

    with open(config_path, "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_persist_model_picker_last_used_selection_respects_no_home(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    home = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        request = _make_request()
        request.home = home
        request.no_home = True
        settings = Settings(home=str(home))

        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
    finally:
        os.chdir(previous_cwd)

    assert persisted is False
    assert not (home / "fast-agent.yaml").exists()


def test_persist_model_picker_last_used_selection_writes_explicit_config_file(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "config-root"
    workspace = tmp_path / "workspace"
    config_root.mkdir(parents=True)
    workspace.mkdir(parents=True)

    config_path = config_root / "fast-agent.yaml"
    config_path.write_text("default_model: claude-haiku-4-5\n", encoding="utf-8")

    request = _make_request(config_path=str(config_path))
    previous_cwd = Path.cwd()
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        os.chdir(workspace)
        settings = _load_request_settings(request)
        persisted = _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec="gpt-4.1-mini",
        )
        reloaded = _load_request_settings(request)
    finally:
        os.chdir(previous_cwd)
        if previous_home is not None:
            os.environ["FAST_AGENT_HOME"] = previous_home

    assert persisted is True
    assert not (workspace / ".fast-agent" / "fast-agent.yaml").exists()

    with open(config_path, "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"
    initial_selection = _resolve_model_picker_initial_selection(settings=reloaded)
    assert initial_selection.provider == "openai"
    assert initial_selection.model_spec == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_run_agent_request_persists_and_reloads_last_used_for_shell_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fast_agent
    from fast_agent import config as config_module

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    request = _make_request()
    request.shell_runtime = True

    async def fake_select_model_from_picker(*args, **kwargs) -> str:
        del args, kwargs
        return "gpt-4.1-mini"

    class _AbortFastAgent:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("stop-after-persist")

    old_settings = config_module._settings
    previous_cwd = Path.cwd()
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        config_module._settings = Settings(
            model_references={"system": {"last_used": "claude-haiku-4-5"}}
        )
        os.chdir(workspace)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr(
            "fast_agent.cli.runtime.agent_setup._select_model_from_picker",
            fake_select_model_from_picker,
        )
        monkeypatch.setattr(fast_agent, "FastAgent", _AbortFastAgent)

        with pytest.raises(RuntimeError, match="stop-after-persist"):
            await run_agent_request(request)

        config_module._settings = None
        settings = _load_request_settings(_make_request())
    finally:
        os.chdir(previous_cwd)
        if previous_home is not None:
            os.environ["FAST_AGENT_HOME"] = previous_home
        config_module._settings = old_settings

    config_path = workspace / ".fast-agent" / "fast-agent.yaml"
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"
    initial_selection = _resolve_model_picker_initial_selection(settings=settings)
    assert initial_selection.provider == "openai"
    assert initial_selection.model_spec == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_run_agent_request_uses_last_used_for_noninteractive_startup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import fast_agent
    from fast_agent import config as config_module

    workspace = tmp_path / "workspace"
    home = workspace / ".cdx"
    home.mkdir(parents=True)
    (home / "fast-agent.yaml").write_text(
        "default_model: null\nmodel_references:\n  system:\n    last_used: claude-haiku-4-5\n",
        encoding="utf-8",
    )

    request = _make_request()
    request.mode = "serve"
    request.transport = "acp"
    request.home = home

    class _AbortFastAgent:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("stop-after-model-resolution")

    old_settings = config_module._settings
    previous_cwd = Path.cwd()
    previous_home = os.environ.get("FAST_AGENT_HOME")
    try:
        config_module._settings = None
        os.chdir(workspace)
        os.environ["FAST_AGENT_HOME"] = str(home)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        monkeypatch.setattr(fast_agent, "FastAgent", _AbortFastAgent)

        with pytest.raises(RuntimeError, match="stop-after-model-resolution"):
            await run_agent_request(request)
    finally:
        os.chdir(previous_cwd)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home
        config_module._settings = old_settings

    assert request.model == "claude-haiku-4-5"
