"""HF route: inline images are fitted to the router's request-size limit at the request boundary."""

import base64
import functools
import json
import random
from collections.abc import Callable
from io import BytesIO
from typing import Any

import httpx2
import pytest
from mcp.types import CallToolResult, ImageContent, TextContent
from openai import APIStatusError
from PIL import Image

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_openai import (
    OpenAILLM,
    _OpenAICompletionRequest,
    _OpenAICompletionResponse,
)
from fast_agent.llm.provider.request_images import (
    IMAGE_OMITTED_PLACEHOLDER,
    request_bytes,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

_UNLIMITED = HuggingFaceSettings(
    image_max_dimension=None, image_reencode_min_bytes=None, max_request_bytes=None
)


def _llm(settings: HuggingFaceSettings | None = None) -> HuggingFaceLLM:
    llm = ModelFactory.create_factory("hf.zai-org/GLM-5.3-Flash:zai-org")(
        agent=LlmAgent(AgentConfig(name="hf-images")),
        context=Context(config=Settings(hf=settings or HuggingFaceSettings())),
    )
    assert isinstance(llm, HuggingFaceLLM)
    return llm


@functools.lru_cache(maxsize=None)
def _image_b64(size: tuple[int, int], noisy: bool, alpha: bool, seed: int) -> str:
    """Build deterministic PNG fixture bytes; noise stays large, flat colour stays tiny."""
    if noisy:
        width, height = size
        total = width * height * 3
        chunk = random.Random(seed).randbytes(min(total, 65_536))
        raw = (chunk * ((total // len(chunk)) + 1))[:total]
        image = Image.frombytes("RGB", size, raw)
        if alpha:
            image = image.convert("RGBA")
    else:
        image = Image.new("RGBA" if alpha else "RGB", size, (40, 100, 180))
    if alpha:
        image.putalpha(128)
    output = BytesIO()
    # Uncompressed stored-block PNG for noisy inputs avoids expensive zlib passes
    # over incompressible data while remaining a standard, Pillow-decodable PNG.
    image.save(output, format="PNG", compress_level=0 if noisy else 1)
    return base64.b64encode(output.getvalue()).decode()


def _image(
    size: tuple[int, int],
    *,
    noisy: bool = True,
    alpha: bool = False,
    seed: int = 0,
) -> ImageContent:
    return ImageContent(
        type="image",
        mime_type="image/png",
        data=_image_b64(size, noisy, alpha, seed),
    )


def _request(llm: HuggingFaceLLM, *images: ImageContent) -> _OpenAICompletionRequest:
    history = [
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="look")]),
        PromptMessageExtended(
            role="user",
            tool_results={
                f"call-{index}": CallToolResult(content=[image])
                for index, image in enumerate(images)
            },
        ),
    ]
    return llm._openai_completion_request(
        llm._convert_extended_messages_to_provider(history), None, None
    )


def _image_urls(arguments: dict[str, Any]) -> list[str]:
    return [
        part["image_url"]["url"]
        for message in arguments["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]


def _placeholders(arguments: dict[str, Any]) -> int:
    return json.dumps(arguments).count(IMAGE_OMITTED_PLACEHOLDER)


def _decoded(url: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_small_images_pass_through_byte_identical() -> None:
    llm = _llm()
    request = _request(llm, _image((64, 48)), _image((800, 600), noisy=False))
    fitted, report = llm._fit_request_images(request, llm._image_budget())
    assert not report.changed
    assert fitted.arguments == request.arguments


def test_oversized_image_is_downscaled_and_history_is_untouched() -> None:
    llm = _llm()
    request = _request(llm, _image((4000, 1000)))
    before = json.dumps(request.arguments)
    fitted, report = llm._fit_request_images(request, llm._image_budget())
    (url,) = _image_urls(fitted.arguments)
    with _decoded(url) as image:
        assert image.size == (2048, 512)
    assert report.resized == 1
    assert json.dumps(request.arguments) == before
    assert fitted.messages is request.messages


def test_heavy_opaque_png_is_reencoded_smaller() -> None:
    llm = _llm()
    request = _request(llm, _image((1000, 1000)))
    (original,) = _image_urls(request.arguments)
    fitted, report = llm._fit_request_images(request, llm._image_budget())
    (url,) = _image_urls(fitted.arguments)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) < len(original)
    assert report.reencoded == 1


def test_transparency_is_preserved() -> None:
    llm = _llm()
    request = _request(llm, _image((3000, 3000), noisy=False, alpha=True))
    fitted, _ = llm._fit_request_images(request, llm._image_budget())
    (url,) = _image_urls(fitted.arguments)
    assert url.startswith("data:image/png;base64,")
    with _decoded(url) as image:
        assert image.mode == "RGBA"
        assert max(image.size) == 2048


def test_budget_omits_oldest_images_first_and_fits() -> None:
    llm = _llm(_UNLIMITED.model_copy(update={"max_request_bytes": 1_500_000}))
    images = [_image((600, 600), seed=i) for i in range(5)]  # ~1.4 MB base64 each
    request = _request(llm, *images)
    fitted, report = llm._fit_request_images(request, llm._image_budget())
    assert request_bytes(fitted.arguments) <= 1_500_000
    assert report.omitted == _placeholders(fitted.arguments) > 0
    kept = _image_urls(fitted.arguments)
    assert kept == _image_urls(request.arguments)[report.omitted :]


def test_fitting_is_deterministic_for_prompt_caching() -> None:
    images = [_image((2200, 1500)), _image((1200, 900))]
    first_llm, second_llm = _llm(), _llm()
    first = first_llm._fit_request_images(_request(first_llm, *images), first_llm._image_budget())
    again = first_llm._fit_request_images(_request(first_llm, *images), first_llm._image_budget())
    fresh = second_llm._fit_request_images(
        _request(second_llm, *images), second_llm._image_budget()
    )
    assert first[0].arguments == again[0].arguments == fresh[0].arguments


def test_undecodable_image_is_left_for_the_provider() -> None:
    llm = _llm()
    bogus = ImageContent(
        type="image", mime_type="image/png", data=base64.b64encode(b"x" * 400_000).decode()
    )
    request = _request(llm, bogus)
    fitted, report = llm._fit_request_images(request, llm._image_budget())
    assert not report.changed
    assert fitted.arguments == request.arguments


def test_null_settings_disable_fitting() -> None:
    llm = _llm(_UNLIMITED)
    request = _request(llm, _image((2400, 2400)))
    fitted, report = llm._fit_request_images(request, llm._image_budget())
    assert not report.changed
    assert fitted.arguments == request.arguments


def _too_large() -> APIStatusError:
    request = httpx2.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    return APIStatusError(
        "Error code: 413", response=httpx2.Response(413, request=request), body=None
    )


def _fake_router(
    limit: int, sent: list[dict[str, Any]]
) -> Callable[[OpenAILLM, _OpenAICompletionRequest], object]:
    async def run(self: OpenAILLM, request: _OpenAICompletionRequest) -> _OpenAICompletionResponse:
        sent.append(request.arguments)
        if request_bytes(request.arguments) > limit:
            raise _too_large()
        return _OpenAICompletionResponse(response=None, streamed_reasoning=[])

    return run


@pytest.mark.asyncio
async def test_413_with_images_is_recovered_by_one_smaller_resend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The budget is off, so only the router (a 2 MB limit here) can reject the request.
    llm = _llm(_UNLIMITED)
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(OpenAILLM, "_run_openai_completion_request", _fake_router(2_000_000, sent))
    request = _request(llm, *[_image((600, 600), seed=i) for i in range(3)])
    await llm._run_openai_completion_request(request)
    assert len(sent) == 2
    assert request_bytes(sent[1]) <= request_bytes(sent[0]) // 2
    assert _placeholders(sent[1]) > 0
    assert _image_urls(sent[1]), "the newest image is kept"


@pytest.mark.asyncio
async def test_413_without_images_is_not_resent(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm()
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(OpenAILLM, "_run_openai_completion_request", _fake_router(10, sent))
    request = llm._openai_completion_request([{"role": "user", "content": "hello"}], None, None)
    with pytest.raises(APIStatusError):
        await llm._run_openai_completion_request(request)
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_custom_endpoint_requests_are_sent_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm(HuggingFaceSettings(base_url="https://my-endpoint.example/v1"))
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(OpenAILLM, "_run_openai_completion_request", _fake_router(10, sent))
    request = _request(llm, _image((2400, 2400)))
    with pytest.raises(APIStatusError):
        await llm._run_openai_completion_request(request)
    assert sent == [request.arguments], "no fitting and no 413 resend"


def test_image_count_budget_omits_oldest_images_first() -> None:
    from fast_agent.llm.provider.request_images import ChatRequestImageFitter, ImageBudget

    llm = _llm(_UNLIMITED)
    images = [_image((32, 32), noisy=False) for _ in range(4)]
    request = _request(llm, *images)
    fitter = ChatRequestImageFitter()
    fitted_args, report = fitter.fit(
        request.arguments,
        ImageBudget(
            max_dimension=None,
            reencode_min_bytes=None,
            max_request_bytes=None,
            max_images=2,
        ),
    )
    assert report.images == 4
    assert report.omitted == 2
    assert _placeholders(fitted_args) == 2
    assert _image_urls(fitted_args) == _image_urls(request.arguments)[2:]


def test_per_image_byte_cap_shrinks_oversized_image() -> None:
    from fast_agent.llm.provider.request_images import ChatRequestImageFitter, ImageBudget

    llm = _llm(_UNLIMITED)
    big_noisy = _image((1200, 1200), noisy=True)
    request = _request(llm, big_noisy)
    fitter = ChatRequestImageFitter()
    fitted_args, report = fitter.fit(
        request.arguments,
        ImageBudget(
            max_dimension=None,
            reencode_min_bytes=None,
            max_request_bytes=None,
            max_image_bytes=120_000,
        ),
    )
    (url,) = _image_urls(fitted_args)
    raw = base64.b64decode(url.split(",", 1)[1])
    assert len(raw) <= 120_000
    assert report.changed


def test_max_request_bytes_default_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_MAX_REQUEST_BYTES", raising=False)
    assert _llm(HuggingFaceSettings())._image_budget().max_request_bytes == 10_485_760

    monkeypatch.setenv("HF_MAX_REQUEST_BYTES", "16777216")
    assert _llm(HuggingFaceSettings())._image_budget().max_request_bytes == 16_777_216

    # Explicit config setting takes precedence over environment variable, even when equal to default.
    monkeypatch.setenv("HF_MAX_REQUEST_BYTES", "0")
    assert _llm(HuggingFaceSettings())._image_budget().max_request_bytes is None
    assert (
        _llm(HuggingFaceSettings(max_request_bytes=10_485_760))._image_budget().max_request_bytes
        == 10_485_760
    )
    assert (
        _llm(HuggingFaceSettings(max_request_bytes=6_000_000))._image_budget().max_request_bytes
        == 6_000_000
    )

    # "none" / "0" / "off" disable the budget via environment variable.
    for disable_value in ("none", "0", "off"):
        monkeypatch.setenv("HF_MAX_REQUEST_BYTES", disable_value)
        assert _llm(HuggingFaceSettings())._image_budget().max_request_bytes is None

    # Invalid or negative values warn and fall back to default.
    for invalid_value in ("not-a-number", "-10"):
        monkeypatch.setenv("HF_MAX_REQUEST_BYTES", invalid_value)
        with pytest.warns(UserWarning, match="HF_MAX_REQUEST_BYTES"):
            assert _llm(HuggingFaceSettings())._image_budget().max_request_bytes == 10_485_760
