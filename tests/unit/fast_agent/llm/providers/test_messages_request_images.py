from __future__ import annotations

import base64
import io
import json
import os
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import httpx
import httpx2
import pytest
from anthropic import AsyncAnthropic
from PIL import Image

from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.copilot import broker, images
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM
from fast_agent.llm.provider.request_images import (
    IMAGE_OMITTED_PLACEHOLDER,
    ImageBudget,
    MessagesRequestImageFitter,
    ResponsesRequestImageFitter,
    request_bytes,
)
from fast_agent.types import LlmStopReason, RequestParams

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _png_bytes(width: int, height: int, *, noisy: bool = False, alpha: bool = False) -> bytes:
    mode = "RGBA" if alpha else "RGB"
    if noisy:
        channels = 4 if alpha else 3
        raw = os.urandom(width * height * channels)
        image = Image.frombytes(mode, (width, height), raw)
    else:
        color = (20, 40, 80, 128) if alpha else (20, 40, 80)
        image = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _messages_image(raw: bytes, *, mime: str = "image/png", **extra: Any) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": base64.b64encode(raw).decode("ascii"),
        },
        **extra,
    }


def _anthropic_ok_sse(text: str = "ok") -> bytes:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def test_messages_fitter_normalizes_direct_and_tool_result_images_without_mutating_input() -> None:
    direct_raw = _png_bytes(1920, 1080, noisy=True)
    nested_raw = _png_bytes(2400, 1200, alpha=True)
    arguments: dict[str, Any] = {
        "model": "claude-fable-5.2",
        "messages": [
            {
                "role": "user",
                "content": [
                    _messages_image(direct_raw),
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [_messages_image(nested_raw)],
                    },
                ],
            }
        ],
    }
    snapshot = deepcopy(arguments)
    budget = ImageBudget(
        max_dimension=1568,
        reencode_min_bytes=200_000,
        max_request_bytes=28 * 1024 * 1024,
        low_water_request_bytes=16 * 1024 * 1024,
        max_image_bytes=3_750_000,
    )

    fitted, report = MessagesRequestImageFitter().fit(arguments, budget)

    assert arguments == snapshot
    assert report.images == 2
    assert report.resized == 2
    assert report.omitted == 0

    direct_out = fitted["messages"][0]["content"][0]["source"]
    assert direct_out["media_type"] == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(direct_out["data"]))) as opened:
        assert max(opened.size) == 1568

    nested_out = fitted["messages"][0]["content"][1]["content"][0]["source"]
    assert nested_out["media_type"] == "image/png"
    with Image.open(io.BytesIO(base64.b64decode(nested_out["data"]))) as opened:
        assert opened.mode == "RGBA"
        assert opened.size == (1568, 784)


def test_batch_eviction_stabilizes_prefix_across_turns_and_resets_on_history_change() -> None:
    fitter = MessagesRequestImageFitter()
    imgs = [_messages_image(_png_bytes(600, 600, noisy=True)) for _ in range(6)]
    probe_fitted, _ = MessagesRequestImageFitter().fit(
        {"messages": [{"role": "user", "content": [imgs[0]]}]},
        ImageBudget(max_dimension=1568, reencode_min_bytes=200_000, max_request_bytes=None),
    )
    per_img_bytes = request_bytes(probe_fitted)
    # High-water fits ~4 images; low-water fits ~2 images.
    high_water = round(per_img_bytes * 4.2)
    low_water = round(per_img_bytes * 2.2)
    budget = ImageBudget(
        max_dimension=1568,
        reencode_min_bytes=200_000,
        max_request_bytes=high_water,
        low_water_request_bytes=low_water,
    )

    # Turn 1: 5 images exceed high-water -> batch-evict 3 images down to <= low-water.
    turn1_msgs = [{"role": "user", "content": [img]} for img in imgs[:5]]
    fitted1, report1 = fitter.fit({"messages": turn1_msgs}, budget)
    assert report1.omitted == 3
    assert request_bytes(fitted1) <= low_water
    evicted_count = report1.omitted

    # Turn 2: pure-text follow-up keeps the exact same evicted prefix (prompt cache stable).
    turn2_msgs = [*turn1_msgs, {"role": "user", "content": [{"type": "text", "text": "next"}]}]
    fitted2, report2 = fitter.fit({"messages": turn2_msgs}, budget)
    assert report2.omitted == evicted_count
    assert fitted2["messages"][:5] == fitted1["messages"]

    # Turn 3: adding 1 more image stays between low-water and high-water -> zero extra evictions!
    turn3_msgs = [*turn2_msgs, {"role": "user", "content": [imgs[5]]}]
    fitted3, report3 = fitter.fit({"messages": turn3_msgs}, budget)
    assert report3.omitted == evicted_count
    assert fitted3["messages"][:5] == fitted1["messages"]
    assert fitted3["messages"][-1]["content"][0]["type"] == "image"

    # Compaction / rewind changes the leading image sequence -> watermark resets automatically.
    compacted_msgs = [{"role": "user", "content": [imgs[5]]}]
    fitted_after_compact, report_compact = fitter.fit({"messages": compacted_msgs}, budget)
    assert report_compact.omitted == 0
    assert fitted_after_compact["messages"][0]["content"][0]["type"] == "image"


def test_responses_fitter_normalizes_and_batch_rotates_input_images() -> None:
    fitter = ResponsesRequestImageFitter()
    imgs = [
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,"
            + base64.b64encode(_png_bytes(600, 600, noisy=True)).decode("ascii"),
        }
        for _ in range(4)
    ]
    payload: dict[str, Any] = {
        "input": [
            {"role": "user", "content": [imgs[0]]},
            {"type": "function_call_output", "call_id": "c1", "output": [imgs[1], imgs[2]]},
            {"role": "user", "content": [imgs[3]]},
        ]
    }
    budget = ImageBudget(
        max_dimension=512,
        reencode_min_bytes=100_000,
        max_request_bytes=900_000,
        low_water_request_bytes=500_000,
    )
    fitted, report = fitter.fit(payload, budget)
    assert report.omitted >= 2
    assert fitted["input"][0]["content"][0] == {
        "type": "input_text",
        "text": IMAGE_OMITTED_PLACEHOLDER,
    }
    assert fitted["input"][-1]["content"][0]["type"] == "input_image"
    assert request_bytes(fitted) <= 500_000


@pytest.mark.asyncio
async def test_copilot_messages_frees_url_slots_on_rotation_and_guards_upload_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = CopilotEndpoint(
        model_id="claude-fable-5.2",
        wire_api="messages",
        transport="sse",
        base_url="https://copilot.example",
        headers={"authorization": "Bearer synthetic"},
    )

    class _Broker:
        async def resolve(self, model_id: str, **_: Any) -> CopilotEndpoint:
            return endpoint

    monkeypatch.setattr(broker, "get_copilot_broker", lambda _ctx: _Broker())

    upload_should_fail = False
    uploaded_payloads: list[bytes] = []
    original_http_client = httpx.AsyncClient

    def upload_handler(request: httpx.Request) -> httpx.Response:
        if upload_should_fail:
            return httpx.Response(503, text="unavailable")
        uploaded_payloads.append(request.content)
        return httpx.Response(
            201,
            json={"url": f"https://attachment.example/img-{len(uploaded_payloads)}"},
        )

    monkeypatch.setattr(
        images.httpx,
        "AsyncClient",
        lambda **kw: original_http_client(**kw, transport=httpx.MockTransport(upload_handler)),
    )

    sent_bodies: list[dict[str, Any]] = []

    async def anthropic_handler(request: httpx2.Request) -> httpx2.Response:
        sent_bodies.append(json.loads(await request.aread()))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_anthropic_ok_sse(),
        )

    def client_factory(**kwargs: Any) -> AsyncAnthropic:
        kwargs["http_client"] = httpx2.AsyncClient(
            transport=httpx2.MockTransport(anthropic_handler)
        )
        return AsyncAnthropic(**kwargs)

    monkeypatch.setattr("fast_agent.llm.provider.copilot.messages.AsyncAnthropic", client_factory)

    raw_images = [_png_bytes(650, 650, noisy=True) for _ in range(5)]
    probe_fitted, _ = MessagesRequestImageFitter().fit(
        {"messages": [{"role": "user", "content": [_messages_image(raw_images[0])]}]},
        ImageBudget(max_dimension=1568, reencode_min_bytes=100_000, max_request_bytes=None),
    )
    one_inline = request_bytes(probe_fitted)
    # 2 URL slots + high-water < 2 inline images + low-water < 1 inline image ->
    # 5 images must evict oldest 3 so surviving 2 newer images both fit as URLs.
    high_water = round(one_inline * 1.6)
    low_water = round(one_inline * 0.6)
    tight_budget = ImageBudget(
        max_dimension=1568,
        reencode_min_bytes=100_000,
        max_request_bytes=high_water,
        low_water_request_bytes=low_water,
        url_image_slots=2,
    )

    llm = CopilotMessagesLLM(context=Context(), model="claude-fable-5.2")
    monkeypatch.setattr(llm, "_anthropic_image_budget", lambda: tight_budget)
    monkeypatch.setattr(images, "MAX_MESSAGES_URL_IMAGES", 2)

    msg: dict[str, Any] = {
        "role": "user",
        "content": [_messages_image(raw) for raw in raw_images],
    }
    params = RequestParams(model="claude-fable-5.2", max_tokens=64)
    await llm._anthropic_completion(msg, params)

    content = sent_bodies[-1]["messages"][-1]["content"]
    assert [b["type"] for b in content[:3]] == ["text", "text", "text"]
    assert [b["source"]["type"] for b in content[3:]] == ["url", "url"]

    # Transient upload failure on a fresh LLM still stays under max_request_bytes without
    # polluting the persistent watermark once uploads recover.
    llm_fail = CopilotMessagesLLM(context=Context(), model="claude-fable-5.2")
    monkeypatch.setattr(llm_fail, "_anthropic_image_budget", lambda: replace(tight_budget))
    two_img_msg: dict[str, Any] = {
        "role": "user",
        "content": [_messages_image(raw_images[0]), _messages_image(raw_images[1])],
    }
    upload_should_fail = True
    await llm_fail._anthropic_completion(two_img_msg, params)
    fail_body = sent_bodies[-1]
    assert len(json.dumps(fail_body, separators=(",", ":")).encode()) <= high_water
    assert llm_fail._image_fitter._evicted_prefix == ()

    upload_should_fail = False
    await llm_fail._anthropic_completion(two_img_msg, params)
    recovered_content = sent_bodies[-1]["messages"][-1]["content"]
    assert [b["source"]["type"] for b in recovered_content] == ["url", "url"]


@pytest.mark.asyncio
async def test_anthropic_413_shrinks_images_once_and_preserves_cache_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[dict[str, Any]] = []
    proxy_body_limit: int | None = None

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal proxy_body_limit
        raw_body = await request.aread()
        if proxy_body_limit is None:
            proxy_body_limit = int(len(raw_body) * 0.6)
        body = json.loads(raw_body)
        attempts.append(body)
        if len(raw_body) > proxy_body_limit:
            return httpx2.Response(
                413,
                headers={"content-type": "application/json"},
                json={
                    "type": "error",
                    "error": {"type": "request_too_large", "message": "too big"},
                },
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_anthropic_ok_sse("recovered"),
        )

    llm = AnthropicLLM(context=Context(), model="claude-sonnet-4-6", api_key="sk-ant-test")
    monkeypatch.setattr(
        llm,
        "_initialize_anthropic_client",
        lambda: AsyncAnthropic(
            api_key="sk-ant-test",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        ),
    )

    msg: dict[str, Any] = {
        "role": "user",
        "content": [
            _messages_image(_png_bytes(700, 700, noisy=True), cache_control={"type": "ephemeral"}),
            _messages_image(_png_bytes(700, 700, noisy=True)),
            _messages_image(_png_bytes(300, 300, noisy=True)),
        ],
    }
    result = await llm._anthropic_completion(
        msg, RequestParams(model="claude-sonnet-4-6", max_tokens=64)
    )
    assert result.stop_reason == LlmStopReason.END_TURN
    assert result.last_text() == "recovered"
    assert len(attempts) == 2
    retry_content = attempts[1]["messages"][-1]["content"]
    first_block = retry_content[0]
    assert first_block["type"] == "text"
    assert first_block["text"] == IMAGE_OMITTED_PLACEHOLDER
    assert first_block["cache_control"] == {"type": "ephemeral"}
    omitted_on_retry = sum(1 for b in retry_content if b["type"] == "text")
    assert omitted_on_retry >= 1

    # Next canonical turn with the same original PNGs omits that prefix immediately (no second 413)
    # and produces byte-identical surviving image blocks (no double JPEG re-encoding on 413 retry).
    attempts.clear()
    followup_result = await llm._anthropic_completion(
        deepcopy(msg), RequestParams(model="claude-sonnet-4-6", max_tokens=64)
    )
    assert followup_result.stop_reason == LlmStopReason.END_TURN
    assert len(attempts) == 1
    replay_content = attempts[0]["messages"][-1]["content"]
    omitted_on_replay = sum(1 for b in replay_content if b["type"] == "text")
    assert omitted_on_replay == omitted_on_retry
    assert [{k: v for k, v in b.items() if k != "cache_control"} for b in replay_content] == [
        {k: v for k, v in b.items() if k != "cache_control"} for b in retry_content
    ]


def test_tail_rewind_resets_watermark_when_truncated_history_fits() -> None:
    fitter = MessagesRequestImageFitter()
    imgs = [_messages_image(_png_bytes(600, 600, noisy=True)) for _ in range(6)]
    probe_fitted, _ = MessagesRequestImageFitter().fit(
        {"messages": [{"role": "user", "content": [imgs[0]]}]},
        ImageBudget(max_dimension=1568, reencode_min_bytes=200_000, max_request_bytes=None),
    )
    per_img = request_bytes(probe_fitted)
    budget = ImageBudget(
        max_dimension=1568,
        reencode_min_bytes=200_000,
        max_request_bytes=round(per_img * 4.5),
        low_water_request_bytes=round(per_img * 2.5),
    )

    six_msgs = [{"role": "user", "content": [img]} for img in imgs]
    _, report_full = fitter.fit({"messages": six_msgs}, budget)
    assert report_full.omitted >= 3

    # /history rewind truncates the tail back to the first 3 images, which fit under high-water.
    rewound_msgs = six_msgs[:3]
    fitted_rewound, report_rewound = fitter.fit({"messages": rewound_msgs}, budget)
    assert report_rewound.omitted == 0
    assert all(m["content"][0]["type"] == "image" for m in fitted_rewound["messages"])


def test_exif_transpose_play_once_gif_and_rgba_png_recompression() -> None:
    from fast_agent.llm.provider.request_images import ChatRequestImageFitter

    fitter = ChatRequestImageFitter()

    # 1. Orientation-6 JPEG (160x80 stored -> 80x160 upright -> 32x64 at max_dimension=64).
    exif = Image.Exif()
    exif[0x0112] = 6
    jpeg_buf = io.BytesIO()
    Image.new("RGB", (160, 80), color=(90, 20, 40)).save(jpeg_buf, format="JPEG", exif=exif)
    jpeg_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_buf.getvalue()).decode("ascii")
    fitted_exif, _ = fitter.fit(
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": jpeg_url}}]}
            ]
        },
        ImageBudget(max_dimension=64, reencode_min_bytes=1, max_request_bytes=None),
    )
    out_b64 = fitted_exif["messages"][0]["content"][0]["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(out_b64))) as upright:
        assert upright.size == (32, 64)

    # 2. Play-once 2-frame GIF (no loop metadata) is left untouched as an animation.
    gif_buf = io.BytesIO()
    f1 = Image.new("RGB", (200, 200), color=(255, 0, 0))
    f2 = Image.new("RGB", (200, 200), color=(0, 255, 0))
    f1.save(gif_buf, format="GIF", save_all=True, append_images=[f2])
    gif_url = "data:image/gif;base64," + base64.b64encode(gif_buf.getvalue()).decode("ascii")
    fitted_gif, report_gif = fitter.fit(
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": gif_url}}]}
            ]
        },
        ImageBudget(max_dimension=64, reencode_min_bytes=1, max_request_bytes=None),
    )
    assert not report_gif.changed
    assert fitted_gif["messages"][0]["content"][0]["image_url"]["url"] == gif_url

    # 3. Uncompressed RGBA PNG with transparency losslessly recompresses under reencode_min_bytes.
    rgba_buf = io.BytesIO()
    Image.new("RGBA", (512, 512), color=(10, 20, 30, 128)).save(
        rgba_buf, format="PNG", compress_level=0
    )
    uncompressed = rgba_buf.getvalue()
    assert len(uncompressed) > 1_000_000
    rgba_url = "data:image/png;base64," + base64.b64encode(uncompressed).decode("ascii")
    fitted_rgba, report_rgba = fitter.fit(
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": rgba_url}}]}
            ]
        },
        ImageBudget(max_dimension=2048, reencode_min_bytes=100_000, max_request_bytes=1_000_000),
    )
    assert report_rgba.omitted == 0
    assert report_rgba.reencoded == 1
    assert request_bytes(fitted_rgba) < 50_000


def test_max_url_images_caps_existing_remote_urls_without_dropping_inline_images() -> None:
    inline = _messages_image(_png_bytes(40, 40))
    remote_urls = [
        {"type": "image", "source": {"type": "url", "url": f"https://example.com/{i}.png"}}
        for i in range(21)
    ]
    payload: dict[str, Any] = {"messages": [{"role": "user", "content": [inline, *remote_urls]}]}
    budget = ImageBudget(
        max_dimension=1568,
        reencode_min_bytes=200_000,
        max_request_bytes=28 * 1024 * 1024,
        url_image_slots=20,
        max_url_images=20,
    )
    fitted, report = MessagesRequestImageFitter().fit(payload, budget)
    content = fitted["messages"][0]["content"]
    assert report.omitted == 1
    # Inline image stays intact; oldest remote URL is omitted so exactly 20 URLs remain.
    assert content[0]["type"] == "image" and content[0]["source"]["type"] == "base64"
    assert content[1]["type"] == "text"
    assert sum(1 for b in content if b["type"] == "image" and b["source"]["type"] == "url") == 20


def test_fitter_cache_is_bounded_and_gpt6_astra_caps_total_images() -> None:
    from fast_agent.llm.provider.copilot.models import get_copilot_model
    from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM

    assert get_copilot_model("gpt-6-astra").max_prompt_images == 50
    assert get_copilot_model("gpt-6.2-sol").max_prompt_images is None

    budget = CopilotResponsesLLM._copilot_image_budget(
        get_copilot_model("gpt-6-astra"), overhead_bytes=1_000_000
    )
    assert budget.max_images == 50
    assert budget.max_url_images == 50
    assert budget.max_request_bytes == 40 * 1024 * 1024 - 1_000_000

    fitter = MessagesRequestImageFitter()
    norm_budget = ImageBudget(max_dimension=32, reencode_min_bytes=1, max_request_bytes=None)
    for _ in range(270):
        raw = _png_bytes(64, 64, noisy=True)
        fitter.fit({"messages": [{"role": "user", "content": [_messages_image(raw)]}]}, norm_budget)
    assert len(fitter._cache) <= 256


@pytest.mark.asyncio
async def test_copilot_messages_413_resend_claims_freed_url_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = CopilotEndpoint(
        model_id="claude-fable-5.2",
        wire_api="messages",
        transport="sse",
        base_url="https://copilot.example",
        headers={"authorization": "Bearer synthetic"},
    )

    class _Broker:
        async def resolve(self, model_id: str, **_: Any) -> CopilotEndpoint:
            return endpoint

    monkeypatch.setattr(broker, "get_copilot_broker", lambda _ctx: _Broker())
    monkeypatch.setattr(images, "MAX_MESSAGES_URL_IMAGES", 1)
    monkeypatch.setattr("fast_agent.llm.provider.copilot.messages.MAX_MESSAGES_URL_IMAGES", 1)

    uploaded_count = 0
    original_http_client = httpx.AsyncClient

    def upload_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal uploaded_count
        uploaded_count += 1
        return httpx.Response(201, json={"url": f"https://attachment.example/u-{uploaded_count}"})

    monkeypatch.setattr(
        images.httpx,
        "AsyncClient",
        lambda **kw: original_http_client(**kw, transport=httpx.MockTransport(upload_handler)),
    )

    sent_bodies: list[dict[str, Any]] = []

    async def anthropic_handler(request: httpx2.Request) -> httpx2.Response:
        raw_body = await request.aread()
        sent_bodies.append(json.loads(raw_body))
        if len(sent_bodies) == 1:
            return httpx2.Response(
                413,
                headers={"content-type": "application/json"},
                json={"type": "error", "error": {"type": "request_too_large", "message": "big"}},
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_anthropic_ok_sse("ok"),
        )

    def client_factory(**kwargs: Any) -> AsyncAnthropic:
        kwargs["http_client"] = httpx2.AsyncClient(
            transport=httpx2.MockTransport(anthropic_handler)
        )
        return AsyncAnthropic(**kwargs)

    monkeypatch.setattr("fast_agent.llm.provider.copilot.messages.AsyncAnthropic", client_factory)

    llm = CopilotMessagesLLM(context=Context(), model="claude-fable-5.2")
    msg: dict[str, Any] = {
        "role": "user",
        "content": [
            _messages_image(_png_bytes(600, 600, noisy=True)),
            _messages_image(_png_bytes(600, 600, noisy=True)),
        ],
    }
    res = await llm._anthropic_completion(
        msg, RequestParams(model="claude-fable-5.2", max_tokens=64)
    )
    assert res.stop_reason == LlmStopReason.END_TURN
    assert len(sent_bodies) == 2
    # Attempt 1 had 1 URL + 1 inline; 413 omitted image 0, freeing the single URL slot for image 1.
    retry_content = sent_bodies[1]["messages"][-1]["content"]
    assert retry_content[0]["type"] == "text"
    assert retry_content[1]["type"] == "image"
    assert retry_content[1]["source"]["type"] == "url"


def test_combined_url_cap_and_byte_budget_eviction_keeps_watermark_aligned() -> None:
    fitter = MessagesRequestImageFitter()
    inline_a = _messages_image(_png_bytes(500, 500, noisy=True))
    inline_b = _messages_image(_png_bytes(500, 500, noisy=True))
    url_0 = {"type": "image", "source": {"type": "url", "url": "https://example.com/0.png"}}
    url_2 = {"type": "image", "source": {"type": "url", "url": "https://example.com/2.png"}}

    one_fitted, _ = MessagesRequestImageFitter().fit(
        {"messages": [{"role": "user", "content": [inline_b]}]},
        ImageBudget(max_dimension=1568, reencode_min_bytes=100_000, max_request_bytes=None),
    )
    one_bytes = request_bytes(one_fitted)
    budget = ImageBudget(
        max_dimension=1568,
        reencode_min_bytes=100_000,
        max_request_bytes=round(one_bytes * 1.4),
        low_water_request_bytes=round(one_bytes * 1.2),
        max_url_images=0,
    )
    # Interleaved [URL_0, inline_a, URL_2, inline_b]: URL cap drops indices 0 and 2,
    # byte budget drops inline_a (index 1), leaving only inline_b (orig_index=3).
    msgs = [{"role": "user", "content": [url_0, inline_a, url_2, inline_b]}]
    fitted1, report1 = fitter.fit({"messages": msgs}, budget)
    assert report1.omitted == 3
    assert [b["type"] for b in fitted1["messages"][0]["content"]] == [
        "text",
        "text",
        "text",
        "image",
    ]
    assert len(fitter._evicted_prefix) == 3

    # Next turn preserves the exact 3-slot evicted prefix without evicting inline_b.
    msgs2 = [*msgs, {"role": "user", "content": [{"type": "text", "text": "follow up"}]}]
    fitted2, report2 = fitter.fit({"messages": msgs2}, budget)
    assert report2.omitted == 3
    assert fitted2["messages"][0] == fitted1["messages"][0]


@pytest.mark.asyncio
async def test_copilot_responses_handles_omit_in_extra_body_and_codex_inlines_file_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import openai

    from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM
    from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM

    endpoint = CopilotEndpoint(
        model_id="gpt-6-astra",
        wire_api="responses",
        transport="sse",
        base_url="https://copilot.example",
        headers={"authorization": "Bearer synthetic"},
    )
    copilot_llm = CopilotResponsesLLM(context=Context(), model="gpt-6-astra")
    copilot_llm._copilot_endpoint.set(endpoint)

    original_http_client = httpx.AsyncClient
    monkeypatch.setattr(
        images.httpx,
        "AsyncClient",
        lambda **kw: original_http_client(
            **kw,
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(201, json={"url": "https://attachment.example/ok"})
            ),
        ),
    )

    data_url = "data:image/png;base64," + base64.b64encode(_png_bytes(64, 64)).decode("ascii")
    params_with_omit = RequestParams(
        model="gpt-6-astra",
        metadata={"extra_body": {"temperature": openai.omit}},
    )
    normalized = await copilot_llm._normalize_request_files(
        copilot_llm._responses_client(),
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": data_url}],
            }
        ],
        display_model="gpt-6-astra",
        request_params=params_with_omit,
    )
    assert normalized[0]["content"][0]["image_url"] == "https://attachment.example/ok"

    # CodexResponsesLLM inlines even extensionless file:// input_image URLs (no /files API)
    # by sniffing image bytes rather than emitting application/octet-stream, and fits oversized PNGs.
    img_file = tmp_path / "extensionless_capture"
    img_file.write_bytes(_png_bytes(2400, 1400, noisy=True))
    codex_llm = CodexResponsesLLM(context=Context(), model="gpt-6.2-sol")

    class _NoFilesClient:
        @property
        def files(self) -> Any:
            raise AssertionError("CodexResponsesLLM must not call OpenAI Files API")

    codex_input = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": img_file.as_uri(), "detail": "high"}],
        }
    ]
    codex_snapshot = deepcopy(codex_input)
    codex_out = await codex_llm._normalize_request_files(
        cast("AsyncOpenAI", _NoFilesClient()),
        codex_input,
        display_model="gpt-6.2-sol",
        request_params=RequestParams(model="gpt-6.2-sol"),
    )
    assert codex_input == codex_snapshot
    out_part = codex_out[0]["content"][0]
    assert out_part["detail"] == "high"
    assert out_part["image_url"].startswith("data:image/jpeg;base64,")
    raw_out = base64.b64decode(out_part["image_url"].split(",", 1)[1])
    with Image.open(io.BytesIO(raw_out)) as opened:
        assert max(opened.size) == 2048
