import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from mcp_types import TextContent
from openai import AsyncOpenAI
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import ValidationError

from fast_agent.config import Settings, XAISettings, XAIWebSearchSettings
from fast_agent.constants import OPENAI_ASSISTANT_MESSAGE_ITEMS, OPENAI_REASONING_ENCRYPTED
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ResponsesWebSocketError,
    StatelessResponsesWsPlanner,
    resolve_responses_ws_url,
)
from fast_agent.llm.provider.openai.tool_stream_state import OpenAIToolStreamState
from fast_agent.llm.provider.openai.xai_image_uploads import XAIImageUploadManager
from fast_agent.llm.provider.openai.xai_responses import (
    DEFAULT_XAI_MODEL,
    GROK_EXTENDED_STREAMING_TIMEOUT,
    XAIResponsesLLM,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.usage_tracking import UsageSchema
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

REPO_ROOT = Path(__file__).resolve().parents[5]


class _XAIStreamingHarness(XAIResponsesLLM):
    def __init__(self) -> None:
        super().__init__(
            context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
            model="grok-4.3",
            x_search=True,
        )
        self.events: list[tuple[str, dict]] = []

    def _notify_tool_stream_listeners(self, event_type, payload=None) -> None:
        self.events.append((event_type, payload or {}))


class _XAIFileAPISimulator:
    def __init__(self, *, upload_status: int = 200, public_url_status: int = 200) -> None:
        self.upload_status = upload_status
        self.public_url_status = public_url_status
        self.upload_bodies: list[bytes] = []
        self.public_url_bodies: list[bytes] = []

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = await request.aread()
        if request.url.path == "/v1/files":
            self.upload_bodies.append(body)
            if self.upload_status != 200:
                return httpx2.Response(
                    self.upload_status,
                    json={"error": {"message": "upload unavailable", "type": "server_error"}},
                )
            file_number = len(self.upload_bodies)
            return httpx2.Response(
                200,
                json={
                    "id": f"file_{file_number}",
                    "bytes": 8,
                    "created_at": 1_786_800_000,
                    "expires_at": 1_786_886_400,
                    "filename": "image.png",
                    "object": "file",
                    "purpose": "assistants",
                    "status": "uploaded",
                },
            )
        if request.url.path.startswith("/v1/files/file_") and request.url.path.endswith(
            "/public-url"
        ):
            self.public_url_bodies.append(body)
            if self.public_url_status != 200:
                return httpx2.Response(
                    self.public_url_status,
                    json={
                        "error": {
                            "message": "public URL unavailable",
                            "type": "server_error",
                        }
                    },
                )
            file_id = request.url.path.split("/")[-2]
            return httpx2.Response(
                200,
                json={"public_url": f"https://files-cdn.x.ai/test/{file_id}.png"},
            )
        return httpx2.Response(404)


def _xai_file_client(simulator: _XAIFileAPISimulator) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(simulator)),
    )


def _inline_image_part(data: bytes = b"\x89PNG\r\n\x1a\n") -> dict[str, str]:
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{encoded}",
        "detail": "high",
    }


def test_xai_responses_provider_defaults_to_websocket_transport() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm.provider == Provider.XAI
    assert llm.configured_transport == "websocket"


def test_xai_image_upload_settings_default_to_public_urls_and_validate_ttl() -> None:
    settings = XAISettings()

    assert settings.image_upload_mode == "public_url"
    assert settings.image_upload_ttl_seconds == 86_400
    assert XAISettings(image_upload_ttl_seconds=3_600).image_upload_ttl_seconds == 3_600
    assert XAISettings(image_upload_ttl_seconds=2_592_000).image_upload_ttl_seconds == 2_592_000
    with pytest.raises(ValidationError):
        XAISettings(image_upload_ttl_seconds=3_599)
    with pytest.raises(ValidationError):
        XAISettings(image_upload_ttl_seconds=2_592_001)


@pytest.mark.asyncio
async def test_xai_image_upload_reuses_public_url_across_replayed_history() -> None:
    simulator = _XAIFileAPISimulator()
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    image_upload_mode="public_url",
                    image_upload_ttl_seconds=86_400,
                )
            )
        ),
        model="grok-4.6",
    )
    image_part = _inline_image_part()
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [image_part, {"type": "input_text", "text": "Describe it"}],
        }
    ]
    original = json.loads(json.dumps(input_items))

    async with _xai_file_client(simulator) as client:
        first = await llm._normalize_input_files(client, input_items)
        second = await llm._normalize_input_files(client, input_items)

    expected_image = {
        "type": "input_image",
        "image_url": "https://files-cdn.x.ai/test/file_1.png",
        "detail": "high",
    }
    assert first[0]["content"][0] == expected_image
    assert second[0]["content"][0] == expected_image
    assert input_items == original
    assert len(simulator.upload_bodies) == 1
    assert simulator.public_url_bodies == [b"{}"]

    upload_body = simulator.upload_bodies[0]
    assert b'name="purpose"\r\n\r\nassistants' in upload_body
    assert b'name="expires_after[anchor]"\r\n\r\ncreated_at' in upload_body
    assert b'name="expires_after[seconds]"\r\n\r\n86400' in upload_body
    assert b"image/png" in upload_body
    assert upload_body.index(b'name="expires_after[seconds]"') < upload_body.index(b'name="file"')


@pytest.mark.asyncio
async def test_xai_image_upload_leaves_remote_and_unsupported_images_unchanged() -> None:
    simulator = _XAIFileAPISimulator()
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(xai=XAISettings(api_key="test-key", image_upload_mode="public_url"))
        ),
        model="grok-4.6",
    )
    remote = {"type": "input_image", "image_url": "https://example.com/image.png"}
    unsupported = {
        "type": "input_image",
        "image_url": "data:image/webp;base64,AAAA",
    }

    async with _xai_file_client(simulator) as client:
        normalized_remote, remote_changed = await llm._normalize_input_image_part(client, remote)
        normalized_unsupported, unsupported_changed = await llm._normalize_input_image_part(
            client, unsupported
        )

    assert normalized_remote is remote
    assert remote_changed is False
    assert normalized_unsupported is unsupported
    assert unsupported_changed is False
    assert simulator.upload_bodies == []


@pytest.mark.asyncio
async def test_xai_image_upload_failure_falls_back_to_inline_data() -> None:
    simulator = _XAIFileAPISimulator(upload_status=503)
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(xai=XAISettings(api_key="test-key", image_upload_mode="public_url"))
        ),
        model="grok-4.6",
    )
    image = _inline_image_part()

    async with _xai_file_client(simulator) as client:
        normalized, changed = await llm._normalize_input_image_part(client, image)

    assert normalized is image
    assert changed is False
    assert len(simulator.upload_bodies) == 1
    assert simulator.public_url_bodies == []


@pytest.mark.asyncio
async def test_xai_public_url_failure_falls_back_without_caching_upload() -> None:
    simulator = _XAIFileAPISimulator(public_url_status=503)
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(xai=XAISettings(api_key="test-key", image_upload_mode="public_url"))
        ),
        model="grok-4.6",
    )
    image = _inline_image_part()

    async with _xai_file_client(simulator) as client:
        first, first_changed = await llm._normalize_input_image_part(client, image)
        second, second_changed = await llm._normalize_input_image_part(client, image)

    assert first is image
    assert first_changed is False
    assert second is image
    assert second_changed is False
    assert len(simulator.upload_bodies) == 2
    assert simulator.public_url_bodies == [b"{}", b"{}"]


@pytest.mark.asyncio
async def test_xai_image_upload_cache_refreshes_before_expiry() -> None:
    now = 100.0
    manager = XAIImageUploadManager(ttl_seconds=3_600, clock=lambda: now)
    simulator = _XAIFileAPISimulator()
    image_url = _inline_image_part()["image_url"]

    async with _xai_file_client(simulator) as client:
        first = await manager.public_url(client, image_url)
        now = 3_641.0
        second = await manager.public_url(client, image_url)

    assert first == "https://files-cdn.x.ai/test/file_1.png"
    assert second == "https://files-cdn.x.ai/test/file_2.png"
    assert len(simulator.upload_bodies) == 2
    assert len(simulator.public_url_bodies) == 2


def test_xai_websocket_usage_preserves_missing_cache_write_as_unknown() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "tests"
            / "fixtures"
            / "llm_traces"
            / "sanitized"
            / "xai_responses_websocket_usage_20260715.json"
        ).read_text()
    )
    input_details = payload.pop("input_tokens_details")
    output_details = payload.pop("output_tokens_details")
    usage = ResponseUsage.model_construct(
        **payload,
        input_tokens_details=InputTokensDetails.model_construct(**input_details),
        output_tokens_details=OutputTokensDetails.model_construct(**output_details),
    )
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.5",
    )

    turn = llm._translate_responses_usage(
        usage,
        provider=Provider.XAI,
        model="grok-4.5",
    )

    assert turn.usage_schema is UsageSchema.OPENAI_RESPONSES_COMPATIBLE
    assert turn.prompt.total == 373
    assert turn.prompt.uncached is None
    assert turn.prompt.cache_read == 128
    assert turn.prompt.cache_write is None
    assert turn.completion.total == 138
    assert turn.completion.reasoning == 124


def test_xai_responses_default_model_used_when_model_missing() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="",
    )

    assert llm.default_request_params.model == DEFAULT_XAI_MODEL


@pytest.mark.parametrize(
    ("model", "reasoning_effort", "expected_timeout"),
    [
        ("grok-4.5", "high", GROK_EXTENDED_STREAMING_TIMEOUT),
        ("grok-4.5", "medium", 120.0),
        ("grok-4.5", "low", 120.0),
        ("grok-4.6", "high", GROK_EXTENDED_STREAMING_TIMEOUT),
        ("grok-4.6", "xhigh", GROK_EXTENDED_STREAMING_TIMEOUT),
        ("grok-4.6", "medium", 120.0),
        ("grok-4.6", "low", 120.0),
        ("grok-4.3", "high", 120.0),
    ],
)
def test_xai_high_reasoning_gets_extended_streaming_timeout(
    model: str,
    reasoning_effort: str,
    expected_timeout: float,
) -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model=model,
        reasoning_effort=reasoning_effort,
    )

    assert llm.default_request_params.streaming_timeout == expected_timeout


@pytest.mark.parametrize(
    ("model", "reasoning_effort", "streaming_timeout"),
    [
        ("grok-4.5", "high", 45.0),
        ("grok-4.5", "high", None),
        ("grok-4.6", "xhigh", 45.0),
        ("grok-4.6", "xhigh", None),
    ],
)
def test_xai_explicit_streaming_timeout_overrides_high_reasoning_default(
    model: str,
    reasoning_effort: str,
    streaming_timeout: float | None,
) -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model=model,
        reasoning_effort=reasoning_effort,
        request_params=RequestParams(streaming_timeout=streaming_timeout),
    )

    assert llm.default_request_params.streaming_timeout == streaming_timeout


def test_xai_implicit_request_timeout_does_not_block_high_reasoning_default() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.5",
        reasoning_effort="high",
        request_params=RequestParams(model="grok-4.5", use_history=False),
    )

    assert llm.default_request_params.streaming_timeout == GROK_EXTENDED_STREAMING_TIMEOUT


def test_xai_responses_uses_xai_config_fallback() -> None:
    settings = Settings(
        xai=XAISettings(
            api_key="xai-key",
            base_url="https://gateway.example/xai/v1",
            default_headers={"X-Test": "1"},
            default_model="grok-4.5",
        )
    )
    llm = XAIResponsesLLM(context=Context(config=settings), model="")

    assert llm._api_key() == "xai-key"
    assert llm._base_url() == "https://gateway.example/xai/v1"
    assert llm._default_headers() == {"X-Test": "1"}
    assert llm.default_request_params.model == "grok-4.5"


def test_xai_responses_websocket_url_uses_responses_endpoint() -> None:
    assert resolve_responses_ws_url("https://api.x.ai/v1") == "wss://api.x.ai/v1/responses"


def test_xai_responses_websocket_headers_are_not_openai_beta_headers() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    default_headers={"X-Test": "1"},
                )
            )
        ),
        model="grok-4.3",
    )

    headers = llm._build_websocket_headers()

    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-Test"] == "1"
    assert "OpenAI-Beta" not in headers


def test_xai_websocket_disables_client_generated_keepalive_pings() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm._websocket_keepalive_options() == {"ping_interval": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key_source", ["config", "environment", "init"])
async def test_xai_api_key_401_does_not_enter_oauth_refresh(
    monkeypatch, api_key_source: str
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    settings = Settings(xai=XAISettings())
    init_api_key: str | None = None
    if api_key_source == "config":
        settings = Settings(xai=XAISettings(api_key="configured-key"))
    elif api_key_source == "environment":
        monkeypatch.setenv("XAI_API_KEY", "environment-key")
    else:
        init_api_key = "init-key"

    rejected = ResponsesWebSocketError("rejected API key", status=401)

    async def reject_connection(
        self, url: str, headers: dict[str, str], timeout_seconds: float | None
    ):
        del self, url, headers, timeout_seconds
        raise rejected

    monkeypatch.setattr(ResponsesLLM, "_create_websocket_connection", reject_connection)
    llm = XAIResponsesLLM(
        context=Context(config=settings),
        model="grok-4.3",
        api_key=init_api_key,
    )

    with pytest.raises(ResponsesWebSocketError) as exc_info:
        await llm._create_websocket_connection("wss://api.x.ai/v1/responses", {}, None)

    assert exc_info.value is rejected


def test_xai_responses_uses_stateless_websocket_planner() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert isinstance(llm._new_ws_request_planner(), StatelessResponsesWsPlanner)


def test_xai_responses_builds_parallel_response_payload_with_default_reasoning() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert args["model"] == "grok-4.3"
    assert args["store"] is False
    assert args["input"] == input_items
    assert args["parallel_tool_calls"] is True
    assert args["include"] == ["reasoning.encrypted_content"]
    assert args["reasoning"] == {"effort": "high"}
    assert "service_tier" not in args
    assert "stream" not in args
    assert "background" not in args


def test_xai_responses_builds_payload_with_selected_reasoning_effort() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        reasoning_effort="high",
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert llm.reasoning_effort == ReasoningEffortSetting(kind="effort", value="high")
    assert args["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize("model", ["grok-4.5", "grok-4.6"])
def test_xai_responses_builds_experimental_streaming_payload(model: str) -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    reasoning_summary="concise",
                    stream_tool_calls=True,
                )
            )
        ),
        model=model,
    )

    args = llm._build_response_args([], llm.default_request_params, tools=None)

    assert args["reasoning"] == {"effort": "high", "summary": "concise"}
    assert args["extra_body"] == {"stream_tool_calls": True}


def test_xai_responses_flattens_stream_tool_calls_for_websocket() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(xai=XAISettings(api_key="test-key", stream_tool_calls=True))
        ),
        model="grok-4.6",
    )
    args = llm._build_response_args([], llm.default_request_params, tools=None)

    llm._prepare_websocket_arguments(args)

    assert args["stream_tool_calls"] is True
    assert "extra_body" not in args


def test_xai_responses_rejects_unverified_experimental_model() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(xai=XAISettings(api_key="test-key", stream_tool_calls=True))
        ),
        model="grok-4.3",
    )

    with pytest.raises(ModelConfigError, match="supported only for grok-4.5, grok-4.6"):
        llm._build_response_args([], llm.default_request_params, tools=None)


def test_xai_grok_46_builds_payload_with_xhigh_reasoning() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.6",
        reasoning_effort="xhigh",
    )

    args = llm._build_response_args([], llm.default_request_params, tools=None)

    assert llm.reasoning_effort == ReasoningEffortSetting(kind="effort", value="xhigh")
    assert args["reasoning"] == {"effort": "xhigh"}


def test_xai_prompt_cache_key_is_stable_per_conversation_and_rotates_on_clear() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.6",
    )

    first = llm._build_response_args([], llm.default_request_params, tools=None)
    second = llm._build_response_args([], llm.default_request_params, tools=None)
    first_key = first["prompt_cache_key"]

    assert isinstance(first_key, str)
    assert first_key
    assert second["prompt_cache_key"] == first_key
    assert "extra_body" not in first

    planned = llm._new_ws_request_planner().plan(first)
    assert planned.arguments["prompt_cache_key"] == first_key

    llm.clear()
    after_clear = llm._build_response_args([], llm.default_request_params, tools=None)
    assert after_clear["prompt_cache_key"] != first_key


@pytest.mark.parametrize("model", ["grok-4.5", "grok-4.6"])
def test_xai_replays_distinct_assistant_messages_when_provider_reuses_item_id(
    model: str,
) -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model=model,
    )
    messages: list[PromptMessageExtended] = []
    for user_text, assistant_text in (
        ("good evening", "Hello."),
        ("write an essay", "The essay."),
        ("was that fun?", "Yes."),
    ):
        messages.append(
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=user_text)],
            )
        )
        raw_item = {
            "type": "message",
            "id": "msg_reused",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": assistant_text}],
        }
        messages.append(
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text=assistant_text)],
                channels={
                    OPENAI_ASSISTANT_MESSAGE_ITEMS: [
                        TextContent(type="text", text=json.dumps(raw_item))
                    ]
                },
            )
        )

    input_items = llm._convert_to_provider_format(messages)
    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)
    planned = llm._new_ws_request_planner().plan(args)
    replayed = planned.arguments["input"]

    assert [item["role"] for item in replayed] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [item["content"][0]["text"] for item in replayed if item["role"] == "assistant"] == [
        "Hello.",
        "The essay.",
        "Yes.",
    ]
    assert all("id" not in item for item in replayed if item["role"] == "assistant")


@pytest.mark.parametrize("model", ["grok-4.5", "grok-4.6"])
def test_xai_replays_distinct_reasoning_when_provider_reuses_item_id(model: str) -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model=model,
    )
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[],
            channels={
                OPENAI_REASONING_ENCRYPTED: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "schema": "fast-agent.openai-responses.reasoning-replay",
                                "version": 1,
                                "item": {
                                    "type": "reasoning",
                                    "id": "rs_reused",
                                    "summary": [
                                        {
                                            "type": "summary_text",
                                            "text": f"summary-{ciphertext}",
                                        }
                                    ],
                                    "encrypted_content": ciphertext,
                                },
                            }
                        ),
                    )
                ]
            },
        )
        for ciphertext in ("cipher-turn-1", "cipher-turn-2", "cipher-turn-2")
    ]

    input_items = llm._convert_to_provider_format(messages)
    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)
    planned = llm._new_ws_request_planner().plan(args)
    reasoning = [item for item in planned.arguments["input"] if item["type"] == "reasoning"]

    assert [item["encrypted_content"] for item in reasoning] == [
        "cipher-turn-1",
        "cipher-turn-2",
    ]
    assert [item["id"] for item in reasoning] == ["rs_reused", "rs_reused"]
    assert [item["summary"][0]["text"] for item in reasoning] == [
        "summary-cipher-turn-1",
        "summary-cipher-turn-2",
    ]
    assert all("schema" not in item and "version" not in item for item in reasoning)


def test_xai_responses_advertises_web_search() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm.web_search_supported is True
    assert llm.web_search_enabled is False


def test_xai_responses_builds_web_search_tool_when_enabled() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    web_search=XAIWebSearchSettings(enabled=True),
                )
            )
        ),
        model="grok-4.3",
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert args["tools"] == [{"type": "web_search"}]
    assert args["include"] == ["reasoning.encrypted_content"]


def test_xai_responses_builds_xai_web_search_options() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    web_search=XAIWebSearchSettings(
                        enabled=True,
                        excluded_domains=["example.com"],
                        enable_image_understanding=True,
                    ),
                )
            )
        ),
        model="grok-4.3",
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert args["tools"] == [
        {
            "type": "web_search",
            "filters": {"excluded_domains": ["example.com"]},
            "enable_image_understanding": True,
        }
    ]


def test_xai_web_search_rejects_conflicting_domain_filters() -> None:
    with pytest.raises(ValueError):
        XAIWebSearchSettings(
            allowed_domains=["example.com"],
            excluded_domains=["example.org"],
        )


def test_xai_web_search_domain_filter_limit() -> None:
    domains = [f"domain-{index}.example.com" for index in range(6)]

    with pytest.raises(ValueError, match="at most 5 domains"):
        XAIWebSearchSettings(allowed_domains=domains)

    with pytest.raises(ValueError, match="at most 5 domains"):
        XAIWebSearchSettings(excluded_domains=domains)


def test_xai_responses_advertises_x_search() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm.x_search_supported is True
    assert llm.x_search_enabled is False


def test_xai_responses_builds_x_search_tool_when_enabled() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert args["tools"] == [{"type": "x_search"}]


def test_xai_responses_stream_renders_x_search_internal_calls_as_remote_tools() -> None:
    harness = _XAIStreamingHarness()

    handled = harness._handle_responses_output_item_added(
        event=SimpleNamespace(
            output_index=0,
            item_id="fc_1",
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="xs_1",
                name="x_keyword_search",
            ),
        ),
        tool_state=OpenAIToolStreamState(),
        notified_tool_indices=set(),
        model="grok-4.3",
    )

    assert handled is True
    assert len(harness.events) == 1
    event_type, payload = harness.events[0]
    assert event_type == "start"
    assert payload["tool_name"] == "x_keyword_search"
    assert payload["presentation_family"] == "remote_tool"
    assert payload["preserve_details"] is True
    assert payload["tool_display_name"] == "remote tool: x_keyword_search"


def test_xai_responses_filters_x_search_internal_function_calls() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="x_keyword_search",
                arguments='{"query":"evalstate"}',
            )
        ],
    )

    assert llm._extract_tool_calls(response) is None


def test_xai_responses_records_x_search_internal_calls_as_server_metadata() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                id="ctc_1",
                call_id="xs_1",
                name="x_keyword_search",
                input='{"query":"from:evalstate","limit":"5"}',
                status="completed",
            )
        ],
    )

    payloads = llm._extract_provider_mcp_metadata(response)

    assert len(payloads) == 1
    assert isinstance(payloads[0], TextContent)
    payload = json.loads(payloads[0].text)
    assert payload == {
        "type": "server_tool_use",
        "provider_tool_type": "x_search_call",
        "name": "x_keyword_search",
        "id": "xs_1",
        "status": "completed",
        "arguments": '{"query":"from:evalstate","limit":"5"}',
        "input": {"query": "from:evalstate", "limit": "5"},
    }


def test_xai_responses_records_clean_x_search_tool_use_id() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                id="  ctc_1  ",
                call_id=123,
                name="x_keyword_search",
                input="{}",
            )
        ],
    )

    payloads = llm._extract_provider_mcp_metadata(response)

    assert len(payloads) == 1
    assert isinstance(payloads[0], TextContent)
    payload = json.loads(payloads[0].text)
    assert payload["id"] == "ctc_1"


def test_xai_responses_falls_back_to_arguments_when_input_is_not_text() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                id="ctc_1",
                name="x_keyword_search",
                input={"query": "ignored-provider-shape"},
                arguments='{"query":"from:evalstate"}',
            )
        ],
    )

    payloads = llm._extract_provider_mcp_metadata(response)

    assert len(payloads) == 1
    assert isinstance(payloads[0], TextContent)
    payload = json.loads(payloads[0].text)
    assert payload["arguments"] == '{"query":"from:evalstate"}'
    assert payload["input"] == {"query": "from:evalstate"}


def test_xai_responses_preserves_regular_function_calls_when_x_search_enabled() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="local_tool",
                arguments='{"value":1}',
            )
        ],
    )

    tool_calls = llm._extract_tool_calls(response)

    assert tool_calls is not None
    assert tool_calls["call_1"].params.name == "local_tool"
