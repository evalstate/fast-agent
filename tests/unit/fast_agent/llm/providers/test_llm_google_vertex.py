import json
import types
from typing import TYPE_CHECKING, cast

import pytest
from google.genai import types as google_types
from mcp import Tool
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent
from pydantic import BaseModel

from fast_agent.config import GoogleSettings, Settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.llm.provider.google.llm_google_native import (
    GOOGLE_DIAGNOSTICS_CHANNEL,
    GoogleNativeLLM,
)
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.llm.request_params import StructuredToolPolicy


class StructuredSample(BaseModel):
    name: str
    count: int = 3
    tags: dict[str, str]


def _build_llm(config: Settings) -> GoogleNativeLLM:
    """Create a Google LLM instance with the provided config."""
    return GoogleNativeLLM(context=Context(config=config))


def test_vertex_cfg_accepts_model_object_and_expands_model_names() -> None:
    """Vertex config may arrive as a pydantic model with a custom attr object."""
    google_settings = GoogleSettings()
    setattr(
        google_settings,
        "vertex_ai",
        types.SimpleNamespace(enabled=True, project_id="proj", location="loc"),
    )
    config = Settings(google=google_settings)

    llm = _build_llm(config)
    enabled, project_id, location = llm._vertex_cfg()

    assert enabled is True
    assert project_id == "proj"
    assert location == "loc"

    resolved = llm._resolve_model_name("gemini-2.5-flash")
    assert (
        resolved
        == "projects/proj/locations/loc/publishers/google/models/gemini-2.5-flash"
    )


def test_vertex_cfg_accepts_dict_and_provider_key_manager_allows_adc() -> None:
    """Vertex config may also arrive as a dict after merging secrets/model_dump."""
    config = Settings.model_validate(
        {
            "google": {
                "vertex_ai": {
                    "enabled": True,
                    "project_id": "proj",
                    "location": "europe-west4",
                }
            }
        }
    )

    llm = _build_llm(config)
    enabled, project_id, location = llm._vertex_cfg()

    assert enabled is True
    assert project_id == "proj"
    assert location == "europe-west4"

    resolved = llm._resolve_model_name("gemini-3-flash-preview")
    assert resolved.endswith("gemini-3-flash-preview")
    assert resolved.startswith(
        "projects/proj/locations/europe-west4/publishers/google/models/"
    )

    # When Vertex is enabled, no API key should be required (ADC path).
    assert ProviderKeyManager.get_api_key("google", config) == ""


def test_vertex_partner_model_names_are_not_rewritten_to_google_publisher() -> None:
    """Vertex partner models should keep the provider-native model id."""
    config = Settings.model_validate(
        {
            "google": {
                "vertex_ai": {
                    "enabled": True,
                    "project_id": "proj",
                    "location": "global",
                }
            }
        }
    )

    llm = _build_llm(config)

    assert llm._resolve_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert (
        llm._resolve_model_name("publishers/anthropic/models/claude-sonnet-4-6")
        == "publishers/anthropic/models/claude-sonnet-4-6"
    )


def test_vertex_first_party_non_gemini_models_are_rewritten_to_google_publisher() -> None:
    config = Settings.model_validate(
        {
            "google": {
                "vertex_ai": {
                    "enabled": True,
                    "project_id": "proj",
                    "location": "global",
                }
            }
        }
    )

    llm = _build_llm(config)

    assert (
        llm._resolve_model_name("text-embedding-005")
        == "projects/proj/locations/global/publishers/google/models/text-embedding-005"
    )


def test_initialize_google_client_prefers_vertex_with_dict_config(monkeypatch) -> None:
    """Ensure dict-based vertex config builds a Vertex client (ADC, no API key)."""
    config = Settings.model_validate(
        {
            "google": {
                "vertex_ai": {
                    "enabled": True,
                    "project_id": "proj",
                    "location": "europe-west4",
                }
            }
        }
    )
    llm = _build_llm(config)

    called: dict[str, dict] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            called["kwargs"] = kwargs

    monkeypatch.setattr(
        "fast_agent.llm.provider.google.llm_google_native.genai.Client", FakeClient
    )

    client = llm._initialize_google_client()

    assert isinstance(client, FakeClient)
    assert called["kwargs"]["vertexai"] is True
    assert called["kwargs"]["project"] == "proj"
    assert called["kwargs"]["location"] == "europe-west4"


def test_structured_schema_with_tools_is_deferred_until_tool_result() -> None:
    llm = _build_llm(Settings())
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    tool = Tool(
        name="lookup_probe_payload",
        description="Return the probe payload for validation.",
        inputSchema={"type": "object", "properties": {}},
    )
    params = RequestParams(structured_schema=schema, structured_tool_policy="defer")

    _, prepared_params = llm._prepare_structured_request(
        [Prompt.user("call the tool, then return json")],
        params,
        [tool],
    )

    assert params.structured_schema == schema
    assert prepared_params.structured_schema is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "expected_tools"),
    [
        ("auto", True),
        ("always", True),
        ("no_tools", False),
    ],
)
@pytest.mark.asyncio
async def test_structured_schema_in_generate_path_can_keep_google_tools(
    policy: str, expected_tools: bool
) -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    captured: dict[str, object] = {}

    class FakeModels:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)
            return google_types.GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": '{"answer":"ok"}'}],
                            },
                            "finish_reason": "STOP",
                        }
                    ]
                }
            )

    class FakeAio:
        def __init__(self) -> None:
            self.models = FakeModels()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.aio = FakeAio()

    class Harness(GoogleNativeLLM):
        def _initialize_google_client(self):
            return FakeClient()

    llm = Harness(context=Context(config=Settings()), model="gemini-2.0-flash")
    response = await llm._google_completion(
        [google_types.Content(role="user", parts=[google_types.Part.from_text(text="answer")])],
        request_params=RequestParams(
            model="gemini-2.0-flash",
            structured_schema=schema,
            structured_tool_policy=cast("StructuredToolPolicy", policy),
        ),
        tools=[
            Tool(
                name="lookup_probe_payload",
                description="Return the probe payload for validation.",
                inputSchema={"type": "object", "properties": {}},
            )
        ],
    )

    config = cast("google_types.GenerateContentConfig", captured["config"])
    assert config.response_mime_type == "application/json"
    assert config.response_schema is not None
    assert bool(config.tools) is expected_tools
    assert response.last_text() == '{"answer":"ok"}'


@pytest.mark.asyncio
async def test_structured_model_path_passes_pydantic_model_to_google_sdk() -> None:
    captured: dict[str, object] = {}

    class FakeModels:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)
            return google_types.GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "text": (
                                            '{"name":"Ada","count":3,'
                                            '"tags":{"role":"engineer"}}'
                                        )
                                    }
                                ],
                            },
                            "finish_reason": "STOP",
                        }
                    ]
                }
            )

    class FakeAio:
        def __init__(self) -> None:
            self.models = FakeModels()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.aio = FakeAio()

    class Harness(GoogleNativeLLM):
        def _initialize_google_client(self):
            return FakeClient()

    llm = Harness(context=Context(config=Settings()), model="gemini-2.0-flash")

    parsed, response = await llm._apply_prompt_provider_specific_structured(
        [Prompt.user("return json")],
        StructuredSample,
        RequestParams(model="gemini-2.0-flash"),
    )

    config = cast("google_types.GenerateContentConfig", captured["config"])
    assert config.response_mime_type == "application/json"
    assert config.response_schema is StructuredSample
    assert isinstance(parsed, StructuredSample)
    assert parsed.tags == {"role": "engineer"}
    assert response.last_text() == '{"name":"Ada","count":3,"tags":{"role":"engineer"}}'


@pytest.mark.asyncio
async def test_structured_schema_in_generate_path_returns_google_tool_calls() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    class FakeModels:
        async def generate_content(self, **kwargs):
            return google_types.GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "function_call": {
                                            "id": "call_lookup",
                                            "name": "lookup_probe_payload",
                                            "args": {},
                                        }
                                    }
                                ],
                            },
                            "finish_reason": "STOP",
                        }
                    ]
                }
            )

    class FakeAio:
        def __init__(self) -> None:
            self.models = FakeModels()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.aio = FakeAio()

    class Harness(GoogleNativeLLM):
        def _initialize_google_client(self):
            return FakeClient()

    llm = Harness(context=Context(config=Settings()), model="gemini-2.0-flash")
    response = await llm._google_completion(
        [google_types.Content(role="user", parts=[google_types.Part.from_text(text="answer")])],
        request_params=RequestParams(
            model="gemini-2.0-flash",
            structured_schema=schema,
            structured_tool_policy="always",
        ),
        tools=[
            Tool(
                name="lookup_probe_payload",
                description="Return the probe payload for validation.",
                inputSchema={"type": "object", "properties": {}},
            )
        ],
    )

    assert response.tool_calls
    assert list(response.tool_calls) == ["call_lookup"]
    [tool_call] = response.tool_calls.values()
    assert tool_call.params.name == "lookup_probe_payload"
    assert response.stop_reason == "toolUse"


@pytest.mark.asyncio
async def test_tool_result_request_preserves_google_function_call_id_and_history() -> None:
    captured: dict[str, object] = {}

    class FakeModels:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)
            return google_types.GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "The weather is sunny."}],
                            },
                            "finish_reason": "STOP",
                        }
                    ]
                }
            )

    class FakeAio:
        def __init__(self) -> None:
            self.models = FakeModels()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.aio = FakeAio()

    class Harness(GoogleNativeLLM):
        async def _stream_generate_content(self, **kwargs):
            return None

        def _initialize_google_client(self):
            return FakeClient()

    llm = Harness(context=Context(config=Settings()), model="gemini-3.5-flash")
    llm.history.set(
        [
            google_types.Content(role="user", parts=[google_types.Part.from_text(text="weather")]),
            google_types.Content(
                role="model",
                parts=[
                    google_types.Part(
                        text="thinking",
                        thought=True,
                        thought_signature=b"signature",
                    ),
                    google_types.Part(
                        function_call=google_types.FunctionCall(
                            id="call_weather",
                            name="weather",
                            args={"city": "Paris"},
                        )
                    ),
                ],
            ),
        ]
    )

    await llm._apply_prompt_provider_specific(
        [
            Prompt.user("weather"),
            Prompt.assistant(
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call_weather": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="weather",
                            arguments={"city": "Paris"},
                        ),
                    )
                },
            ),
            PromptMessageExtended(
                role="user",
                content=[],
                tool_results={
                    "call_weather": CallToolResult(
                        content=[TextContent(type="text", text="Sunny")],
                        isError=False,
                    )
                },
            ),
        ],
        request_params=RequestParams(model="gemini-3.5-flash"),
        tools=[
            Tool(
                name="weather",
                description="Check weather",
                inputSchema={"type": "object", "properties": {}},
            )
        ],
    )

    contents = cast("list[google_types.Content]", captured["contents"])
    assert [content.role for content in contents] == ["user", "model", "user", "model"]

    model_parts = contents[1].parts or []
    assert model_parts[0].thought is True
    assert model_parts[0].thought_signature == b"signature"
    assert model_parts[1].function_call is not None
    assert model_parts[1].function_call.id == "call_weather"

    response_parts = contents[2].parts or []
    fn_response = response_parts[0].function_response
    assert fn_response is not None
    assert fn_response.id == "call_weather"
    assert fn_response.name == "weather"
    assert fn_response.response == {"result": "Sunny"}


@pytest.mark.asyncio
async def test_google_thought_parts_are_preserved_as_reasoning_channel() -> None:
    class FakeModels:
        async def generate_content(self, **kwargs):
            return google_types.GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"text": "private analysis", "thought": True},
                                    {"text": "final answer"},
                                ],
                            },
                            "finish_reason": "STOP",
                        }
                    ]
                }
            )

    class FakeAio:
        def __init__(self) -> None:
            self.models = FakeModels()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.aio = FakeAio()

    class Harness(GoogleNativeLLM):
        async def _stream_generate_content(self, **kwargs):
            return None

        def _initialize_google_client(self):
            return FakeClient()

    llm = Harness(context=Context(config=Settings()), model="gemini-3.5-flash")
    response = await llm._google_completion(
        [google_types.Content(role="user", parts=[google_types.Part.from_text(text="hello")])],
        request_params=RequestParams(model="gemini-3.5-flash"),
    )

    assert response.last_text() == "final answer"
    assert response.channels is not None
    reasoning = response.channels[REASONING]
    assert len(reasoning) == 1
    assert isinstance(reasoning[0], TextContent)
    assert reasoning[0].text == "private analysis"
    diagnostics = response.channels[GOOGLE_DIAGNOSTICS_CHANNEL]
    assert len(diagnostics) == 1
    assert isinstance(diagnostics[0], TextContent)
    payload = json.loads(diagnostics[0].text)
    assert payload["transport"] == "google-genai"
    assert payload["request_type"] == "models.generate_content"
    assert payload["streaming"] is False
    assert payload["model"] == "gemini-3.5-flash"
    assert payload["phase_ms"]["total"] >= 0
