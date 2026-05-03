import types
from typing import TYPE_CHECKING, cast

import pytest
from google.genai import types as google_types
from mcp import Tool

from fast_agent.config import GoogleSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from fast_agent.llm.request_params import StructuredToolPolicy


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
    [tool_call] = response.tool_calls.values()
    assert tool_call.params.name == "lookup_probe_payload"
    assert response.stop_reason == "toolUse"
