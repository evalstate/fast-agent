import types

from fast_agent.config import GoogleSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM
from fast_agent.llm.provider_key_manager import ProviderKeyManager


def _build_llm(config: Settings) -> GoogleNativeLLM:
    """Create a Google LLM instance with the provided config."""
    return GoogleNativeLLM(context=Context(config=config))


def test_vertex_cfg_accepts_model_object_and_resolves_preview_names() -> None:
    """Vertex config may arrive as a pydantic model with a custom attr object."""
    google_settings = GoogleSettings()
    google_settings.vertex_ai = types.SimpleNamespace(enabled=True, project_id="proj", location="loc")
    config = Settings(google=google_settings)

    llm = _build_llm(config)
    enabled, project_id, location = llm._vertex_cfg()

    assert enabled is True
    assert project_id == "proj"
    assert location == "loc"

    resolved = llm._resolve_model_name("gemini-2.5-flash-preview-09-2025")
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

    resolved = llm._resolve_model_name("gemini-2.5-flash-preview-09-2025")
    assert resolved.endswith("gemini-2.5-flash")
    assert resolved.startswith(
        "projects/proj/locations/europe-west4/publishers/google/models/"
    )

    # When Vertex is enabled, no API key should be required (ADC path).
    assert ProviderKeyManager.get_api_key("google", config) == ""


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
