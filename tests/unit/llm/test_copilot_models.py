"""Public Copilot model/config contracts, independent of network access."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from fast_agent.config import CopilotSettings, Settings
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_database import (
    COPILOT_ROUTE_OVERRIDE_FIELDS,
    ModelDatabase,
    ModelParameters,
)
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.copilot.models import COPILOT_MODELS, get_copilot_model
from fast_agent.llm.provider_types import Provider

PUBLIC_IDS = {
    "claude-haiku-4.5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-fable-5",
    "claude-fable-5.1",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-6-astra",
}


def test_public_registry_contract():
    assert set(COPILOT_MODELS) == PUBLIC_IDS
    assert Provider.COPILOT.value == "copilot"
    assert Provider.COPILOT.display_name == "GitHub Copilot"
    assert not get_copilot_model("claude-fable-5.1").supports_named_tool_choice
    spec = get_copilot_model("claude-haiku-4.5")
    with pytest.raises(FrozenInstanceError):
        spec.model_id = "changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(ModelConfigError):
        get_copilot_model("unknown")


@pytest.mark.parametrize("model_id", sorted(PUBLIC_IDS))
def test_model_resolution_and_transport(model_id):
    public_id = f"copilot.{model_id}"
    resolved = ModelFactory.resolve_model_spec(public_id)
    assert resolved.provider == Provider.COPILOT
    assert resolved.wire_model_name == model_id
    spec = get_copilot_model(model_id)
    assert resolved.model_config.transport == (
        "websocket" if spec.wire_api == "responses" else "sse"
    )
    assert ModelFactory.parse_model_string(public_id + "?transport=sse").transport == "sse"
    if spec.wire_api == "responses":
        assert ModelFactory.parse_model_string(public_id + "?transport=auto").transport == "auto"
        assert (
            ModelFactory.parse_model_string(public_id + "?transport=websocket").transport
            == "websocket"
        )
    else:
        with pytest.raises(ModelConfigError):
            ModelFactory.parse_model_string(public_id + "?transport=websocket")
    assert ModelDatabase.get_response_transports(public_id) == spec.transports


def test_provider_scoped_metadata():
    native = ModelDatabase.get_model_params("gpt-6-astra")
    copilot = ModelDatabase.get_model_params("copilot.gpt-6-astra")
    assert native is not None and copilot is not None
    assert native.default_provider != Provider.COPILOT
    assert copilot.default_provider == Provider.COPILOT
    assert ModelDatabase.get_default_provider("gpt-5.6-terra") != Provider.COPILOT
    assert "claude-haiku-4.5" not in ModelDatabase.MODELS
    with pytest.raises(ModelConfigError):
        ModelFactory.parse_model_string("copilot.unknown")


def _copilot_pairs() -> list[tuple[str, str]]:
    """(copilot public id, base catalog id) for every Copilot model with a catalog entry."""
    pairs = []
    for model_id in COPILOT_MODELS:
        for base_id in (model_id, model_id.replace(".", "-")):
            if base_id in ModelDatabase.MODELS:
                pairs.append((f"copilot.{model_id}", base_id))
                break
    assert ("copilot.claude-opus-5", "claude-opus-5") in pairs
    return pairs


@pytest.mark.parametrize(("public_id", "base_id"), _copilot_pairs())
def test_copilot_route_inherits_base_catalog_entry(public_id: str, base_id: str) -> None:
    """Same model, different route: everything but the route-specific fields is inherited."""
    copilot = ModelDatabase.get_model_params(public_id)
    base = ModelDatabase.get_model_params(base_id)
    assert copilot is not None and base is not None
    inherited = set(ModelParameters.model_fields) - COPILOT_ROUTE_OVERRIDE_FIELDS
    assert inherited, "invariant must cover fields"
    assert "cache_ttl" in inherited
    assert copilot.model_dump(include=inherited) == base.model_dump(include=inherited)
    assert copilot.default_provider == Provider.COPILOT
    assert copilot.response_transports == get_copilot_model(public_id[len("copilot.") :]).transports
    assert copilot.response_service_tiers == ()
    assert not copilot.codex_responses_lite
    assert copilot.long_context_window is None
    assert copilot.anthropic_web_search_version is None
    assert copilot.anthropic_web_fetch_version is None
    assert copilot.anthropic_required_betas is None


def test_copilot_opus_keeps_anthropic_model_specific_prompt() -> None:
    expected = ModelDatabase.get_model_specific("claude-opus-5")
    assert expected
    assert ModelDatabase.get_model_specific("copilot.claude-opus-5") == expected


def test_mixed_factory_dispatch_ignores_bare_overrides(monkeypatch):
    messages = Mock()
    responses = Mock()
    modules = {
        "fast_agent.llm.provider.copilot.messages": SimpleNamespace(CopilotMessagesLLM=messages),
        "fast_agent.llm.provider.copilot.responses": SimpleNamespace(CopilotResponsesLLM=responses),
    }
    monkeypatch.setattr("fast_agent.llm.model_factory.import_module", modules.__getitem__)
    monkeypatch.setitem(ModelFactory.MODEL_SPECIFIC_CLASSES, "claude-sonnet-5", Mock())
    agent = Mock()
    for model_id in ("claude-sonnet-5", "gpt-6-astra", "claude-sonnet-5"):
        ModelFactory.create_factory(f"copilot.{model_id}")(agent)
    assert messages.call_count == 2
    assert responses.call_count == 1
    assert Provider.COPILOT not in ModelFactory.PROVIDER_CLASSES


def test_settings_defaults_and_validation():
    first = Settings()
    second = Settings()
    assert first.copilot == CopilotSettings()
    assert first.copilot is not second.copilot
    assert first.copilot.base_url == "https://api.githubcopilot.com"
    assert first.copilot.integration_id == "copilot-sdk"
    assert first.copilot.runtime_timeout_seconds == 30
    assert first.copilot.cache_mode == "auto"
    assert first.copilot.cache_ttl is None
    assert set(CopilotSettings.model_fields) == {
        "base_url",
        "integration_id",
        "runtime_timeout_seconds",
        "cache_mode",
        "cache_ttl",
    }
    for timeout in (0, -1):
        with pytest.raises(ValidationError):
            CopilotSettings(runtime_timeout_seconds=timeout)
    configured = Settings(copilot=CopilotSettings(base_url="https://copilot.invalid/"))
    assert configured.copilot.base_url == "https://copilot.invalid"


@pytest.mark.parametrize(
    "integration_id",
    ["", " ", "bad id", "bad\tid", "bad\r\nid", "sdk\n", "sdk\x00", "sdk\x7f", "café", None],
)
def test_integration_id_must_be_a_nonempty_ascii_header_value(integration_id: object) -> None:
    with pytest.raises(ValidationError):
        CopilotSettings.model_validate({"integration_id": integration_id})


def test_integration_id_can_be_configured_in_settings_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        Settings.model_validate(
            {"copilot": {"integration_id": "partner-agent"}}
        ).copilot.integration_id
        == "partner-agent"
    )
    monkeypatch.setenv("COPILOT__INTEGRATION_ID", "partner-agent")
    assert Settings().copilot.integration_id == "partner-agent"


@pytest.mark.parametrize(
    "field", ["backend", "cli_path", "use_environment_token", "direct_base_url", "api_key"]
)
def test_obsolete_and_secret_settings_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted") as error:
        Settings.model_validate({"copilot": {field: "obsolete"}})
    assert error.value.errors()[0]["loc"] == ("copilot", field)


@pytest.mark.parametrize(
    "field", ["BACKEND", "CLI_PATH", "USE_ENVIRONMENT_TOKEN", "DIRECT_BASE_URL"]
)
def test_obsolete_environment_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setenv(f"COPILOT__{field}", "obsolete")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings()


@pytest.mark.parametrize("model_id", sorted(PUBLIC_IDS))
@pytest.mark.parametrize("tier", ["fast", "flex"])
def test_copilot_factory_rejects_service_tiers(model_id: str, tier: str) -> None:
    with pytest.raises(ModelConfigError, match="service_tier"):
        ModelFactory.create_factory(f"copilot.{model_id}?service_tier={tier}")


@pytest.mark.parametrize("mode", ["off", "prompt", "auto"])
@pytest.mark.parametrize("ttl", [None, "5m", "1h"])
def test_copilot_cache_settings_accept_supported_values(mode: str, ttl: str | None) -> None:
    settings = CopilotSettings.model_validate({"cache_mode": mode, "cache_ttl": ttl})
    assert settings.cache_mode == mode
    assert settings.cache_ttl == ttl


@pytest.mark.parametrize(
    "values",
    [{"cache_mode": "on"}, {"cache_mode": None}, {"cache_ttl": "10m"}, {"cache_ttl": 300}],
)
def test_copilot_cache_settings_reject_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CopilotSettings.model_validate(values)
