"""Native Gemini 4 registration contracts; no API requests are made."""

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("variant", ["argon", "hydrogen", "carbon"])
@pytest.mark.parametrize("form", ["short", "alias", "canonical", "qualified"])
def test_native_gemini4_names_resolve_with_capabilities(variant: str, form: str) -> None:
    canonical = f"gemini-4-{variant}"
    names = {
        "short": variant,
        "alias": f"gemini4{variant}",
        "canonical": canonical,
        "qualified": f"google.{canonical}",
    }
    resolved = ModelFactory.resolve_model_spec(names[form])

    assert resolved.provider is Provider.GOOGLE
    assert resolved.wire_model_name == canonical
    assert resolved.context_window == 1_048_576
    assert resolved.max_output_tokens == 65_536
    assert resolved.json_mode == "schema"
    assert resolved.google_service_tiers == ("flex",)
    assert resolved.reasoning_mode == "google_thinking"


@pytest.mark.parametrize("variant", ["argon", "hydrogen", "carbon"])
def test_native_gemini4_factory_exposes_reasoning_defaults_and_flex(variant: str) -> None:
    factory = ModelFactory.create_factory(f"gemini4{variant}?service_tier=flex")
    llm = factory(LlmAgent(AgentConfig(name="Gemini registration test")))

    assert isinstance(llm, GoogleNativeLLM)
    assert llm.default_request_params.service_tier == "flex"
    spec = llm.reasoning_effort_spec
    assert spec is not None
    assert spec.default == ReasoningEffortSetting(
        kind="effort", value="high" if variant == "hydrogen" else "medium"
    )
    expected_levels = ["low", "medium", "high"]
    if variant == "hydrogen":
        expected_levels.append("xhigh")
    assert spec.allowed_efforts == expected_levels
    assert not spec.allow_toggle_disable
    assert not spec.allow_auto


@pytest.mark.parametrize("variant", ["argon", "hydrogen", "carbon"])
@pytest.mark.parametrize(
    "mime", ["text/plain", "image/png", "application/pdf", "audio/mpeg", "video/mp4"]
)
def test_native_gemini4_accepts_documented_inputs(variant: str, mime: str) -> None:
    assert ModelDatabase.supports_mime(f"gemini-4-{variant}", mime)


@pytest.mark.parametrize("variant", ["argon", "hydrogen", "carbon"])
def test_native_gemini4_keeps_existing_tool_contract(variant: str) -> None:
    resolved = ModelFactory.resolve_model_spec(f"gemini4{variant}")
    previous = ModelFactory.resolve_model_spec("gemini38")

    assert resolved.model_params is not None
    assert previous.model_params is not None
    assert resolved.model_params.google_search_supported
    assert resolved.structured_tool_policy == previous.structured_tool_policy
    assert resolved.model_params.shell_tool_profile == previous.model_params.shell_tool_profile
    assert resolved.model_params.shell_edit_tool == previous.model_params.shell_edit_tool


@pytest.mark.parametrize("route", ["google", "copilot"])
@pytest.mark.parametrize("variant", ["argon", "hydrogen", "carbon"])
def test_gemini4_ptc_is_opt_out(route: str, variant: str) -> None:
    model = f"{route}.gemini-4-{variant}"
    assert ModelFactory.resolve_model_spec(model).resolve_ptc_enabled()
    assert not ModelFactory.resolve_model_spec(f"{model}?ptc=off").resolve_ptc_enabled()


def test_native_gemini4_picker_entries_dispatch_without_changing_default_alias() -> None:
    entries = ModelSelectionCatalog.list_current_entries(Provider.GOOGLE)
    gemini4_entries = [entry for entry in entries if entry.model.startswith("google.gemini-4-")]

    assert {entry.model for entry in gemini4_entries} == {
        "google.gemini-4-argon",
        "google.gemini-4-hydrogen",
        "google.gemini-4-carbon",
    }
    for entry in gemini4_entries:
        from_alias = ModelFactory.resolve_model_spec(entry.alias)
        from_model = ModelFactory.resolve_model_spec(entry.model)
        assert from_alias.provider is from_model.provider is Provider.GOOGLE
        assert from_alias.wire_model_name == from_model.wire_model_name
        assert entry.fast is (from_model.wire_model_name == "gemini-4-hydrogen")
        assert ModelDatabase.is_fast_model(entry.model) is entry.fast
    default = ModelFactory.resolve_model_spec("gemini")
    assert default.provider is Provider.GOOGLE
    assert default.wire_model_name == "gemini-3.8-flash"


@pytest.mark.parametrize("variant", ["argon", "hydrogen", "carbon"])
@pytest.mark.parametrize(
    "query", ["", "?web_search=on", "?web_search=off", "?service_tier=flex&web_search=on"]
)
def test_native_gemini4_factory_supports_optional_flex_and_search(variant: str, query: str) -> None:
    factory = ModelFactory.create_factory(f"gemini4{variant}{query}")
    llm = factory(LlmAgent(AgentConfig(name="Gemini optional capabilities test")))

    assert isinstance(llm, GoogleNativeLLM)
    assert llm.service_tier_supported
    assert llm.available_service_tiers == ("flex",)
    assert llm.service_tier == ("flex" if "service_tier=flex" in query else None)
    assert llm.web_search_supported
    assert llm.web_search_enabled is ("web_search=on" in query)
    assert llm.get_request_params().service_tier == llm.service_tier

    llm.set_service_tier("flex")
    llm.set_web_search_enabled(True)
    assert llm.service_tier == "flex"
    assert llm.web_search_enabled

    llm.set_service_tier(None)
    llm.set_web_search_enabled(None)
    assert llm.service_tier is None
    assert not llm.web_search_enabled
