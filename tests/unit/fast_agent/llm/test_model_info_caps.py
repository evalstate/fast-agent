import pathlib
import sys
import types
from typing import TYPE_CHECKING, cast

sys.path.append(str(pathlib.Path(__file__).resolve().parents[4] / "src"))

if "a2a" not in sys.modules:
    a2a_module = types.ModuleType("a2a")
    types_module = types.ModuleType("a2a.types")

    class AgentCard:  # minimal stub for imports
        pass

    setattr(types_module, "AgentCard", AgentCard)
    setattr(a2a_module, "types", types_module)
    sys.modules["a2a"] = a2a_module
    sys.modules["a2a.types"] = types_module

from fast_agent.llm.model_factory import ModelConfig
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.resolved_model import ResolvedModelSpec

if TYPE_CHECKING:
    from fast_agent.interfaces import FastAgentLLMProtocol


class DummyLLM:
    def __init__(self, model: str, provider: Provider = Provider.GOOGLE) -> None:
        self.model_name = model
        self.provider = provider
        self.resolved_model = ResolvedModelSpec(
            raw_input=model,
            selected_model_name=model,
            source="direct",
            model_config=ModelConfig(provider=provider, model_name=model),
            provider=provider,
            wire_model_name=model,
        )
        self.default_request_params = type("Params", (), {"model": model})()

    @property
    def model_info(self) -> "ModelInfo | None":
        if not self.model_name:
            return None
        return ModelInfo.from_name(self.model_name, self.provider)


class DummyAgent:
    def __init__(self, model: str, provider: Provider = Provider.GOOGLE) -> None:
        self.llm = DummyLLM(model, provider=provider)


def test_model_alias_capabilities_match_canonical() -> None:
    alias = ModelInfo.from_name("gemini25")
    canonical = ModelInfo.from_name("gemini-2.5-flash-preview-09-2025")

    assert alias is not None
    assert canonical is not None
    assert alias.name == canonical.name
    assert alias.tokenizes == canonical.tokenizes
    assert alias.tdv_flags == (True, True, True)


def test_model_info_from_llm_uses_canonical_name() -> None:
    info = ModelInfo.from_llm(cast("FastAgentLLMProtocol", DummyLLM("gemini25")))
    assert info is not None
    assert info.name == "gemini-2.5-flash-preview-09-2025"
    assert info.tdv_flags == (True, True, True)


def test_model_info_from_agent_llm_capabilities() -> None:
    agent = DummyAgent("gemini-2.5-pro", provider=Provider.GOOGLE)
    info = ModelInfo.from_llm(cast("FastAgentLLMProtocol", agent.llm))
    assert info is not None
    assert info.name == "gemini-2.5-pro"
    assert info.tdv_flags == (True, True, True)


def test_model_info_from_resolved_model_uses_resolved_metadata() -> None:
    resolved_model = ResolvedModelSpec(
        raw_input="haikutiny",
        selected_model_name="haikutiny",
        source="overlay",
        model_config=ModelConfig(provider=Provider.ANTHROPIC, model_name="claude-haiku-4-5"),
        provider=Provider.ANTHROPIC,
        wire_model_name="claude-haiku-4-5",
    )

    info = ModelInfo.from_resolved_model(resolved_model)

    assert info is not None
    assert info.name == "claude-haiku-4-5"
    assert info.provider == Provider.ANTHROPIC


def test_unknown_model_defaults_to_text_only() -> None:
    info = ModelInfo.from_name("unknown-model-id")
    assert info is not None
    assert info.tdv_flags == (True, False, False)


def test_codexspark_alias_is_text_only() -> None:
    info = ModelInfo.from_name("codexspark")
    assert info is not None
    assert info.name == "codexresponses.gpt-5.3-codex-spark"
    assert info.tdv_flags == (True, False, False)


def test_model_info_supports_overlay_tokenizes() -> None:
    info = ModelInfo(
        name="unsloth/Qwen3.5-9B-GGUF",
        provider=Provider.OPENRESPONSES,
        context_window=75264,
        max_output_tokens=2048,
        tokenizes=["text/plain", "image/jpeg", "image/png", "image/webp"],
        json_mode=None,
        reasoning=None,
    )

    assert info.supports_mime("image/png")
    assert info.supports_vision


def test_model_info_openai_chat_documents_remain_pdf_only() -> None:
    info = ModelInfo.from_name("gpt-4o", provider=Provider.OPENAI)

    assert info is not None
    assert info.supports_mime("application/pdf")
    assert not info.supports_mime(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_model_info_responses_models_support_office_documents() -> None:
    info = ModelInfo.from_name("o4-mini", provider=Provider.RESPONSES)

    assert info is not None
    assert info.supports_mime(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_model_info_anthropic_models_support_office_documents() -> None:
    info = ModelInfo.from_name("claude-sonnet-4-5", provider=Provider.ANTHROPIC)

    assert info is not None
    assert info.supports_mime(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert not info.supports_mime(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        resource_source="link",
    )
    assert info.supports_mime("image/png", resource_source="link")


def test_model_info_anthropic_vertex_models_do_not_support_office_documents() -> None:
    info = ModelInfo.from_name("claude-sonnet-4-5", provider=Provider.ANTHROPIC_VERTEX)

    assert info is not None
    assert info.supports_mime("application/pdf")
    assert not info.supports_mime(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
