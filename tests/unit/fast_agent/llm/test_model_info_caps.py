import pathlib
import sys
import types

sys.path.append(str(pathlib.Path(__file__).resolve().parents[4] / "src"))

if "a2a" not in sys.modules:
    a2a_module = types.ModuleType("a2a")
    types_module = types.ModuleType("a2a.types")

    class AgentCard:  # minimal stub for imports
        pass

    types_module.AgentCard = AgentCard
    a2a_module.types = types_module
    sys.modules["a2a"] = a2a_module
    sys.modules["a2a.types"] = types_module

from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider


class DummyLLM:
    def __init__(self, model: str, provider: Provider = Provider.GOOGLE) -> None:
        self.model_name = model
        self.provider = provider
        self.default_request_params = type("Params", (), {"model": model})()


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
    info = ModelInfo.from_llm(DummyLLM("gemini25"))
    assert info is not None
    assert info.name == "gemini-2.5-flash-preview-09-2025"
    assert info.tdv_flags == (True, True, True)


def test_model_info_from_agent_llm_capabilities() -> None:
    agent = DummyAgent("gemini-2.5-pro", provider=Provider.GOOGLE)
    info = ModelInfo.from_llm(agent.llm)
    assert info is not None
    assert info.name == "gemini-2.5-pro"
    assert info.tdv_flags == (True, True, True)


def test_unknown_model_defaults_to_text_only() -> None:
    info = ModelInfo.from_name("unknown-model-id")
    assert info is not None
    assert info.tdv_flags == (True, False, False)
