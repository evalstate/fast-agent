"""HF Tencent Hy4 and Hy4-flash contracts exercised through resolution and request shaping."""

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams


def _llm(model: str, *, configured_backend: str | None = None) -> HuggingFaceLLM:
    llm = ModelFactory.create_factory(model)(
        agent=LlmAgent(AgentConfig(name="hy4-contract")),
        context=Context(
            config=Settings(hf=HuggingFaceSettings(default_provider=configured_backend))
        ),
    )
    assert isinstance(llm, HuggingFaceLLM)
    return llm


def _request(llm: HuggingFaceLLM, params: RequestParams | None = None) -> dict[str, object]:
    return llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        params or llm.default_request_params,
    )


@pytest.mark.parametrize(
    ("alias", "expected_wire", "expected_fast"),
    (
        ("hy4", "tencent/Hy4:tencent", False),
        ("hy4-flash", "tencent/Hy4-flash:tencent", True),
    ),
)
def test_hy4_aliases_and_catalog_metadata(
    alias: str, expected_wire: str, expected_fast: bool
) -> None:
    llm = _llm(alias)
    request = _request(llm)

    assert request["model"] == expected_wire
    assert "max_tokens" not in request
    assert request["reasoning_effort"] == "high"
    assert "extra_body" not in request

    assert llm.model_name is not None
    params = ModelDatabase.get_model_params(llm.model_name)
    assert params is not None
    assert params.ptc_enabled is True
    assert params.context_window == 1_048_576
    assert params.max_output_tokens == 65536
    assert params.json_mode == "schema"
    assert params.reasoning == "reasoning_content"
    assert params.fast is expected_fast
    assert "image/png" in params.tokenizes

    hf_entries = ModelSelectionCatalog.list_entries(provider=Provider.HUGGINGFACE)
    matched = [entry for entry in hf_entries if entry.alias == alias]
    assert len(matched) == 1
    assert matched[0].model == f"hf.{expected_wire}"
    assert matched[0].fast is expected_fast


@pytest.mark.parametrize("alias", ("hy4", "hy4-flash"))
@pytest.mark.parametrize(
    ("setting", "expected_effort"),
    (
        (None, "high"),
        ("high", "high"),
        ("on", "high"),
        ("true", "high"),
        ("none", "none"),
        ("off", "none"),
        ("false", "none"),
    ),
)
def test_hy4_reasoning_wire_contract(alias: str, setting: str | None, expected_effort: str) -> None:
    query = f"?reasoning={setting}" if setting else ""
    request = _request(_llm(f"{alias}{query}"))

    assert request["reasoning_effort"] == expected_effort
    assert "extra_body" not in request
    assert "thinking" not in request


def test_hy4_explicit_max_tokens_and_ptc_query_override() -> None:
    parsed = ModelFactory.parse_model_spec("hy4-flash?ptc=off")
    assert parsed.query_overrides.ptc is False

    llm = _llm("hy4-flash")
    explicit = _request(llm, RequestParams(max_tokens=4096))
    assert explicit["max_tokens"] == 4096


def test_hy4_preserves_raw_reasoning_for_replay() -> None:
    for alias in ("hy4", "hy4-flash"):
        llm = _llm(alias)
        assert llm.model_name is not None
        assert llm._preserves_raw_reasoning_history(llm.model_name) is True


@pytest.mark.parametrize(
    ("spec", "expected_wire"),
    (
        ("hf.tencent/hy4:tencent", "tencent/Hy4:tencent"),
        ("hf.tencent/hy4:together", "tencent/Hy4:together"),
        ("hf.tencent/hy4-flash:tencent", "tencent/Hy4-flash:tencent"),
    ),
)
def test_hy4_lowercase_ids_recover_canonical_wire_casing(spec: str, expected_wire: str) -> None:
    request = _request(_llm(spec))
    assert request["model"] == expected_wire
