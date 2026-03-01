from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.config import Settings
from fast_agent.context import Context


def _ensure_hf_inference_acp_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_root = repo_root / "publish" / "hf-inference-acp" / "src"
    sys.path.insert(0, str(package_root))


@pytest.mark.asyncio
async def test_apply_model_does_not_override_request_params_model(monkeypatch) -> None:
    pytest.importorskip("ruamel.yaml")
    _ensure_hf_inference_acp_on_path()

    import hf_inference_acp.agents as agents_mod  # ty: ignore[unresolved-import]
    from hf_inference_acp.agents import HuggingFaceAgent  # ty: ignore[unresolved-import]

    calls: list[dict] = []

    def fake_get_model_factory(context, model=None, request_params=None, **kwargs):
        calls.append({"model": model, "request_params_model": getattr(request_params, "model", None)})

        def dummy_factory(*_args, **_kwargs):
            return None

        return dummy_factory

    monkeypatch.setattr(agents_mod, "get_model_factory", fake_get_model_factory)

    async def fake_attach_llm(self, llm_factory, model=None, request_params=None, **kwargs):
        return None

    monkeypatch.setattr(HuggingFaceAgent, "attach_llm", fake_attach_llm, raising=True)

    context = Context(config=Settings(mcp_ui_mode="disabled", default_model=None))

    agent = HuggingFaceAgent(config=AgentConfig(name="huggingface"), context=context)

    await agent.apply_model("hf.moonshotai/Kimi-K2-Instruct-0905:groq")

    assert agent.config.model == "hf.moonshotai/Kimi-K2-Instruct-0905:groq"
    assert agent.config.default_request_params is not None
    assert "model" not in agent.config.default_request_params.model_dump(exclude_unset=True)
    assert calls and calls[-1]["model"] == "hf.moonshotai/Kimi-K2-Instruct-0905:groq"


def test_alias_display_preserves_query_when_overriding_provider_suffix() -> None:
    pytest.importorskip("ruamel.yaml")
    _ensure_hf_inference_acp_on_path()

    from hf_inference_acp.agents import _resolve_alias_display  # ty: ignore[unresolved-import]

    resolved = _resolve_alias_display("qwen35:groq")
    assert resolved is not None
    _alias, target = resolved
    assert target.startswith("hf.Qwen/Qwen3.5-397B-A17B:groq?")
    assert "temperature=0.6" in target
    assert "top_k=20" in target
