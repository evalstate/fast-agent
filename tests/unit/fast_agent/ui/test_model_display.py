"""
Testing notes:

- This module owns user-visible display formatting: stripping provider prefixes,
  hiding routing suffixes, and honoring resolved-model display metadata such as
  overlay labels or Anthropic Vertex markers.
- Prefer end-to-end display smoke tests through real resolved models or pure
  formatting helpers; avoid restating catalog tables here unless the exact
  rendered label is the product contract.
- Alias/default selection logic belongs in test_model_factory.py and
  test_model_selection_catalog.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.llm.model_display_name import resolve_llm_display_name, resolve_model_display_name
from fast_agent.llm.model_factory import ModelConfig, ModelFactory
from fast_agent.llm.model_overlays import LoadedModelOverlay, ModelOverlayManifest
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.resolved_model import ResolvedModelSpec


@dataclass(slots=True)
class _StubLLM:
    resolved_model: ResolvedModelSpec


def _make_llm(model: str):
    return ModelFactory.create_factory(model)(LlmAgent(AgentConfig(name="display-test")))


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("codexplan?reasoning=high", "gpt-5.4"),
        ("sonnet", "claude-sonnet-4-6"),
        ("glm", "GLM-5.1"),
        ("hf.moonshotai/Kimi-K2-Instruct-0905:groq", "Kimi-K2-Instruct-0905"),
    ],
)
def test_resolve_llm_display_name_smoke(model: str, expected: str) -> None:
    llm = _make_llm(model)

    assert llm.resolved_model is not None
    assert llm.resolved_model.display_name == expected
    assert resolve_llm_display_name(llm) == expected


def test_resolve_model_display_name_formats_raw_model_strings() -> None:
    assert (
        resolve_model_display_name("moonshotai/Kimi-K2-Instruct-0905:groq")
        == "Kimi-K2-Instruct-0905"
    )
    assert resolve_model_display_name("zai-org/GLM-5.1:together") == "GLM-5.1"
    assert (
        resolve_model_display_name("anthropic-vertex.claude-sonnet-4-6")
        == "claude-sonnet-4-6 · Vertex"
    )


def test_resolve_llm_display_name_uses_overlay_name() -> None:
    overlay = LoadedModelOverlay(
        manifest=ModelOverlayManifest.model_validate(
            {
                "name": "haikutiny",
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "picker": {"label": "Haiku Tiny"},
            }
        ),
        manifest_path=Path("/tmp/haikutiny.yaml"),
    )
    resolved_model = ResolvedModelSpec(
        raw_input="haikutiny?reasoning=low",
        selected_model_name="haikutiny?reasoning=low",
        source="overlay",
        model_config=ModelConfig(
            provider=Provider.ANTHROPIC,
            model_name="claude-haiku-4-5",
        ),
        provider=Provider.ANTHROPIC,
        wire_model_name="claude-haiku-4-5",
        overlay=overlay,
    )

    assert resolved_model.display_name == "haikutiny"
    assert resolve_llm_display_name(_StubLLM(resolved_model)) == "haikutiny"
