from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fast_agent.llm.model_display_name import resolve_llm_display_name, resolve_model_display_name
from fast_agent.llm.model_factory import ModelConfig
from fast_agent.llm.model_overlays import LoadedModelOverlay, ModelOverlayManifest
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.resolved_model import ResolvedModelSpec


@dataclass(slots=True)
class _StubLLM:
    resolved_model: ResolvedModelSpec


def test_resolve_llm_display_name_uses_wire_model_name_for_builtin_presets() -> None:
    resolved_model = ResolvedModelSpec(
        raw_input="codexplan?reasoning=high",
        selected_model_name="codexplan?reasoning=high",
        source="preset",
        model_config=ModelConfig(
            provider=Provider.CODEX_RESPONSES,
            model_name="gpt-5.4",
        ),
        provider=Provider.CODEX_RESPONSES,
        wire_model_name="gpt-5.4",
    )

    assert resolved_model.selected_model_token == "codexplan"
    assert resolved_model.display_name == "gpt-5.4"
    assert resolve_llm_display_name(_StubLLM(resolved_model)) == "gpt-5.4"


def test_resolve_llm_display_name_uses_wire_model_name_for_anthropic_presets() -> None:
    resolved_model = ResolvedModelSpec(
        raw_input="sonnet",
        selected_model_name="sonnet",
        source="preset",
        model_config=ModelConfig(
            provider=Provider.ANTHROPIC,
            model_name="claude-sonnet-4-6",
        ),
        provider=Provider.ANTHROPIC,
        wire_model_name="claude-sonnet-4-6",
    )

    assert resolved_model.display_name == "claude-sonnet-4-6"
    assert resolve_llm_display_name(_StubLLM(resolved_model)) == "claude-sonnet-4-6"


def test_resolve_llm_display_name_marks_anthropic_vertex_route() -> None:
    resolved_model = ResolvedModelSpec(
        raw_input="anthropic-vertex.claude-sonnet-4-6",
        selected_model_name="anthropic-vertex.claude-sonnet-4-6",
        source="preset",
        model_config=ModelConfig(
            provider=Provider.ANTHROPIC_VERTEX,
            model_name="claude-sonnet-4-6",
        ),
        provider=Provider.ANTHROPIC_VERTEX,
        wire_model_name="claude-sonnet-4-6",
    )

    assert resolved_model.display_name == "claude-sonnet-4-6 · Vertex"
    assert resolve_llm_display_name(_StubLLM(resolved_model)) == "claude-sonnet-4-6 · Vertex"


def test_resolve_llm_display_name_uses_wire_model_name_for_provider_routed_presets() -> None:
    resolved_model = ResolvedModelSpec(
        raw_input="glm",
        selected_model_name="glm",
        source="preset",
        model_config=ModelConfig(
            provider=Provider.HUGGINGFACE,
            model_name="zai-org/GLM-5:novita",
        ),
        provider=Provider.HUGGINGFACE,
        wire_model_name="zai-org/GLM-5:novita",
    )

    assert resolved_model.display_name == "GLM-5"
    assert resolve_llm_display_name(_StubLLM(resolved_model)) == "GLM-5"


def test_resolve_llm_display_name_uses_wire_model_name_for_direct_models() -> None:
    resolved_model = ResolvedModelSpec(
        raw_input="hf.moonshotai/Kimi-K2-Instruct-0905:groq",
        selected_model_name="hf.moonshotai/Kimi-K2-Instruct-0905:groq",
        source="direct",
        model_config=ModelConfig(
            provider=Provider.HUGGINGFACE,
            model_name="moonshotai/Kimi-K2-Instruct-0905:groq",
        ),
        provider=Provider.HUGGINGFACE,
        wire_model_name="moonshotai/Kimi-K2-Instruct-0905:groq",
    )

    assert resolved_model.display_name == "Kimi-K2-Instruct-0905"
    assert resolve_llm_display_name(_StubLLM(resolved_model)) == "Kimi-K2-Instruct-0905"


def test_resolve_model_display_name_formats_raw_model_strings() -> None:
    assert (
        resolve_model_display_name("moonshotai/Kimi-K2-Instruct-0905:groq")
        == "Kimi-K2-Instruct-0905"
    )
    assert resolve_model_display_name("zai-org/GLM-5:novita") == "GLM-5"
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
