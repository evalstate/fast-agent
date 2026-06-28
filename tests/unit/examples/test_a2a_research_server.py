from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _research_server_module() -> ModuleType:
    path = Path(__file__).parents[3] / "examples" / "a2a" / "research" / "server.py"
    spec = importlib.util.spec_from_file_location("a2a_research_server_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_research_decision_round_trips_json() -> None:
    module = _research_server_module()
    decision = module.ResearchDecision(
        kind=module.ResearchDecisionKind.NEEDS_REFINEMENT,
        message="More detail needed",
    )

    parsed = module.ResearchDecision.from_json(decision.to_json())

    assert parsed == decision


@pytest.mark.asyncio
async def test_research_server_loads_agent_card_definitions_from_environment() -> None:
    module = _research_server_module()

    assert module.QUICK_REFINER_AGENT not in module.fast.agents
    assert module.RESEARCH_WORKER_AGENT not in module.fast.agents

    async with module.fast.harness():
        assert module.QUICK_REFINER_AGENT in module.fast.agents
        assert module.RESEARCH_WORKER_AGENT in module.fast.agents


def test_research_refinement_requests_missing_fields_with_context() -> None:
    module = _research_server_module()

    decision = module._decision_for_prompt("research recent A2A work")

    assert decision.kind == module.ResearchDecisionKind.NEEDS_REFINEMENT
    assert "'research recent A2A work'" in decision.message
    assert "the intended audience" in decision.message
    assert "the desired output format" in decision.message
    assert "Goal:" not in decision.message


def test_research_refinement_accepts_natural_language_actionable_task() -> None:
    module = _research_server_module()
    prompt = "goal is to research new models, html report, audience scientists"

    decision = module._decision_for_prompt(prompt)

    assert decision.kind == module.ResearchDecisionKind.BEGIN_RESEARCH
    assert decision.message == "Research task accepted"
    assert decision.goal == prompt


def test_research_refinement_also_accepts_labeled_actionable_task() -> None:
    module = _research_server_module()
    prompt = "Goal: compare A2A. Audience: developers. Output: markdown report."

    decision = module._decision_for_prompt(prompt)

    assert decision.kind == module.ResearchDecisionKind.BEGIN_RESEARCH
    assert decision.message == "Research task accepted"
    assert decision.goal == prompt
