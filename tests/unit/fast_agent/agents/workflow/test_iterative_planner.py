from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.iterative_planner import IterativePlanner
from fast_agent.agents.workflow.orchestrator_models import (
    AgentTask,
    PlanningStep,
    PlanResult,
    Step,
    StepResult,
)
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from rich.text import Text


class _ScriptedIterativePlanner(IterativePlanner):
    def __init__(self, steps: list[PlanningStep], *, plan_iterations: int = -1) -> None:
        super().__init__(
            AgentConfig("planner"),
            [LlmAgent(AgentConfig("worker"))],
            plan_iterations=plan_iterations,
        )
        self._steps = steps
        self.final_prompts: list[str] = []

    async def _get_next_step(
        self,
        objective: str,
        plan_result: PlanResult,
        request_params: RequestParams | None,
    ) -> PlanningStep | None:
        if not self._steps:
            return None
        return self._steps.pop(0)

    async def _execute_step(
        self,
        step: Step,
        previous_result: PlanResult,
        request_params: RequestParams | None = None,
    ) -> StepResult:
        return StepResult(step=step, result="done")

    async def _planner_generate_str(
        self,
        message: str,
        request_params: RequestParams | None,
    ) -> PromptMessageExtended:
        self.final_prompts.append(message)
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="final answer")],
        )

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_items: str | list[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool | None = None,
        render_message: bool = True,
        show_reprint_banner: bool = False,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_execute_step_reports_unknown_worker_agent() -> None:
    planner = IterativePlanner(AgentConfig("planner"), [LlmAgent(AgentConfig("known"))])
    step = Step(
        description="delegate work",
        tasks=[AgentTask(description="do the missing work", agent="missing")],
    )
    previous_result = PlanResult(objective="finish the objective", step_results=[])

    result = await planner._execute_step(step, previous_result)

    assert len(result.task_results) == 1
    task_result = result.task_results[0]
    assert task_result.description == "do the missing work"
    assert task_result.agent == "missing"
    assert task_result.result == "ERROR: Agent 'missing' is not available"


@pytest.mark.asyncio
async def test_execute_plan_marks_result_complete_when_planner_finishes() -> None:
    planner = _ScriptedIterativePlanner(
        [
            PlanningStep(
                description="objective already satisfied",
                tasks=[],
                is_complete=True,
            )
        ]
    )

    result = await planner._execute_plan("finish the objective", request_params=None)

    assert result.is_complete is True
    assert result.max_iterations_reached is False
    assert result.result == "final answer"
    assert "<fastagent:status>Complete</fastagent:status>" in planner.final_prompts[0]


@pytest.mark.asyncio
async def test_execute_plan_marks_result_when_iteration_cap_is_reached() -> None:
    planner = _ScriptedIterativePlanner(
        [
            PlanningStep(
                description="one step",
                tasks=[],
                is_complete=False,
            )
        ],
        plan_iterations=1,
    )

    result = await planner._execute_plan("finish the objective", request_params=None)

    assert result.is_complete is False
    assert result.max_iterations_reached is True
    assert result.result == "final answer"
    assert "Reached maximum number of iterations (1)" in planner.final_prompts[0]
