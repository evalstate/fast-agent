from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import BaseModel

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.evaluator_optimizer import (
    EvaluationResult,
    EvaluatorOptimizerAgent,
    QualityRating,
)
from fast_agent.agents.workflow.iterative_planner import IterativePlanner
from fast_agent.agents.workflow.maker_agent import MakerAgent, MatchStrategy
from fast_agent.agents.workflow.orchestrator_models import AgentTask, PlanResult, Step
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.agents.workflow.request_params import child_request_params
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp import Tool
    from mcp.types import PromptMessage


class StructuredResult(BaseModel):
    value: str


class RecordingAgent(LlmAgent):
    def __init__(
        self,
        name: str,
        responses: list[str] | None = None,
        evaluations: list[EvaluationResult] | None = None,
    ) -> None:
        super().__init__(AgentConfig(name))
        self.responses = responses or []
        self.evaluations = evaluations or []
        self.generate_inputs: list[list[PromptMessageExtended]] = []
        self.generate_params: list[RequestParams | None] = []
        self.structured_inputs: list[list[PromptMessageExtended]] = []
        self.structured_params: list[RequestParams | None] = []
        self.structured_schema_inputs: list[list[PromptMessageExtended]] = []
        self.structured_schema_params: list[RequestParams | None] = []
        self.structured_schemas: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list["Tool"] | None = None,
    ) -> PromptMessageExtended:
        del tools
        assert isinstance(messages, list)
        self.generate_inputs.append(cast("list[PromptMessageExtended]", messages))
        self.generate_params.append(request_params)
        text = self.responses.pop(0) if self.responses else self.name
        return PromptMessageExtended(role="assistant", content=[text_content(text)])

    async def structured(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        model: type[StructuredResult],
        request_params: RequestParams | None = None,
    ) -> tuple[BaseModel | None, PromptMessageExtended]:
        assert isinstance(messages, list)
        self.structured_inputs.append(cast("list[PromptMessageExtended]", messages))
        self.structured_params.append(request_params)
        message = PromptMessageExtended(role="assistant", content=[text_content("structured")])
        if self.evaluations:
            return self.evaluations.pop(0), message
        return model(value=self.name), message

    async def structured_schema(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> tuple[dict[str, str], PromptMessageExtended]:
        assert isinstance(messages, list)
        self.structured_schema_inputs.append(cast("list[PromptMessageExtended]", messages))
        self.structured_schema_params.append(request_params)
        self.structured_schemas.append(schema)
        message = PromptMessageExtended(role="assistant", content=[text_content("structured")])
        return {"value": self.name}, message


def _parent_params() -> RequestParams:
    return RequestParams(
        model="parent-model",
        systemPrompt="parent instructions",
        maxTokens=17,
        use_history=False,
    )


def _assert_child_params(params: RequestParams | None) -> None:
    assert params is not None
    assert params.model is None
    assert params.systemPrompt is None
    assert params.maxTokens == 2048
    assert params.use_history is False


def test_child_request_params_only_forward_explicit_workflow_controls() -> None:
    parent_defaults = RequestParams.model_construct(
        _fields_set=set(),
        tool_result_mode="passthrough",
        emit_loop_progress=True,
        mcp_metadata={"trace": "parent-default"},
    )

    assert parent_defaults.model_fields_set == set()
    assert child_request_params(parent_defaults) is None

    explicit = RequestParams(
        tool_result_mode="passthrough",
        emit_loop_progress=True,
        mcp_metadata={"trace": "per-call"},
    )

    delegated = child_request_params(explicit)

    assert delegated is not None
    assert delegated.tool_result_mode == "passthrough"
    assert delegated.emit_loop_progress is True
    assert delegated.mcp_metadata == {"trace": "per-call"}

    explicit_no_progress = RequestParams(emit_loop_progress=False)
    delegated_no_progress = child_request_params(explicit_no_progress)

    assert delegated_no_progress is not None
    assert delegated_no_progress.emit_loop_progress is False

    explicit_postprocess = RequestParams(tool_result_mode="postprocess")
    delegated_postprocess = child_request_params(explicit_postprocess)

    assert delegated_postprocess is not None
    assert delegated_postprocess.tool_result_mode == "postprocess"


def test_maker_normalized_match_strategy_casefolds_and_collapses_whitespace() -> None:
    maker = MakerAgent(
        AgentConfig("maker"),
        worker_agent=RecordingAgent("worker"),
        match_strategy=MatchStrategy.NORMALIZED,
    )

    assert maker._normalize_response("  YES\t\n") == "yes"


@pytest.mark.asyncio
async def test_evaluator_optimizer_evaluates_and_returns_final_refinement() -> None:
    generator = RecordingAgent("generator", responses=["initial draft", "refined answer"])
    evaluator = RecordingAgent(
        "evaluator",
        evaluations=[
            EvaluationResult(
                rating=QualityRating.FAIR,
                feedback="Add the missing detail.",
                needs_improvement=True,
                focus_areas=["detail"],
            ),
            EvaluationResult(
                rating=QualityRating.GOOD,
                feedback="Good now.",
                needs_improvement=False,
            ),
        ],
    )
    optimizer = EvaluatorOptimizerAgent(
        AgentConfig("optimizer"),
        generator_agent=generator,
        evaluator_agent=evaluator,
        max_refinements=1,
    )

    result = await optimizer.generate("Write the answer", _parent_params())

    assert result.all_text() == "refined answer"
    assert len(evaluator.structured_inputs) == 2
    refinement_prompt = generator.generate_inputs[1][0].all_text()
    assert "Write the answer" in refinement_prompt
    assert "initial draft" in refinement_prompt
    assert "Add the missing detail." in refinement_prompt
    assert "detail" in refinement_prompt
    for params in generator.generate_params + evaluator.structured_params:
        _assert_child_params(params)


@pytest.mark.asyncio
async def test_evaluator_optimizer_structured_reparses_generated_text() -> None:
    generator = RecordingAgent("generator", responses=["final prose"])
    evaluator = RecordingAgent(
        "evaluator",
        evaluations=[
            EvaluationResult(
                rating=QualityRating.GOOD,
                feedback="Good.",
                needs_improvement=False,
            ),
        ],
    )
    optimizer = EvaluatorOptimizerAgent(
        AgentConfig("optimizer"),
        generator_agent=generator,
        evaluator_agent=evaluator,
    )

    result, message = await optimizer.structured(
        "Write the answer",
        StructuredResult,
        _parent_params(),
    )

    assert result == StructuredResult(value="generator")
    assert message.all_text() == "structured"
    assert (
        "Convert this evaluator-optimizer response" in generator.structured_inputs[0][0].all_text()
    )
    assert generator.structured_inputs[0][0].all_text().endswith("final prose")
    _assert_child_params(generator.structured_params[0])


@pytest.mark.asyncio
async def test_parallel_agent_uses_child_request_params_for_fan_out_and_fan_in() -> None:
    first = RecordingAgent("first", responses=["one"])
    second = RecordingAgent("second", responses=["two"])
    fan_in = RecordingAgent("fan_in", responses=["combined"])
    parallel = ParallelAgent(AgentConfig("parallel"), fan_in, [first, second])

    result = await parallel.generate("combine", _parent_params())

    assert result.all_text() == "combined"
    fan_in_prompt = fan_in.generate_inputs[0][0].all_text()
    assert "<fastagent:request>\ncombine\n</fastagent:request>" in fan_in_prompt
    assert '<fastagent:response agent="first">\none\n</fastagent:response>' in fan_in_prompt
    assert '<fastagent:response agent="second">\ntwo\n</fastagent:response>' in fan_in_prompt
    for params in first.generate_params + second.generate_params + fan_in.generate_params:
        _assert_child_params(params)


def test_parallel_agent_rejects_empty_fan_out_agents() -> None:
    fan_in = RecordingAgent("fan_in", responses=["combined"])

    with pytest.raises(AgentConfigError, match="requires at least one fan-out agent"):
        ParallelAgent(AgentConfig("parallel"), fan_in, [])


def test_parallel_agent_response_formatting_requires_matching_fan_out_results() -> None:
    first = RecordingAgent("first", responses=["one"])
    second = RecordingAgent("second", responses=["two"])
    fan_in = RecordingAgent("fan_in", responses=["combined"])
    parallel = ParallelAgent(AgentConfig("parallel"), fan_in, [first, second])

    with pytest.raises(ValueError, match="zip\\(\\) argument 2 is shorter"):
        parallel._format_responses(["one"])


@pytest.mark.asyncio
async def test_parallel_agent_structured_schema_uses_fan_in_agent() -> None:
    first = RecordingAgent("first", responses=["one"])
    second = RecordingAgent("second", responses=["two"])
    fan_in = RecordingAgent("fan_in", responses=["combined"])
    parallel = ParallelAgent(AgentConfig("parallel"), fan_in, [first, second])
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    result, message = await parallel.structured_schema("combine", schema, _parent_params())

    assert result == {"value": "fan_in"}
    assert message.all_text() == "structured"
    assert len(first.generate_params) == 1
    assert len(second.generate_params) == 1
    assert fan_in.structured_inputs == []
    assert len(fan_in.structured_schema_inputs) == 1
    assert fan_in.structured_schemas == [schema]
    fan_in_prompt = fan_in.structured_schema_inputs[0][0].all_text()
    assert "<fastagent:request>\ncombine\n</fastagent:request>" in fan_in_prompt
    assert '<fastagent:response agent="first">\none\n</fastagent:response>' in fan_in_prompt
    assert '<fastagent:response agent="second">\ntwo\n</fastagent:response>' in fan_in_prompt
    for params in first.generate_params + second.generate_params + fan_in.structured_schema_params:
        _assert_child_params(params)


@pytest.mark.asyncio
async def test_iterative_planner_uses_child_request_params_for_worker_tasks() -> None:
    worker = RecordingAgent("worker", responses=["done"])
    planner = IterativePlanner(AgentConfig("planner"), [worker])
    step = Step(
        description="delegate work",
        tasks=[AgentTask(description="do the work", agent="worker")],
    )
    previous_result = PlanResult(objective="finish the objective", step_results=[])

    result = await planner._execute_step(step, previous_result, _parent_params())

    assert result.task_results[0].result == "done"
    _assert_child_params(worker.generate_params[0])


@pytest.mark.asyncio
async def test_maker_samples_disable_history_unless_caller_explicitly_sets_it() -> None:
    worker = RecordingAgent("worker", responses=["same", "same"])
    maker = MakerAgent(AgentConfig("maker"), worker_agent=worker, k=1, max_samples=2)

    result = await maker.generate("vote")

    assert result.all_text() == "same"
    assert worker.generate_params == [RequestParams(use_history=False)]


@pytest.mark.asyncio
async def test_maker_samples_preserve_explicit_history_choice() -> None:
    worker = RecordingAgent("worker", responses=["same", "same"])
    maker = MakerAgent(AgentConfig("maker"), worker_agent=worker, k=1, max_samples=2)

    result = await maker.generate("vote", RequestParams(use_history=True))

    assert result.all_text() == "same"
    assert worker.generate_params == [RequestParams(use_history=True)]


@pytest.mark.asyncio
async def test_maker_structured_parses_voted_response_without_second_worker_call() -> None:
    worker = RecordingAgent("worker", responses=['{"value": "voted"}'])
    maker = MakerAgent(AgentConfig("maker"), worker_agent=worker, k=1, max_samples=2)

    result, message = await maker.structured("vote", StructuredResult)

    assert result == StructuredResult(value="voted")
    assert message.all_text() == '{"value": "voted"}'
    assert len(worker.generate_inputs) == 1
    assert worker.structured_inputs == []


@pytest.mark.asyncio
async def test_maker_structured_delegates_voted_text_to_worker_when_not_json() -> None:
    worker = RecordingAgent("worker", responses=["plain voted answer"])
    maker = MakerAgent(AgentConfig("maker"), worker_agent=worker, k=1, max_samples=2)

    result, message = await maker.structured("vote", StructuredResult)

    assert result == StructuredResult(value="worker")
    assert message.all_text() == "structured"
    assert len(worker.generate_inputs) == 1
    assert len(worker.structured_inputs) == 1
    assert worker.structured_inputs[0][-1].all_text().endswith("plain voted answer")
    assert worker.structured_params == [RequestParams(use_history=False)]


@pytest.mark.asyncio
async def test_maker_structured_delegates_voted_json_with_trailing_text() -> None:
    worker = RecordingAgent("worker", responses=['{"value": "voted"} trailing'])
    maker = MakerAgent(AgentConfig("maker"), worker_agent=worker, k=1, max_samples=2)

    result, message = await maker.structured("vote", StructuredResult)

    assert result == StructuredResult(value="worker")
    assert message.all_text() == "structured"
    assert len(worker.generate_inputs) == 1
    assert len(worker.structured_inputs) == 1
    assert worker.structured_inputs[0][-1].all_text().endswith('{"value": "voted"} trailing')
    assert worker.structured_params == [RequestParams(use_history=False)]


@pytest.mark.asyncio
async def test_maker_structured_delegates_voted_json_when_schema_mismatches() -> None:
    worker = RecordingAgent("worker", responses=['{"other": "voted"}'])
    maker = MakerAgent(AgentConfig("maker"), worker_agent=worker, k=1, max_samples=2)

    result, message = await maker.structured("vote", StructuredResult)

    assert result == StructuredResult(value="worker")
    assert message.all_text() == "structured"
    assert len(worker.generate_inputs) == 1
    assert len(worker.structured_inputs) == 1
    assert worker.structured_inputs[0][-1].all_text().endswith('{"other": "voted"}')
    assert worker.structured_params == [RequestParams(use_history=False)]


@pytest.mark.asyncio
async def test_maker_structured_schema_parses_voted_response_without_second_worker_call() -> None:
    worker = RecordingAgent("worker", responses=['{"value": "voted"}'])
    maker = MakerAgent(AgentConfig("maker"), worker_agent=worker, k=1, max_samples=2)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    result, message = await maker.structured_schema("vote", schema)

    assert result == {"value": "voted"}
    assert message.all_text() == '{"value": "voted"}'
    assert len(worker.generate_inputs) == 1
    assert worker.structured_schema_inputs == []


@pytest.mark.asyncio
async def test_maker_structured_schema_delegates_voted_text_to_worker() -> None:
    worker = RecordingAgent("worker", responses=["plain voted answer"])
    maker = MakerAgent(AgentConfig("maker"), worker_agent=worker, k=1, max_samples=2)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    result, message = await maker.structured_schema("vote", schema)

    assert result == {"value": "worker"}
    assert message.all_text() == "structured"
    assert len(worker.generate_inputs) == 1
    assert len(worker.structured_schema_inputs) == 1
    assert worker.structured_schema_inputs[0][-1].all_text().endswith("plain voted answer")
    assert worker.structured_schema_params == [RequestParams(use_history=False)]
