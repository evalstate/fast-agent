"""Unit tests for the OrchestratorAgent class."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from mcp.types import TextContent

from fast_agent.agents import OrchestratorAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.workflow.orchestrator_models import (
    AgentTask,
    Plan,
    PlanningStep,
    PlanResult,
    Step,
)
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


@pytest_asyncio.fixture
async def orchestrator_fixture():
    """Fixture to create an OrchestratorAgent for testing."""
    # Create mock agents with proper _name attributes
    agent1 = MagicMock()
    agent1._name = "agent1"  # OrchestratorAgent expects _name, not name
    agent1.name = "agent1"  # Keep this for backward compatibility
    agent1.instruction = "Test agent 1 for testing purposes"
    agent1.generate = AsyncMock(
        return_value=PromptMessageExtended(
            role="assistant", content=[TextContent(type="text", text="Agent1 response")]
        )
    )
    # Mock the initialize method
    agent1.initialize = AsyncMock()
    agent1.initialized = False

    agent2 = MagicMock()
    agent2._name = "agent2"  # OrchestratorAgent expects _name, not name
    agent2.name = "agent2"  # Keep this for backward compatibility
    agent2.instruction = "Test agent 2 for testing purposes"
    agent2.generate = AsyncMock(
        return_value=PromptMessageExtended(
            role="assistant", content=[TextContent(type="text", text="Agent2 response")]
        )
    )
    # Mock the initialize method
    agent2.initialize = AsyncMock()
    agent2.initialized = False

    # Use real AgentConfig instead of mock
    config = AgentConfig(name="orchestrator", servers=[])

    # Create mock LLM for the orchestrator
    llm = MagicMock()
    llm.structured = AsyncMock()
    llm.generate = AsyncMock(
        return_value=PromptMessageExtended(
            role="assistant", content=[TextContent(type="text", text="LLM response")]
        )
    )

    # Create orchestrator
    orchestrator = OrchestratorAgent(config=config, agents=[agent1, agent2], plan_type="full")
    orchestrator._llm = llm

    return orchestrator, agent1, agent2, llm


@pytest.mark.asyncio
async def test_execute_step(orchestrator_fixture):
    """Test the _execute_step method."""
    orchestrator, agent1, agent2, _ = orchestrator_fixture

    # Create a test step
    step = Step(
        description="Test step",
        tasks=[
            AgentTask(description="Task for agent1", agent="agent1"),
            AgentTask(description="Task for agent2", agent="agent2"),
        ],
    )

    # Create an initial plan result for context
    plan_result = PlanResult(objective="Test objective", step_results=[])

    # Execute the step
    step_result = await orchestrator._execute_step(step, plan_result, RequestParams())

    # Check the result
    assert step_result is not None
    assert len(step_result.task_results) == 2
    assert step_result.task_results[0].agent == "agent1"
    assert step_result.task_results[1].agent == "agent2"
    assert "Agent1 response" in step_result.task_results[0].result
    assert "Agent2 response" in step_result.task_results[1].result

    # Verify agent call counts
    assert agent1.generate.call_count == 1
    assert agent2.generate.call_count == 1


@pytest.mark.asyncio
async def test_invalid_agent_handling(orchestrator_fixture):
    """Test handling of invalid agent references."""
    orchestrator, agent1, _, _ = orchestrator_fixture

    # Create a step with an invalid agent
    step = Step(
        description="Step with invalid agent",
        tasks=[
            AgentTask(description="Task for agent1", agent="agent1"),
            AgentTask(description="Task for invalid", agent="invalid_agent"),
        ],
    )

    # Create an initial plan result
    plan_result = PlanResult(objective="Test objective", step_results=[])

    # Execute the step
    step_result = await orchestrator._execute_step(step, plan_result, RequestParams())

    # Check results
    assert step_result is not None
    assert len(step_result.task_results) == 2

    # Check valid agent result
    valid_task = next(task for task in step_result.task_results if task.agent == "agent1")
    assert "Agent1 response" in valid_task.result

    # Check invalid agent result
    invalid_task = next(task for task in step_result.task_results if task.agent == "invalid_agent")
    assert "ERROR" in invalid_task.result

    # Verify only valid agent was called
    assert agent1.generate.call_count == 1


@pytest.mark.asyncio
async def test_plan_execution_flow(orchestrator_fixture):
    """Test the complete plan execution flow."""
    orchestrator, agent1, agent2, _ = orchestrator_fixture

    # Create a test plan
    test_plan = Plan(
        steps=[
            Step(
                description="First step",
                tasks=[AgentTask(description="Task for agent1", agent="agent1")],
            ),
            Step(
                description="Second step",
                tasks=[AgentTask(description="Task for agent2", agent="agent2")],
            ),
        ],
        is_complete=True,
    )

    # Mock the orchestrator's _get_full_plan method
    orchestrator._get_full_plan = AsyncMock(return_value=test_plan)

    # Set up synthesis response
    orchestrator._planner_generate_str = AsyncMock(return_value="Final synthesized result")

    # Execute the plan
    objective = "Test objective"
    result = await orchestrator._execute_plan(objective, RequestParams())

    # Check result
    assert result is not None
    assert result.objective == objective
    assert result.is_complete is True
    assert len(result.step_results) == 2
    assert result.result == "Final synthesized result"

    # Check first step result
    first_step = result.step_results[0]
    assert first_step.step.description == "First step"
    assert len(first_step.task_results) == 1
    assert first_step.task_results[0].agent == "agent1"
    assert "Agent1 response" in first_step.task_results[0].result

    # Check second step result
    second_step = result.step_results[1]
    assert second_step.step.description == "Second step"
    assert len(second_step.task_results) == 1
    assert second_step.task_results[0].agent == "agent2"
    assert "Agent2 response" in second_step.task_results[0].result

    # Verify agent call counts
    assert agent1.generate.call_count == 1
    assert agent2.generate.call_count == 1


@pytest.mark.asyncio
async def test_iterative_planning(orchestrator_fixture):
    """Test iterative planning mode."""
    orchestrator, agent1, agent2, _ = orchestrator_fixture

    # Create test steps
    step1 = PlanningStep(
        description="First iterative step",
        tasks=[AgentTask(description="Task for agent1", agent="agent1")],
        is_complete=False,
    )

    step2 = PlanningStep(
        description="Second iterative step",
        tasks=[AgentTask(description="Task for agent2", agent="agent2")],
        is_complete=True,
    )

    # Mock the orchestrator's _get_next_step method
    get_next_step_mock = AsyncMock()
    get_next_step_mock.side_effect = [step1, step2]
    orchestrator._get_next_step = get_next_step_mock

    # Set up plan type
    orchestrator.plan_type = "iterative"

    # Set up synthesis response
    orchestrator._planner_generate_str = AsyncMock(return_value="Iterative result")

    # Execute the plan
    objective = "Test iterative objective"
    result = await orchestrator._execute_plan(objective, RequestParams())

    # Check result
    assert result is not None
    assert result.objective == objective
    assert result.is_complete is True
    assert len(result.step_results) == 2
    assert result.result == "Iterative result"

    # Check agent calls
    assert agent1.generate.call_count == 1
    assert agent2.generate.call_count == 1

    # Check _get_next_step call count
    assert get_next_step_mock.call_count == 2


@pytest.mark.asyncio
async def test_orchestrator_with_empty_agents_raises_error():
    """Test that creating orchestrator with no agents raises error."""
    config = AgentConfig(name="test_orchestrator", servers=[])

    with pytest.raises(Exception):  # Should raise AgentConfigError
        OrchestratorAgent(config=config, agents=[], plan_type="full")


@pytest.mark.asyncio
async def test_execute_step_with_mixed_success_and_error(orchestrator_fixture):
    """Test step execution with both successful and failed tasks."""
    orchestrator, agent1, agent2, _ = orchestrator_fixture

    # Make agent2 raise an exception
    agent2.generate = AsyncMock(side_effect=Exception("Agent2 failed"))

    # Create a test step
    step = Step(
        description="Mixed success/failure step",
        tasks=[
            AgentTask(description="Task for agent1", agent="agent1"),
            AgentTask(description="Task for agent2", agent="agent2"),
        ],
    )

    # Create an initial plan result for context
    plan_result = PlanResult(objective="Test mixed results", step_results=[])

    # Execute the step
    step_result = await orchestrator._execute_step(step, plan_result, RequestParams())

    # Check the result
    assert step_result is not None
    assert len(step_result.task_results) == 2

    # Find the results for each agent
    agent1_result = next(task for task in step_result.task_results if task.agent == "agent1")
    agent2_result = next(task for task in step_result.task_results if task.agent == "agent2")

    # Agent1 should succeed
    assert "Agent1 response" in agent1_result.result

    # Agent2 should have error
    assert "ERROR" in agent2_result.result
    assert "Agent2 failed" in agent2_result.result


@pytest.mark.asyncio
async def test_format_agent_info(orchestrator_fixture):
    """Test the _format_agent_info method."""
    orchestrator, agent1, agent2, _ = orchestrator_fixture

    # Test formatting agent info
    agent1_info = orchestrator._format_agent_info("agent1")
    assert "agent1" in agent1_info
    assert "Test agent 1 for testing purposes" in agent1_info
    assert "<fastagent:agent" in agent1_info

    # Test with non-existent agent
    missing_agent_info = orchestrator._format_agent_info("nonexistent")
    assert missing_agent_info == ""


@pytest.mark.asyncio
async def test_validate_agent_names(orchestrator_fixture):
    """Test the _validate_agent_names method."""
    orchestrator, _, _, _ = orchestrator_fixture

    # Create a plan with valid agents
    valid_plan = Plan(
        steps=[
            Step(
                description="Valid step",
                tasks=[
                    AgentTask(description="Task for agent1", agent="agent1"),
                    AgentTask(description="Task for agent2", agent="agent2"),
                ],
            )
        ],
        is_complete=True,
    )

    # This should not raise any exceptions
    orchestrator._validate_agent_names(valid_plan)

    # Create a plan with invalid agents
    invalid_plan = Plan(
        steps=[
            Step(
                description="Invalid step",
                tasks=[
                    AgentTask(description="Task for agent1", agent="agent1"),
                    AgentTask(description="Task for invalid", agent="invalid_agent"),
                ],
            )
        ],
        is_complete=True,
    )

    # This should log errors but not raise exceptions
    orchestrator._validate_agent_names(invalid_plan)

    # Test with None plan
    orchestrator._validate_agent_names(None)


@pytest.mark.asyncio
async def test_merge_request_params(orchestrator_fixture):
    """Test the _merge_request_params method."""
    orchestrator, _, _, _ = orchestrator_fixture

    # Test with None params
    default_params = orchestrator._merge_request_params(None, None)
    assert default_params.use_history is False
    assert default_params.max_iterations == 5
    assert default_params.maxTokens == 8192

    # Test with custom params
    custom_params = RequestParams(maxTokens=4096, temperature=0.7)
    merged_params = orchestrator._merge_request_params(custom_params, None)
    assert merged_params.use_history is False  # Force setting
    assert merged_params.max_iterations == 5
    assert merged_params.maxTokens == 4096  # Custom override
    assert merged_params.temperature == 0.7  # Custom setting
