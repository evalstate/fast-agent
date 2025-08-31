import pytest

from fast_agent.agents.workflow.orchestrator_models import (
    AgentTask,
    Plan,
    PlanningStep,
    Step,
)
from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.augmented_llm_passthrough import FIXED_RESPONSE_INDICATOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_plan_execution(fast_agent):
    """Test full plan execution mode of the orchestrator agent."""
    fast = fast_agent

    @fast.agent(name="agent1", model="passthrough")
    @fast.agent(name="agent2", model="passthrough")
    @fast.orchestrator(
        name="orchestrator", agents=["agent1", "agent2"], plan_type="full", model="passthrough"
    )
    async def agent_function():
        async with fast.run() as agent:
            # Create test plan
            test_plan = Plan(
                steps=[
                    Step(
                        description="First step",
                        tasks=[
                            AgentTask(description="Task for agent1", agent="agent1"),
                            AgentTask(description="Task for agent2", agent="agent2"),
                        ],
                    ),
                    Step(
                        description="Second step",
                        tasks=[AgentTask(description="Another task for agent1", agent="agent1")],
                    ),
                ],
                is_complete=True,
            )

            agent.orchestrator._get_full_plan

            async def mock_get_full_plan(*args, **kwargs):
                return test_plan

            agent.orchestrator._get_full_plan = mock_get_full_plan

            # Set up agent1 responses
            await agent.agent1._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} Agent1 Task 1 Response")]
            )

            await agent.agent1._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} Agent1 Task 2 Response")]
            )

            # Set up agent2 response
            await agent.agent2._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} Agent2 Task 1 Response")]
            )

            # Set up synthesis response
            await agent.orchestrator._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} Final synthesized result from all steps")]
            )

            # Execute orchestrator
            result = await agent.orchestrator.send("Accomplish this complex task")

            # Verify the result contains the synthesized output
            assert "Final synthesized result" in result

            # Check plan results
            plan_result = agent.orchestrator.plan_result
            assert plan_result is not None
            assert len(plan_result.step_results) == 2
            assert plan_result.is_complete

            # Check task results - The first step has 2 tasks
            first_step = plan_result.step_results[0]
            assert len(first_step.task_results) == 2

            # Check that both agents' tasks were executed - order not guaranteed
            has_agent1 = any(
                "Agent1" in task.result
                for task in first_step.task_results
                if task.agent == "agent1"
            )
            has_agent2 = any(
                "Agent2" in task.result
                for task in first_step.task_results
                if task.agent == "agent2"
            )
            assert has_agent1, "Agent1's task result not found in first step"
            assert has_agent2, "Agent2's task result not found in first step"

            # Check second step
            second_step = plan_result.step_results[1]
            assert len(second_step.task_results) == 1
            assert "Agent1 Task 2 Response" in second_step.task_results[0].result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_iterative_plan_execution(fast_agent):
    """Test iterative plan execution mode of the orchestrator agent."""
    fast = fast_agent

    @fast.agent(name="agent1", model="passthrough")
    @fast.agent(name="agent2", model="passthrough")
    @fast.orchestrator(
        name="orchestrator", agents=["agent1", "agent2"], plan_type="iterative", model="passthrough"
    )
    async def agent_function():
        async with fast.run() as agent:
            # Define first step
            step1 = PlanningStep(
                description="First iterative step",
                tasks=[AgentTask(description="Initial task for agent1", agent="agent1")],
                is_complete=False,
            )

            # Override _get_next_step to return our predefined steps
            agent.orchestrator._get_next_step

            async def mock_get_next_step(*args, **kwargs):
                return step1

            agent.orchestrator._get_next_step = mock_get_next_step

            # Set up agent1 responses for step 1
            await agent.agent1._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} Agent1 Step 1 Response")]
            )

            # When the orchestrator asks for second step, return step2
            # This is tricky with passthrough - we need to modify the LLM's _fixed_response
            # mid-test to return a different response for the second call

            # Execute orchestrator to get first step
            # We'll skip the full test for iterative because of the limitations of passthrough LLM
            await agent.orchestrator.send("Do this step by step")

            # Check that one step was executed
            plan_result = agent.orchestrator.plan_result
            assert plan_result is not None
            assert len(plan_result.step_results) >= 1

            # Verify the first step had the expected agent1 response
            assert "Agent1 Step 1 Response" in plan_result.step_results[0].task_results[0].result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_agent_handling(fast_agent):
    """Test how orchestrator handles plans with invalid agent references."""
    fast = fast_agent

    @fast.agent(name="valid_agent", model="passthrough")
    @fast.orchestrator(
        name="orchestrator", agents=["valid_agent"], plan_type="full", model="passthrough"
    )
    async def agent_function():
        async with fast.run() as agent:
            # Create a plan with one valid and one invalid agent
            test_plan = Plan(
                steps=[
                    Step(
                        description="Step with mixed agent validity",
                        tasks=[
                            AgentTask(description="Task for valid agent", agent="valid_agent"),
                            AgentTask(
                                description="Task for non-existent agent", agent="invalid_agent"
                            ),
                        ],
                    )
                ],
                is_complete=True,
            )

            # Override _get_full_plan to return our predefined plan
            agent.orchestrator._get_full_plan

            async def mock_get_full_plan(*args, **kwargs):
                return test_plan

            agent.orchestrator._get_full_plan = mock_get_full_plan

            # Set up valid_agent response
            await agent.valid_agent._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} Valid agent response")]
            )

            # Set up synthesis response
            await agent.orchestrator._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} Synthesis including error handling")]
            )

            # Execute orchestrator
            result = await agent.orchestrator.send("Test invalid agent reference")

            # Verify the result contains the synthesis
            assert "Synthesis including error handling" in result

            # Check plan results for error information
            plan_result = agent.orchestrator.plan_result
            assert plan_result is not None

            # Should have one step executed
            assert len(plan_result.step_results) == 1
            step_result = plan_result.step_results[0]

            # Should have two tasks (valid and invalid)
            assert len(step_result.task_results) == 2

            # Check for error message in the invalid agent task
            has_error = any(
                "invalid_agent" in task.agent and "ERROR" in task.result
                for task in step_result.task_results
            )
            assert has_error, "Expected error for invalid agent not found"

            # Check that valid agent task was executed
            has_valid = any(
                "valid_agent" in task.agent and "Valid agent response" in task.result
                for task in step_result.task_results
            )
            assert has_valid, "Valid agent should have executed successfully"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_max_iterations_handling(fast_agent):
    """Test how orchestrator handles reaching maximum iterations."""
    fast = fast_agent

    @fast.agent(name="agent1", model="passthrough")
    @fast.orchestrator(
        name="orchestrator", agents=["agent1"], plan_type="iterative", model="passthrough"
    )
    async def agent_function():
        async with fast.run() as agent:
            # Set max_iterations very low to ensure we hit the limit
            agent.orchestrator._default_request_params.max_iterations = 2

            # Create a step that is never complete
            not_complete_step = PlanningStep(
                description="Step that isn't complete",
                tasks=[
                    AgentTask(description="Task that doesn't complete objective", agent="agent1")
                ],
                is_complete=False,
            )

            # Override _get_next_step to return our non-complete step
            agent.orchestrator._get_next_step

            async def mock_get_next_step(*args, **kwargs):
                return not_complete_step

            agent.orchestrator._get_next_step = mock_get_next_step

            # Set up agent1 response
            await agent.agent1._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} Progress, but not complete")]
            )

            # Set up synthesis response indicating incomplete execution
            await agent.orchestrator._llm.generate(
                [
                    Prompt.user(
                        f"{FIXED_RESPONSE_INDICATOR} Incomplete result due to iteration limits"
                    )
                ]
            )

            # Execute orchestrator
            result = await agent.orchestrator.send("Task requiring many steps")

            # Verify the result mentions the incomplete execution
            assert "Incomplete result" in result

            # Check that the max_iterations_reached flag is set
            plan_result = agent.orchestrator.plan_result
            assert plan_result is not None
            assert plan_result.max_iterations_reached

            # Check that we have some step executions
            assert len(plan_result.step_results) > 0
            # Each step should have an agent1 task result
            for step in plan_result.step_results:
                assert len(step.task_results) == 1
                assert step.task_results[0].agent == "agent1"
                assert "Progress, but not complete" in step.task_results[0].result

    await agent_function()
