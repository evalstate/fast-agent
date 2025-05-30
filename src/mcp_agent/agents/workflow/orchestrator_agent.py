"""
OrchestratorAgent implementation using the BaseAgent adapter pattern.

This workflow provides an implementation that manages complex tasks by
dynamically planning, delegating to specialized agents, and synthesizing results.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from mcp.types import TextContent

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.agents.workflow.orchestrator_models import (
    NextStep,
    Plan,
    PlanResult,
    Step,
    TaskWithResult,
    format_plan_result,
    format_step_result_text,
)
from mcp_agent.agents.workflow.orchestrator_prompts import (
    FULL_PLAN_PROMPT_TEMPLATE,
    ITERATIVE_PLAN_PROMPT_TEMPLATE,
    SYNTHESIZE_INCOMPLETE_PLAN_TEMPLATE,
    SYNTHESIZE_PLAN_PROMPT_TEMPLATE,
    TASK_PROMPT_TEMPLATE,
)
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    An agent that implements the orchestrator workflow pattern.

    Dynamically creates execution plans and delegates tasks
    to specialized worker agents, synthesizing their results into a cohesive output.
    Supports both full planning and iterative planning modes.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.ORCHESTRATOR

    def __init__(
        self,
        config: AgentConfig,
        agents: List[Agent],
        plan_type: Literal["full", "iterative"] = "full",
        plan_iterations: int = 5,
        context: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an OrchestratorAgent.

        Args:
            config: Agent configuration or name
            agents: List of specialized worker agents available for task execution
            plan_type: Planning mode ("full" or "iterative")
            context: Optional context object
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, context=context, **kwargs)

        if not agents:
            raise AgentConfigError("At least one worker agent must be provided")

        self.plan_type = plan_type

        # Store agents by name for easier lookup
        self.agents: Dict[str, Agent] = {}
        for agent in agents:
            agent_name = agent.name
            self.logger.info(f"Adding agent '{agent_name}' to orchestrator")
            self.agents[agent_name] = agent
        self.plan_iterations = plan_iterations
        # For tracking state during execution
        self.plan_result: Optional[PlanResult] = None

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Execute an orchestrated plan to process the input.

        Args:
            multipart_messages: Messages to process
            request_params: Optional request parameters

        Returns:
            The final synthesized response from the orchestration
        """
        # Extract user request
        objective = multipart_messages[-1].all_text() if multipart_messages else ""

        # Initialize execution parameters
        params = self._merge_request_params(request_params)

        # Execute the plan
        plan_result = await self._execute_plan(objective, params)
        self.plan_result = plan_result

        # Return the result
        return PromptMessageMultipart(
            role="assistant",
            content=[TextContent(type="text", text=plan_result.result or "No result available")],
        )

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Execute an orchestration plan and parse the result into a structured format.

        Args:
            prompt: List of messages to process
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed final response, or None if parsing fails
        """
        # Generate orchestration result
        response = await self.generate(prompt, request_params)

        # Try to parse the response into the specified model
        try:
            result_text = response.all_text()
            prompt_message = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=result_text)]
            )
            assert self._llm
            return await self._llm.structured([prompt_message], model, request_params)
        except Exception as e:
            self.logger.warning(f"Failed to parse orchestration result: {str(e)}")
            return None, Prompt.assistant(f"Failed to parse orchestration result: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the orchestrator agent and worker agents."""
        await super().initialize()

        # Initialize all worker agents if not already initialized
        for agent_name, agent in self.agents.items():
            if not getattr(agent, "initialized", False):
                self.logger.debug(f"Initializing agent: {agent_name}")
                await agent.initialize()

        self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown the orchestrator agent and worker agents."""
        await super().shutdown()

        # Shutdown all worker agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down agent {agent_name}: {str(e)}")

    async def _execute_plan(self, objective: str, request_params: RequestParams) -> PlanResult:
        """
        Execute a plan to achieve the given objective.

        Args:
            objective: The objective to achieve
            request_params: Request parameters for execution

        Returns:
            PlanResult containing execution results and final output
        """
        iterations = 0
        total_steps_executed = 0
        max_iterations = self.plan_iterations
        max_steps = getattr(request_params, "max_steps", max_iterations * 3)

        # Initialize plan result
        plan_result = PlanResult(objective=objective, step_results=[])
        plan_result.max_iterations_reached = False

        while iterations < max_iterations:
            # Generate plan based on planning mode
            if self.plan_type == "iterative":
                next_step = await self._get_next_step(objective, plan_result, request_params)
                if next_step is None:
                    self.logger.error("Failed to generate next step, ending iteration early")
                    plan_result.max_iterations_reached = True
                    break

                logger.debug(f"Iteration {iterations}: Iterative plan:", data=next_step)
                plan = Plan(steps=[next_step], is_complete=next_step.is_complete)
            elif self.plan_type == "full":
                plan = await self._get_full_plan(objective, plan_result, request_params)
                if plan is None:
                    self.logger.error("Failed to generate full plan, ending iteration early")
                    plan_result.max_iterations_reached = True
                    break

                logger.debug(f"Iteration {iterations}: Full Plan:", data=plan)
            else:
                raise ValueError(f"Invalid plan type: {self.plan_type}")

            # Validate agent names early
            self._validate_agent_names(plan)

            # Store plan in result
            plan_result.plan = plan

            # Execute the steps in the plan
            for step in plan.steps:
                # Check if we've hit the step limit
                if total_steps_executed >= max_steps:
                    self.logger.warning(
                        f"Reached maximum step limit ({max_steps}) without completing objective"
                    )
                    plan_result.max_iterations_reached = True
                    break

                # Execute the step and collect results
                step_result = await self._execute_step(step, plan_result, request_params)

                plan_result.add_step_result(step_result)
                total_steps_executed += 1

            # Check if we need to break due to hitting max steps
            if getattr(plan_result, "max_iterations_reached", False):
                break

            # If the plan is marked complete, finalize the result
            if plan.is_complete:
                plan_result.is_complete = True
                break

            # Increment iteration counter
            iterations += 1

        # Generate final result based on execution status
        if iterations >= max_iterations and not plan_result.is_complete:
            self.logger.warning(f"Failed to complete in {max_iterations} iterations")
            plan_result.max_iterations_reached = True

            # Use incomplete plan template
            synthesis_prompt = SYNTHESIZE_INCOMPLETE_PLAN_TEMPLATE.format(
                plan_result=format_plan_result(plan_result), max_iterations=max_iterations
            )
        else:
            # Either plan is complete or we had other limits
            if not plan_result.is_complete:
                plan_result.is_complete = True

            # Use standard template
            synthesis_prompt = SYNTHESIZE_PLAN_PROMPT_TEMPLATE.format(
                plan_result=format_plan_result(plan_result)
            )

        # Generate final synthesis
        plan_result.result = await self._planner_generate_str(
            synthesis_prompt, request_params.model_copy(update={"max_iterations": 1})
        )

        return plan_result

    async def _execute_step(
        self, step: Step, previous_result: PlanResult, request_params: RequestParams
    ) -> Any:
        """
        Execute a single step from the plan.

        Args:
            step: The step to execute
            previous_result: Results of the plan execution so far
            request_params: Request parameters

        Returns:
            Result of executing the step
        """
        from mcp_agent.agents.workflow.orchestrator_models import StepResult

        # Initialize step result
        step_result = StepResult(step=step, task_results=[])

        # Format context for tasks
        context = format_plan_result(previous_result)

        # Execute all tasks in parallel
        futures = []
        error_tasks = []

        for task in step.tasks:
            # Check agent exists
            agent = self.agents.get(task.agent)
            if not agent:
                self.logger.error(
                    f"No agent found matching '{task.agent}'. Available agents: {list(self.agents.keys())}"
                )
                error_tasks.append(
                    (
                        task,
                        f"Error: Agent '{task.agent}' not found. Available agents: {', '.join(self.agents.keys())}",
                    )
                )
                continue

            # Prepare task prompt
            task_description = TASK_PROMPT_TEMPLATE.format(
                objective=previous_result.objective, task=task.description, context=context
            )

            # Queue task for execution
            futures.append(
                (
                    task,
                    agent.generate(
                        [
                            PromptMessageMultipart(
                                role="user",
                                content=[TextContent(type="text", text=task_description)],
                            )
                        ]
                    ),
                )
            )

        # Wait for all tasks
        task_results = []
        for future in futures:
            task, future_obj = future
            try:
                result = await future_obj
                result_text = result.all_text()

                # Create task result
                task_model = task.model_dump()
                task_result = TaskWithResult(
                    description=task_model["description"],
                    agent=task_model["agent"],
                    result=result_text,
                )
                task_results.append(task_result)
            except Exception as e:
                self.logger.error(f"Error executing task: {str(e)}")
                # Add error result
                task_model = task.model_dump()
                task_results.append(
                    TaskWithResult(
                        description=task_model["description"],
                        agent=task_model["agent"],
                        result=f"ERROR: {str(e)}",
                    )
                )

        # Add all task results to step result
        for task_result in task_results:
            step_result.add_task_result(task_result)

        # Add error task results
        for task, error_message in error_tasks:
            task_model = task.model_dump()
            step_result.add_task_result(
                TaskWithResult(
                    description=task_model["description"],
                    agent=task_model["agent"],
                    result=f"ERROR: {error_message}",
                )
            )

        # Format step result
        step_result.result = format_step_result_text(step_result)
        return step_result

    async def _get_full_plan(
        self, objective: str, plan_result: PlanResult, request_params: RequestParams
    ) -> Optional[Plan]:
        """
        Generate a full plan with all steps.

        Args:
            objective: The objective to achieve
            plan_result: Current plan execution state
            request_params: Request parameters

        Returns:
            Complete Plan with all steps, or None if parsing fails
        """
        # Format agent information for the prompt
        agent_formats = []
        for agent_name in self.agents.keys():
            formatted = self._format_agent_info(agent_name)
            agent_formats.append(formatted)

        agents = "\n".join(agent_formats)

        # Determine plan status
        if plan_result.is_complete:
            plan_status = "Plan Status: Complete"
        elif plan_result.step_results:
            plan_status = "Plan Status: In Progress"
        else:
            plan_status = "Plan Status: Not Started"

        # Calculate iteration information
        max_iterations = self.plan_iterations
        current_iteration = len(plan_result.step_results)
        current_iteration = min(current_iteration, max_iterations - 1)
        iterations_remaining = max(0, max_iterations - current_iteration - 1)
        iterations_info = f"Planning Budget: Iteration {current_iteration + 1} of {max_iterations} (with {iterations_remaining} remaining)"

        # Format the planning prompt
        prompt = FULL_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            plan_status=plan_status,
            iterations_info=iterations_info,
            agents=agents,
        )

        # Get structured response from LLM
        try:
            plan_msg = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=prompt)]
            )
            plan, _ = await self._llm.structured([plan_msg], Plan, request_params)
            return plan
        except Exception as e:
            self.logger.error(f"Failed to parse plan: {str(e)}")
            return None

    async def _get_next_step(
        self, objective: str, plan_result: PlanResult, request_params: RequestParams
    ) -> Optional[NextStep]:
        """
        Generate just the next step for iterative planning.

        Args:
            objective: The objective to achieve
            plan_result: Current plan execution state
            request_params: Request parameters

        Returns:
            Next step to execute, or None if parsing fails
        """
        # Format agent information
        agents = "\n".join(
            [self._format_agent_info(agent_name) for agent_name in self.agents.keys()]
        )

        # Determine plan status
        if plan_result.is_complete:
            plan_status = "Plan Status: Complete"
        elif plan_result.step_results:
            plan_status = "Plan Status: In Progress"
        else:
            plan_status = "Plan Status: Not Started"

        # Calculate iteration information
        max_iterations = request_params.max_iterations
        current_iteration = len(plan_result.step_results)
        iterations_remaining = max_iterations - current_iteration
        iterations_info = (
            f"Planning Budget: {iterations_remaining} of {max_iterations} iterations remaining"
        )

        # Format the planning prompt
        prompt = ITERATIVE_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            plan_status=plan_status,
            iterations_info=iterations_info,
            agents=agents,
        )

        # Get structured response from LLM
        try:
            plan_msg = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=prompt)]
            )
            next_step, _ = await self._llm.structured([plan_msg], NextStep, request_params)
            return next_step
        except Exception as e:
            self.logger.error(f"Failed to parse next step: {str(e)}")
            return None

    def _validate_agent_names(self, plan: Plan) -> None:
        """
        Validate all agent names in a plan before execution.

        Args:
            plan: The plan to validate
        """
        if plan is None:
            self.logger.error("Cannot validate agent names: plan is None")
            return

        invalid_agents = []

        for step in plan.steps:
            for task in step.tasks:
                if task.agent not in self.agents:
                    invalid_agents.append(task.agent)

        if invalid_agents:
            available_agents = ", ".join(self.agents.keys())
            invalid_list = ", ".join(invalid_agents)
            self.logger.error(
                f"Plan contains invalid agent names: {invalid_list}. Available agents: {available_agents}"
            )

    def _format_agent_info(self, agent_name: str) -> str:
        """
        Format agent information for display in prompts.

        Args:
            agent_name: Name of the agent to format

        Returns:
            Formatted agent information string
        """
        agent = self.agents.get(agent_name)
        if not agent:
            self.logger.error(f"Agent '{agent_name}' not found in orchestrator agents")
            return ""

        # Get agent instruction or default description
        instruction = agent.instruction if agent.instruction else f"Agent '{agent_name}'"

        # Format with XML tags
        return f'<fastagent:agent name="{agent_name}">{instruction}</fastagent:agent>'

    async def _planner_generate_str(self, message: str, request_params: RequestParams) -> str:
        """
        Generate string response from the orchestrator's own LLM.

        Args:
            message: Message to send to the LLM
            request_params: Request parameters

        Returns:
            String response from the LLM
        """
        # Create prompt message
        prompt = PromptMessageMultipart(
            role="user", content=[TextContent(type="text", text=message)]
        )

        # Get response from LLM
        response = await self._llm.generate([prompt], request_params)
        return response.all_text()

    def _merge_request_params(self, request_params: Optional[RequestParams]) -> RequestParams:
        """
        Merge provided request parameters with defaults.

        Args:
            request_params: Optional request parameters to merge

        Returns:
            Merged request parameters
        """
        # Create orchestrator-specific defaults
        defaults = RequestParams(
            use_history=False,  # Orchestrator doesn't use history
            max_iterations=5,  # Default to 5 iterations
            maxTokens=8192,  # Higher limit for planning
            parallel_tool_calls=True,
        )

        # If base params provided, merge with defaults
        if request_params:
            # Create copy of defaults
            params = defaults.model_copy()
            # Update with provided params
            if isinstance(request_params, dict):
                params = params.model_copy(update=request_params)
            else:
                params = params.model_copy(update=request_params.model_dump())

            # Force specific settings
            params.use_history = False
            return params

        return defaults
