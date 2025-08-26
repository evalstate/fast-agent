"""
Iterative Planner Agent - works towards an objective using sub-agents
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Type

from mcp.types import TextContent

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.agents.workflow.orchestrator_models import (
    Plan,
    PlanningStep,
    PlanResult,
    Step,
    TaskWithResult,
    format_plan_result,
    format_step_result_text,
)
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

logger = get_logger(__name__)


ITERATIVE_PLAN_SYSTEM_PROMPT_TEMPLATE = """
You are an expert planner, able to Orchestrate complex tasks by breaking them down in to
manageable steps, and delegating tasks to Agents.

You work iteratively - given an Objective, you consider the current state of the plan,
decide the next step towards the goal. You document those steps and create clear instructions
for execution by the Agents, being specific about what you need to know to assess task completion. 

NOTE: A 'Planning Step' has a description, and a list of tasks that can be delegated 
and executed in parallel.

Agents have a 'description' describing their primary function, and a set of 'skills' that
represent Tools they can use in completing their function.

The following Agents are available to you:

{{agents}}

You must specify the Agent name precisely when generating a Planning Step. 

"""


ITERATIVE_PLAN_PROMPT_TEMPLATE2 = """

```
<fastagent:data>
    <fastagent:progress>
{plan_result}
    </fastagent:progress>

    <fastagent:status>
{plan_status}
{iterations_info}
    </fastagent:status>
</fastagent:data>
```

The overall objective is:

```
<fastagent:objective>
    {objective}
</fastagent:objective>
```

Produce the next step in the plan to complete the Objective. 

Consider the previous steps and results, and decide what needs to be done next.

Set "is_complete" to true when ANY of these conditions are met:
1. The objective has been achieved in full or substantively
2. The remaining work is minor or trivial compared to what's been accomplished
3. The plan has gathered sufficient information to answer the original request
4. The plan has no feasible way of completing the objective.

Only set is_complete to `true` if there are no outstanding tasks. 

Be decisive - avoid excessive planning steps that add little value. It's better to complete a plan early than to continue with marginal improvements. 

Focus on the meeting the core intent of the objective.

"""


DELEGATED_TASK_TEMPLATE = """
You are a component in an orchestrated workflow to achieve an objective.

The overall objective is:

```
<fastagent:objective>
    {objective}
</fastagent:objective>
```

Previous context in achieving the objective is below:

```
<fastagent:context>
    {context}
</fastagent:context>
```

Your job is to accomplish the "task" specified in `<fastagent:task>`. The overall objective and
previous context is supplied to inform your approach.

```
<fastagent:task>
   {task}
</fastagent:task>
```

Provide a direct, concise response on completion that makes it simple to assess the status of
the overall plan.
"""


PLAN_RESULT_TEMPLATE = """
The below shows the results of running a plan to meet the specified objective.

```
<fastagent:plan-results>
{plan_result}
</fastagent:plan-results>
```

The plan was stopped because {termination_reason}.

Provide a summary of the tasks completed and their outcomes to complete the Objective.
Use markdown formatting.

If the plan was marked as incomplete but the maximum number of iterations was reached,
make sure to state clearly what was accomplished and what remains to be done.


Complete the plan by providing an appropriate answer for the original objective. Provide a Mermaid diagram
(in code fences) showing the plan steps and their relationships, if applicable.
"""


class IterativePlanner(BaseAgent):
    """
    An agent that implements the orchestrator workflow pattern.

    Dynamically creates execution plans and delegates tasks
    to specialized worker agents, synthesizing their results into a cohesive output.
    Supports both full planning and iterative planning modes.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.ITERATIVE_PLANNER

    def __init__(
        self,
        config: AgentConfig,
        agents: List[Agent],
        plan_iterations: int = -1,
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
        if not agents:
            raise AgentConfigError("At least one worker agent must be provided")

        # Store agents by name for easier lookup
        self.agents: Dict[str, Agent] = {}
        for agent in agents:
            agent_name = agent.name
            self.agents[agent_name] = agent

        super().__init__(config, context=context, **kwargs)

        self.plan_iterations = plan_iterations

    async def initialize(self) -> None:
        """Initialize the orchestrator agent and worker agents."""
        # Initialize all worker agents first if not already initialized
        for agent_name, agent in self.agents.items():
            if not getattr(agent, "initialized", False):
                self.logger.debug(f"Initializing agent: {agent_name}")
                await agent.initialize()

        # Format agent information using agent cards with XML formatting
        agent_descriptions = []
        for agent_name, agent in self.agents.items():
            agent_card = await agent.agent_card()
            # Format as XML for better readability in prompts
            xml_formatted = self._format_agent_card_as_xml(agent_card)
            agent_descriptions.append(xml_formatted)

        agents_str = "\n".join(agent_descriptions)

        # Replace {{agents}} placeholder in the system prompt template
        system_prompt = self.config.instruction.replace("{{agents}}", agents_str)

        # Update the config instruction with the formatted system prompt
        self.instruction = system_prompt

        # Initialize the base agent with the updated system prompt
        await super().initialize()

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
        plan_result = await self._execute_plan(objective, request_params)

        # Return the result
        return PromptMessageMultipart(
            role="assistant",
            content=[TextContent(type="text", text=plan_result.result or "No result available")],
        )

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
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
        response = await self.generate(multipart_messages, request_params)

        # Try to parse the response into the specified model
        try:
            result_text = response.last_text()
            prompt_message = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=result_text)]
            )
            assert self._llm
            return await self._llm.structured([prompt_message], model, request_params)
        except Exception as e:
            self.logger.warning(f"Failed to parse orchestration result: {str(e)}")
            return None, Prompt.assistant(f"Failed to parse orchestration result: {str(e)}")

    async def _execute_plan(
        self, objective: str, request_params: RequestParams | None
    ) -> PlanResult:
        """
        Execute a plan to achieve the given objective.

        Args:
            objective: The objective to achieve
            request_params: Request parameters for execution

        Returns:
            PlanResult containing execution results and final output
        """

        objective_met: bool = False
        terminate_plan: str | None = None
        plan_result = PlanResult(objective=objective, step_results=[])

        while not objective_met and not terminate_plan:
            next_step: PlanningStep | None = await self._get_next_step(
                objective, plan_result, request_params
            )

            if None is next_step:
                terminate_plan = "Failed to generate plan, terminating early"
                self.logger.error("Failed to generate next step, terminating plan early")
                break

            assert next_step  # lets keep the indenting manageable!

            if next_step.is_complete:
                objective_met = True
                terminate_plan = "Plan completed successfully"
                break

            plan = Plan(steps=[next_step], is_complete=next_step.is_complete)
            invalid_agents = self._validate_agent_names(plan)
            if invalid_agents:
                self.logger.error(f"Plan contains invalid agent names: {', '.join(invalid_agents)}")
                terminate_plan = (
                    f"Invalid agent names found ({', '.join(invalid_agents)}), terminating plan"
                )
                break

            for step in plan.steps:  # this will only be one for iterative (change later)
                step_result = await self._execute_step(step, plan_result)
                plan_result.add_step_result(step_result)

            # Store plan in result
            plan_result.plan = plan

            if self.plan_iterations > 0:
                if len(plan_result.step_results) >= self.plan_iterations:
                    terminate_plan = f"Reached maximum number of iterations ({self.plan_iterations}), terminating plan"

        if not terminate_plan:
            terminate_plan = "Unknown termination reason"
        result_prompt = PLAN_RESULT_TEMPLATE.format(
            plan_result=format_plan_result(plan_result), termination_reason=terminate_plan
        )

        # Generate final synthesis
        plan_result.result = await self._planner_generate_str(result_prompt, request_params)
        return plan_result

    async def _execute_step(self, step: Step, previous_result: PlanResult) -> Any:
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

        # Group tasks by agent and execute different agents in parallel
        from collections import defaultdict

        tasks_by_agent = defaultdict(list)
        for task in step.tasks:
            tasks_by_agent[task.agent].append(task)

        async def execute_agent_tasks(agent_name: str, agent_tasks: List) -> List[TaskWithResult]:
            """Execute all tasks for a single agent sequentially (preserves history)"""
            agent = self.agents.get(agent_name)
            assert agent is not None

            results = []
            for task in agent_tasks:
                try:
                    task_description = DELEGATED_TASK_TEMPLATE.format(
                        objective=previous_result.objective, task=task.description, context=context
                    )
                    result = await agent.generate(
                        [
                            PromptMessageMultipart(
                                role="user",
                                content=[TextContent(type="text", text=task_description)],
                            )
                        ]
                    )

                    task_model = task.model_dump()
                    results.append(
                        TaskWithResult(
                            description=task_model["description"],
                            agent=task_model["agent"],
                            result=result.last_text(),
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error executing task: {str(e)}")
                    task_model = task.model_dump()
                    results.append(
                        TaskWithResult(
                            description=task_model["description"],
                            agent=task_model["agent"],
                            result=f"ERROR: {str(e)}",
                        )
                    )
            return results

        # Execute different agents in parallel, tasks within each agent sequentially
        agent_futures = [
            execute_agent_tasks(agent_name, agent_tasks)
            for agent_name, agent_tasks in tasks_by_agent.items()
        ]

        all_results = await asyncio.gather(*agent_futures)
        task_results = [result for agent_results in all_results for result in agent_results]

        # Add all task results to step result
        for task_result in task_results:
            step_result.add_task_result(task_result)

        # Format step result
        step_result.result = format_step_result_text(step_result)
        return step_result

    async def _get_next_step(
        self, objective: str, plan_result: PlanResult, request_params: RequestParams | None
    ) -> PlanningStep | None:
        """
        Generate just the next step for iterative planning.

        Args:
            objective: The objective to achieve
            plan_result: Current plan execution state
            request_params: Request parameters

        Returns:
            Next step to execute, or None if parsing fails
        """

        # Determine plan status
        if plan_result.is_complete:
            plan_status = "Plan Status: Complete"
        elif plan_result.step_results:
            plan_status = "Plan Status: In Progress"
        else:
            plan_status = "Plan Status: Not Started"

        # Calculate iteration information

        if self.plan_iterations > 0:
            max_iterations = self.plan_iterations
            current_iteration = len(plan_result.step_results)
            iterations_remaining = max_iterations - current_iteration
            iterations_info = (
                f"Planning Budget: {iterations_remaining} of {max_iterations} iterations remaining"
            )
        else:
            iterations_info = "Iterating until objective is met."

        # Format the planning prompt
        prompt = ITERATIVE_PLAN_PROMPT_TEMPLATE2.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            plan_status=plan_status,
            iterations_info=iterations_info,
        )

        # Get structured response from LLM
        try:
            plan_msg = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=prompt)]
            )
            assert self._llm
            next_step, _ = await self._llm.structured([plan_msg], PlanningStep, request_params)
            return next_step
        except Exception as e:
            self.logger.error(f"Failed to parse next step: {str(e)}")
            return None

    def _validate_agent_names(self, plan: Plan) -> List[str]:
        """
        Validate all agent names in a plan before execution.

        Args:
            plan: The plan to validate
        """
        invalid_agents = []

        for step in plan.steps:
            for task in step.tasks:
                if task.agent not in self.agents:
                    invalid_agents.append(task.agent)

        return invalid_agents

    @staticmethod
    def _format_agent_card_as_xml(agent_card) -> str:
        """
        Format an agent card as XML for display in prompts.

        This creates a structured XML representation that's more readable than JSON
        and includes all relevant agent information in a hierarchical format.

        Args:
            agent_card: The AgentCard object

        Returns:
            XML formatted agent information
        """
        xml_parts = [f'<fastagent:agent name="{agent_card.name}">']

        # Add description if available
        if agent_card.description:
            xml_parts.append(
                f"  <fastagent:description>{agent_card.description}</fastagent:description>"
            )

        # Add skills if available
        if hasattr(agent_card, "skills") and agent_card.skills:
            xml_parts.append("  <fastagent:skills>")
            for skill in agent_card.skills:
                xml_parts.append(f'    <fastagent:skill name="{skill.name}">')
                if hasattr(skill, "description") and skill.description:
                    xml_parts.append(
                        f"      <fastagent:description>{skill.description}</fastagent:description>"
                    )
                xml_parts.append("    </fastagent:skill>")
            xml_parts.append("  </fastagent:skills>")

        xml_parts.append("</fastagent:agent>")

        return "\n".join(xml_parts)

    async def _planner_generate_str(
        self, message: str, request_params: RequestParams | None
    ) -> str:
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
        assert self._llm, "LLM must be initialized before generating text"
        response = await self._llm.generate([prompt], request_params)
        return response.last_text()
