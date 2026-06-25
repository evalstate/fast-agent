"""
Chain workflow implementation using the clean BaseAgent adapter pattern.

This provides an implementation that delegates operations to a sequence of
other agents, chaining their outputs together.
"""

from typing import Any

from mcp import Tool
from mcp.types import TextContent
from opentelemetry import trace

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.request_params import child_request_params
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import ModelT
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import PromptMessageExtended, RequestParams

logger = get_logger(__name__)


def _chain_messages_with_responses(
    messages: list[PromptMessageExtended],
    responses: list[PromptMessageExtended],
) -> list[PromptMessageExtended]:
    chain_messages = messages.copy()
    chain_messages.extend(Prompt.user(response) for response in responses)
    return chain_messages


class ChainAgent(LlmAgent):
    """
    A chain agent that processes requests through a series of specialized agents in sequence.
    Passes the output of each agent to the next agent in the chain.
    """

    # TODO -- consider adding "repeat" mode
    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.CHAIN

    def __init__(
        self,
        config: AgentConfig,
        agents: list[LlmAgent],
        cumulative: bool = False,
        context: Any | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize a ChainAgent.

        Args:
            config: Agent configuration or name
            agents: List of agents to chain together in sequence
            cumulative: Whether each agent sees all previous responses
            context: Optional context object
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, context=context, **kwargs)
        if not agents:
            raise AgentConfigError(f"Chain '{config.name}' requires at least one agent")
        self.agents = agents
        self.cumulative = cumulative

    async def generate_impl(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Chain the request through multiple agents in sequence.

        Args:
            normalized_messages: Already normalized list of PromptMessageExtended
            request_params: Optional request parameters

        Returns:
            The response from the final agent in the chain
        """
        tracer = trace.get_tracer(__name__)
        forward_params = child_request_params(request_params)

        with tracer.start_as_current_span(f"Chain: '{self._name}' generate"):
            # Get the original user message (last message in the list)
            user_message = messages[-1]

            if not self.cumulative:
                # First agent in chain
                async with self.workflow_telemetry.start_step(
                    "chain.step",
                    server_name=self.name,
                    arguments={"agent": self.agents[0].name, "step": 1, "total": len(self.agents)},
                ) as step:
                    response: PromptMessageExtended = await self.agents[0].generate(
                        messages, forward_params
                    )
                    await step.finish(
                        True, text=f"{self.agents[0].name} completed step 1/{len(self.agents)}"
                    )

                # Process the rest of the agents in the chain
                for i, agent in enumerate(self.agents[1:], start=2):
                    async with self.workflow_telemetry.start_step(
                        "chain.step",
                        server_name=self.name,
                        arguments={"agent": agent.name, "step": i, "total": len(self.agents)},
                    ) as step:
                        next_message = Prompt.user(*response.content)
                        response = await agent.generate([next_message], forward_params)
                        await step.finish(
                            True, text=f"{agent.name} completed step {i}/{len(self.agents)}"
                        )

                return response

            # Track all responses in the chain
            all_responses: list[PromptMessageExtended] = []

            # Initialize list for storing formatted results
            final_results: list[str] = []

            # Add the original request with XML tag
            request_text = f"<fastagent:request>{user_message.all_text() or '<no response>'}</fastagent:request>"
            final_results.append(request_text)

            # Process through each agent in sequence
            for i, agent in enumerate(self.agents):
                async with self.workflow_telemetry.start_step(
                    "chain.step",
                    server_name=self.name,
                    arguments={
                        "agent": agent.name,
                        "step": i + 1,
                        "total": len(self.agents),
                        "cumulative": True,
                    },
                ) as step:
                    # In cumulative mode, include the original message and all previous responses
                    chain_messages = _chain_messages_with_responses(messages, all_responses)
                    current_response = await agent.generate(
                        chain_messages,
                        forward_params,
                    )

                    # Store the response
                    all_responses.append(current_response)

                    response_text = current_response.all_text()
                    attributed_response = f"<fastagent:response agent='{agent.name}'>{response_text}</fastagent:response>"
                    final_results.append(attributed_response)
                    await step.finish(
                        True, text=f"{agent.name} completed step {i + 1}/{len(self.agents)}"
                    )

            # For cumulative mode, return the properly formatted output with XML tags
            response_text = "\n\n".join(final_results)
            return PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text=response_text)],
            )

    async def structured_impl(
        self,
        messages: list[PromptMessageExtended],
        model: type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """
        Chain the request through multiple agents and parse the final response.

        Args:
            prompt: List of messages to send through the chain
            model: Pydantic model to parse the final response into
            request_params: Optional request parameters

        Returns:
            The parsed response from the final agent, or None if parsing fails
        """
        tracer = trace.get_tracer(__name__)
        forward_params = child_request_params(request_params)

        with tracer.start_as_current_span(f"Chain: '{self._name}' structured"):
            if not self.cumulative:
                if len(self.agents) == 1:
                    return await self.agents[0].structured(
                        messages,
                        model,
                        forward_params,
                    )

                final_messages = await self._run_non_cumulative_intermediate_agents(
                    messages,
                    forward_params,
                )
                last_agent = self.agents[-1]
                async with self.workflow_telemetry.start_step(
                    "chain.step_structured",
                    server_name=self.name,
                    arguments={
                        "agent": last_agent.name,
                        "step": len(self.agents),
                        "total": len(self.agents),
                    },
                ) as step:
                    structured_response = await last_agent.structured(
                        final_messages,
                        model,
                        forward_params,
                    )
                    await step.finish(
                        True,
                        text=(
                            f"{last_agent.name} produced structured output "
                            f"{len(self.agents)}/{len(self.agents)}"
                        ),
                    )
                    return structured_response

            final_messages = await self._run_cumulative_intermediate_agents(
                messages,
                forward_params,
            )
            last_agent = self.agents[-1]
            async with self.workflow_telemetry.start_step(
                "chain.step_structured",
                server_name=self.name,
                arguments={
                    "agent": last_agent.name,
                    "step": len(self.agents),
                    "total": len(self.agents),
                    "cumulative": True,
                },
            ) as step:
                structured_response = await last_agent.structured(
                    final_messages,
                    model,
                    forward_params,
                )
                await step.finish(
                    True,
                    text=(
                        f"{last_agent.name} produced structured output "
                        f"{len(self.agents)}/{len(self.agents)}"
                    ),
                )
                return structured_response

    async def structured_schema_impl(
        self,
        messages: list[PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        tracer = trace.get_tracer(__name__)
        forward_params = child_request_params(request_params)

        with tracer.start_as_current_span(f"Chain: '{self._name}' structured_schema"):
            if not self.cumulative:
                if len(self.agents) == 1:
                    return await self.agents[0].structured_schema(
                        messages,
                        schema,
                        forward_params,
                    )

                final_messages = await self._run_non_cumulative_intermediate_agents(
                    messages,
                    forward_params,
                )
                last_agent = self.agents[-1]
                async with self.workflow_telemetry.start_step(
                    "chain.step_structured_schema",
                    server_name=self.name,
                    arguments={
                        "agent": last_agent.name,
                        "step": len(self.agents),
                        "total": len(self.agents),
                    },
                ) as step:
                    structured_response = await last_agent.structured_schema(
                        final_messages,
                        schema,
                        forward_params,
                    )
                    await step.finish(
                        True,
                        text=(
                            f"{last_agent.name} produced structured output "
                            f"{len(self.agents)}/{len(self.agents)}"
                        ),
                    )
                    return structured_response

            final_messages = await self._run_cumulative_intermediate_agents(
                messages,
                forward_params,
            )
            last_agent = self.agents[-1]
            async with self.workflow_telemetry.start_step(
                "chain.step_structured_schema",
                server_name=self.name,
                arguments={
                    "agent": last_agent.name,
                    "step": len(self.agents),
                    "total": len(self.agents),
                    "cumulative": True,
                },
            ) as step:
                structured_response = await last_agent.structured_schema(
                    final_messages,
                    schema,
                    forward_params,
                )
                await step.finish(
                    True,
                    text=(
                        f"{last_agent.name} produced structured output "
                        f"{len(self.agents)}/{len(self.agents)}"
                    ),
                )
                return structured_response

    async def _run_non_cumulative_intermediate_agents(
        self,
        messages: list[PromptMessageExtended],
        forward_params: RequestParams | None,
    ) -> list[PromptMessageExtended]:
        response: PromptMessageExtended | None = None
        for i, agent in enumerate(self.agents[:-1], start=1):
            async with self.workflow_telemetry.start_step(
                "chain.step",
                server_name=self.name,
                arguments={
                    "agent": agent.name,
                    "step": i,
                    "total": len(self.agents),
                },
            ) as step:
                step_messages = messages if response is None else [Prompt.user(*response.content)]
                response = await agent.generate(step_messages, forward_params)
                await step.finish(
                    True,
                    text=f"{agent.name} completed step {i}/{len(self.agents)}",
                )

        if response is None:
            raise AgentConfigError(
                f"Chain '{self.name}' requires at least one intermediate response"
            )
        return [Prompt.user(*response.content)]

    async def _run_cumulative_intermediate_agents(
        self,
        messages: list[PromptMessageExtended],
        forward_params: RequestParams | None,
    ) -> list[PromptMessageExtended]:
        all_responses: list[PromptMessageExtended] = []
        for i, agent in enumerate(self.agents[:-1], start=1):
            async with self.workflow_telemetry.start_step(
                "chain.step",
                server_name=self.name,
                arguments={
                    "agent": agent.name,
                    "step": i,
                    "total": len(self.agents),
                    "cumulative": True,
                },
            ) as step:
                chain_messages = _chain_messages_with_responses(messages, all_responses)
                response = await agent.generate(chain_messages, forward_params)
                all_responses.append(response)
                await step.finish(
                    True,
                    text=f"{agent.name} completed step {i}/{len(self.agents)}",
                )

        return _chain_messages_with_responses(messages, all_responses)

    async def initialize(self) -> None:
        """
        Initialize the chain agent and all agents in the chain.
        """
        if self.initialized:
            return

        await super().initialize()

        # Initialize all agents in the chain if not already initialized
        for agent in self.agents:
            if not agent.initialized:
                await agent.initialize()

    async def shutdown(self) -> None:
        """
        Shutdown the chain agent and all agents in the chain.
        """
        await super().shutdown()

        # Shutdown all agents in the chain
        for agent in self.agents:
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down agent in chain: {e!s}")
