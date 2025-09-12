import asyncio
from typing import Any, List, Optional, Tuple

from mcp import Tool
from mcp.types import TextContent
from opentelemetry import trace

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import AgentProtocol, ModelT
from fast_agent.types import PromptMessageExtended, RequestParams

logger = get_logger(__name__)


class ParallelAgent(LlmAgent):
    """
    LLMs can sometimes work simultaneously on a task (fan-out)
    and have their outputs aggregated programmatically (fan-in).
    This workflow performs both the fan-out and fan-in operations using LLMs.
    From the user's perspective, an input is specified and the output is returned.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.PARALLEL

    def __init__(
        self,
        config: AgentConfig,
        fan_in_agent: AgentProtocol,
        fan_out_agents: List[AgentProtocol],
        include_request: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize a ParallelLLM agent.

        Args:
            config: Agent configuration or name
            fan_in_agent: Agent that aggregates results from fan-out agents
            fan_out_agents: List of agents to execute in parallel
            include_request: Whether to include the original request in the aggregation
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, **kwargs)
        self.fan_in_agent = fan_in_agent
        self.fan_out_agents = fan_out_agents
        self.include_request = include_request

    async def generate_impl(
        self,
        messages: List[PromptMessageExtended],
        request_params: Optional[RequestParams] = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Execute fan-out agents in parallel and aggregate their results with the fan-in agent.

        Args:
            normalized_messages: Already normalized list of PromptMessageExtended
            request_params: Optional parameters to configure the request

        Returns:
            The aggregated response from the fan-in agent
        """

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Parallel: '{self._name}' generate"):
            # Execute all fan-out agents in parallel
            responses: List[PromptMessageExtended] = await asyncio.gather(
                *[agent.generate(messages, request_params) for agent in self.fan_out_agents]
            )

            # Extract the received message from the input
            received_message: Optional[str] = messages[-1].all_text() if messages else None

            # Convert responses to strings for aggregation
            string_responses = []
            for response in responses:
                string_responses.append(response.all_text())

            # Format the responses and send to the fan-in agent
            aggregated_prompt = self._format_responses(string_responses, received_message)

            # Create a new multipart message with the formatted responses
            formatted_prompt = PromptMessageExtended(
                role="user", content=[TextContent(type="text", text=aggregated_prompt)]
            )

            # Use the fan-in agent to aggregate the responses
            return await self.fan_in_agent.generate([formatted_prompt], request_params)

    def _format_responses(self, responses: List[Any], message: Optional[str] = None) -> str:
        """
        Format a list of responses for the fan-in agent.

        Args:
            responses: List of responses from fan-out agents
            message: Optional original message that was sent to the agents

        Returns:
            Formatted string with responses
        """
        formatted = []

        # Include the original message if specified
        if self.include_request and message:
            formatted.append("The following request was sent to the agents:")
            formatted.append(f"<fastagent:request>\n{message}\n</fastagent:request>")

        # Format each agent's response
        for i, response in enumerate(responses):
            agent_name = self.fan_out_agents[i].name
            formatted.append(
                f'<fastagent:response agent="{agent_name}">\n{response}\n</fastagent:response>'
            )
        return "\n\n".join(formatted)

    async def structured_impl(
        self,
        messages: List[PromptMessageExtended],
        model: type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        """
        Apply the prompt and return the result as a Pydantic model.

        This implementation delegates to the fan-in agent's structured method.

        Args:
            messages: List of PromptMessageExtended objects
            model: The Pydantic model class to parse the result into
            request_params: Optional parameters to configure the LLM request

        Returns:
            An instance of the specified model, or None if coercion fails
        """

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Parallel: '{self._name}' generate"):
            # Generate parallel responses first
            responses: List[PromptMessageExtended] = await asyncio.gather(
                *[agent.generate(messages, request_params) for agent in self.fan_out_agents]
            )

            # Extract the received message
            received_message: Optional[str] = messages[-1].all_text() if messages else None

            # Convert responses to strings
            string_responses = [response.all_text() for response in responses]

            # Format the responses for the fan-in agent
            aggregated_prompt = self._format_responses(string_responses, received_message)

            # Create a multipart message
            formatted_prompt = PromptMessageExtended(
                role="user", content=[TextContent(type="text", text=aggregated_prompt)]
            )

            # Use the fan-in agent to parse the structured output
            return await self.fan_in_agent.structured([formatted_prompt], model, request_params)

    async def initialize(self) -> None:
        """
        Initialize the agent and its fan-in and fan-out agents.
        """
        await super().initialize()

        # Initialize fan-in and fan-out agents if not already initialized
        if not self.fan_in_agent.initialized:
            await self.fan_in_agent.initialize()

        for agent in self.fan_out_agents:
            if not agent.initialized:
                await agent.initialize()

    async def shutdown(self) -> None:
        """
        Shutdown the agent and its fan-in and fan-out agents.
        """
        await super().shutdown()

        # Shutdown fan-in and fan-out agents
        try:
            await self.fan_in_agent.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down fan-in agent: {str(e)}")

        for agent in self.fan_out_agents:
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down fan-out agent {agent.name}: {str(e)}")
