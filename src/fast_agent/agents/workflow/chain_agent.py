"""
Chain workflow implementation using the clean BaseAgent adapter pattern.

This provides an implementation that delegates operations to a sequence of
other agents, chaining their outputs together.
"""

from typing import Any, List, Optional, Tuple, Type

from mcp import Tool
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.interfaces import ModelT
from fast_agent.types import PromptMessageExtended, RequestParams

logger = get_logger(__name__)


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
        agents: List[LlmAgent],
        cumulative: bool = False,
        context: Optional[Any] = None,
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
        self.agents = agents
        self.cumulative = cumulative

    async def generate_impl(
        self,
        messages: List[PromptMessageExtended],
        request_params: Optional[RequestParams] = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Chain the request through multiple agents in sequence.

        Args:
            normalized_messages: Already normalized list of PromptMessageExtended
            request_params: Optional request parameters

        Returns:
            The response from the final agent in the chain
        """
        # Get the original user message (last message in the list)
        user_message = messages[-1]

        if not self.cumulative:
            response: PromptMessageExtended = await self.agents[0].generate(
                messages, request_params
            )
            # Process the rest of the agents in the chain
            for agent in self.agents[1:]:
                next_message = Prompt.user(*response.content)
                response = await agent.generate([next_message], request_params)

            return response

        # Track all responses in the chain
        all_responses: List[PromptMessageExtended] = []

        # Initialize list for storing formatted results
        final_results: List[str] = []

        # Add the original request with XML tag
        request_text = (
            f"<fastagent:request>{user_message.all_text() or '<no response>'}</fastagent:request>"
        )
        final_results.append(request_text)

        # Process through each agent in sequence
        for i, agent in enumerate(self.agents):
            # In cumulative mode, include the original message and all previous responses
            chain_messages = messages.copy()

            # Convert previous assistant responses to user messages for the next agent
            for prev_response in all_responses:
                chain_messages.append(Prompt.user(prev_response.all_text()))

            current_response = await agent.generate(
                chain_messages,
                request_params,
            )

            # Store the response
            all_responses.append(current_response)

            response_text = current_response.all_text()
            attributed_response = (
                f"<fastagent:response agent='{agent.name}'>{response_text}</fastagent:response>"
            )
            final_results.append(attributed_response)

        # For cumulative mode, return the properly formatted output with XML tags
        response_text = "\n\n".join(final_results)
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=response_text)],
        )

    async def structured_impl(
        self,
        messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        """
        Chain the request through multiple agents and parse the final response.

        Args:
            prompt: List of messages to send through the chain
            model: Pydantic model to parse the final response into
            request_params: Optional request parameters

        Returns:
            The parsed response from the final agent, or None if parsing fails
        """
        # Generate response through the chain
        response = await self.generate(messages, request_params)
        last_agent = self.agents[-1]
        try:
            return await last_agent.structured([response], model, request_params)
        except Exception as e:
            logger.warning(f"Failed to parse response from chain: {str(e)}")
            return None, Prompt.assistant("Failed to parse response from chain: {str(e)}")

    async def initialize(self) -> None:
        """
        Initialize the chain agent and all agents in the chain.
        """
        await super().initialize()

        # Initialize all agents in the chain if not already initialized
        for agent in self.agents:
            if not getattr(agent, "initialized", False):
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
                logger.warning(f"Error shutting down agent in chain: {str(e)}")
