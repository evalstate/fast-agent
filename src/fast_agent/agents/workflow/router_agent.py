"""
Router agent implementation using the BaseAgent adapter pattern.

This provides a simplified implementation that routes messages to agents
by determining the best agent for a request and dispatching to it.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Type

from mcp import Tool
from opentelemetry import trace
from pydantic import BaseModel

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.interfaces import FastAgentLLMProtocol, LLMFactoryProtocol, ModelT
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from a2a.types import AgentCard

    from fast_agent.context import Context

logger = get_logger(__name__)

# Simple system instruction for the router
ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate agent.
Analyze each request and determine which specialized agent would be best suited to handle it based on their capabilities.

Follow these guidelines:
- Carefully match the request's needs with each agent's capabilities and description
- Select the single most appropriate agent for the request
- Provide your confidence level (high, medium, low) and brief reasoning for your selection
"""

# Default routing instruction with placeholders for context (AgentCard JSON)
ROUTING_AGENT_INSTRUCTION = """
Select from the following agents to handle the request:
<fastagent:agents>
[
{context}
]
</fastagent:agents>

You must respond with the 'name' of one of the agents listed above.

"""


class RoutingResponse(BaseModel):
    """Model for the structured routing response from the LLM."""

    agent: str
    confidence: str
    reasoning: str | None = None


class RouterAgent(LlmAgent):
    """
    A simplified router that uses an LLM to determine the best agent for a request,
    then dispatches the request to that agent and returns the response.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.ROUTER

    def __init__(
        self,
        config: AgentConfig,
        agents: List[LlmAgent],
        routing_instruction: str | None = None,
        context: "Context | None" = None,
        default_request_params: RequestParams | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize a RouterAgent.

        Args:
            config: Agent configuration or name
            agents: List of agents to route between
            routing_instruction: Optional custom routing instruction
            context: Optional application context
            default_request_params: Optional default request parameters
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config=config, context=context, **kwargs)

        if not agents:
            raise AgentConfigError("At least one agent must be provided")

        self.agents = agents
        self.routing_instruction = routing_instruction
        self.agent_map = {agent.name: agent for agent in agents}

        # Set up base router request parameters with just the base instruction for now
        base_params = {"systemPrompt": ROUTING_SYSTEM_INSTRUCTION, "use_history": False}

        if default_request_params:
            merged_params = default_request_params.model_copy(update=base_params)
        else:
            merged_params = RequestParams(**base_params)

        self._default_request_params = merged_params

    async def initialize(self) -> None:
        """Initialize the router and all agents."""
        if not self.initialized:
            await super().initialize()

            # Initialize all agents if not already initialized
            for agent in self.agents:
                if not agent.initialized:
                    await agent.initialize()

            complete_routing_instruction = await self._generate_routing_instruction(
                self.agents, self.routing_instruction
            )

            # Update the system prompt to include the routing instruction with agent cards
            combined_system_prompt = (
                ROUTING_SYSTEM_INSTRUCTION + "\n\n" + complete_routing_instruction
            )
            self._default_request_params.systemPrompt = combined_system_prompt
            self.instruction = combined_system_prompt

    async def shutdown(self) -> None:
        """Shutdown the router and all agents."""
        await super().shutdown()

        # Shutdown all agents
        for agent in self.agents:
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down agent: {str(e)}")

    @staticmethod
    async def _generate_routing_instruction(
        agents: List[LlmAgent], routing_instruction: Optional[str] = None
    ) -> str:
        """
        Generate the complete routing instruction with agent cards.

        Args:
            agents: List of agents to include in routing instruction
            routing_instruction: Optional custom routing instruction template

        Returns:
            Complete routing instruction with agent cards formatted
        """
        # Generate agent descriptions
        agent_descriptions = []
        for agent in agents:
            agent_card: AgentCard = await agent.agent_card()
            agent_descriptions.append(
                agent_card.model_dump_json(
                    include={"name", "description", "skills"}, exclude_none=True
                )
            )

        context = ",\n".join(agent_descriptions)

        # Format the routing instruction
        instruction_template = routing_instruction or ROUTING_AGENT_INSTRUCTION
        return instruction_template.format(context=context)

    async def attach_llm(
        self,
        llm_factory: LLMFactoryProtocol,
        model: str | None = None,
        request_params: RequestParams | None = None,
        **additional_kwargs,
    ) -> FastAgentLLMProtocol:
        return await super().attach_llm(
            llm_factory, model, request_params, verb="Routing", **additional_kwargs
        )

    async def generate_impl(
        self,
        messages: List[PromptMessageExtended],
        request_params: Optional[RequestParams] = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Route the request to the most appropriate agent and return its response.

        Args:
            normalized_messages: Already normalized list of PromptMessageExtended
            request_params: Optional request parameters

        Returns:
            The response from the selected agent
        """

        # implementation note. the duplication between generated and structured
        # is probably the most readable. alt could be a _get_route_agent or
        # some form of dynamic dispatch.. but only if this gets more complex
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Routing: '{self.name}' generate"):
            route, warn = await self._route_request(messages[-1])

            if not route:
                return Prompt.assistant(warn or "No routing result or warning received")

            # Get the selected agent
            agent: LlmAgent = self.agent_map[route.agent]

            # Dispatch the request to the selected agent
            # discarded request_params: use llm defaults for subagents
            return await agent.generate_impl(messages)

    async def structured_impl(
        self,
        messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        """
        Route the request to the most appropriate agent and parse its response.

        Args:
            messages: Messages to route
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed response from the selected agent, or None if parsing fails
        """

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Routing: '{self.name}' structured"):
            route, warn = await self._route_request(messages[-1])

            if not route:
                return None, Prompt.assistant(
                    warn or "No routing result or warning received (structured)"
                )

            # Get the selected agent
            agent: LlmAgent = self.agent_map[route.agent]

            # Dispatch the request to the selected agent
            return await agent.structured_impl(messages, model, request_params)

    async def _route_request(
        self, message: PromptMessageExtended
    ) -> Tuple[RoutingResponse | None, str | None]:
        """
        Determine which agent to route the request to.

        Args:
            request: The request to route

        Returns:
            RouterResult containing the selected agent, or None if no suitable agent was found
        """
        if not self.agents:
            logger.error("No agents available for routing")
            raise AgentConfigError("No agents available for routing - fatal error")

        # go straight to agent if only one available
        if len(self.agents) == 1:
            return RoutingResponse(
                agent=self.agents[0].name, confidence="high", reasoning="Only one agent available"
            ), None

        assert self._llm
        # Display the user's routing request
        self.display.show_user_message(message.first_text(), name=self.name)

        # No need to add routing instruction here - it's already in the system prompt
        response, _ = await self._llm.structured(
            [message],
            RoutingResponse,
            self._default_request_params,
        )

        warn: str | None = None
        if not response:
            warn = "No routing response received from LLM"
        elif response.agent not in self.agent_map:
            warn = f"A response was received, but the agent {response.agent} was not known to the Router"

        if warn:
            logger.warning(warn)
            return None, warn
        else:
            assert response
            logger.info(
                f"Routing structured request to agent: {response.agent or 'error'} (confidence: {response.confidence or ''})"
            )

            routing_message = f"Routing to: {response.agent}"
            if response.reasoning:
                routing_message += f" ({response.reasoning})"

            # Convert highlight_items to highlight_index
            agent_keys = list(self.agent_map.keys())
            highlight_index = None
            try:
                highlight_index = agent_keys.index(response.agent)
            except ValueError:
                pass

            await self.display.show_assistant_message(
                routing_message,
                bottom_items=agent_keys,
                highlight_index=highlight_index,
                name=self.name,
            )

            return response, None
