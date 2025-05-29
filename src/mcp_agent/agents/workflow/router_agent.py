"""
Router agent implementation using the BaseAgent adapter pattern.

This provides a simplified implementation that routes messages to agents
by determining the best agent for a request and dispatching to it.
"""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type

from opentelemetry import trace
from pydantic import BaseModel

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol, ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

if TYPE_CHECKING:
    from a2a_types.types import AgentCard

    from mcp_agent.context import Context

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
DEFAULT_ROUTING_INSTRUCTION = """
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


class RouterAgent(BaseAgent):
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
        agents: List[Agent],
        routing_instruction: Optional[str] = None,
        context: Optional["Context"] = None,
        default_request_params: Optional[RequestParams] = None,
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

        # Set up base router request parameters
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
                if not getattr(agent, "initialized", False):
                    await agent.initialize()

            self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown the router and all agents."""
        await super().shutdown()

        # Shutdown all agents
        for agent in self.agents:
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down agent: {str(e)}")

    async def attach_llm(
        self,
        llm_factory: type[AugmentedLLMProtocol] | Callable[..., AugmentedLLMProtocol],
        model: str | None = None,
        request_params: RequestParams | None = None,
        **additional_kwargs,
    ) -> AugmentedLLMProtocol:
        return await super().attach_llm(
            llm_factory, model, request_params, verb="Routing", **additional_kwargs
        )

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Route the request to the most appropriate agent and return its response.

        Args:
            multipart_messages: Messages to route
            request_params: Optional request parameters

        Returns:
            The response from the selected agent
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Routing: '{self.name}' generate"):
            route, warn = await self._route_request(multipart_messages[-1])

            if not route:
                return Prompt.assistant(warn or "No routing result or warning received")

            # Get the selected agent
            agent: Agent = self.agent_map[route.agent]

            # Dispatch the request to the selected agent
            return await agent.generate(multipart_messages, request_params)

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Route the request to the most appropriate agent and parse its response.

        Args:
            prompt: Messages to route
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed response from the selected agent, or None if parsing fails
        """

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Routing: '{self.name}' structured"):
            route, warn = await self._route_request(multipart_messages[-1])

            if not route:
                return None, Prompt.assistant(
                    warn or "No routing result or warning received (structured)"
                )

            # Get the selected agent
            agent: Agent = self.agent_map[route.agent]

            # Dispatch the request to the selected agent
            return await agent.structured(multipart_messages, model, request_params)

    async def _route_request(
        self, message: PromptMessageMultipart
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

        # If only one agent is available, use it directly
        if len(self.agents) == 1:
            return RoutingResponse(
                agent=self.agents[0].name, confidence="high", reasoning="Only one agent available"
            ), None

        # Generate agent descriptions for the context
        agent_descriptions = []
        for agent in self.agents:
            agent_card: AgentCard = await agent.agent_card()
            agent_descriptions.append(
                agent_card.model_dump_json(
                    include={"name", "description", "skills"}, exclude_none=True
                )
            )

        context = ",\n".join(agent_descriptions)

        # Format the routing prompt
        routing_instruction = self.routing_instruction or DEFAULT_ROUTING_INSTRUCTION
        routing_instruction = routing_instruction.format(context=context)

        assert self._llm
        mutated = message.model_copy(deep=True)
        mutated.add_text(routing_instruction)
        response, _ = await self._llm.structured(
            [mutated],
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

            return response, None
