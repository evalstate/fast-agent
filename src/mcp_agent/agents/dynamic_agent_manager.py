"""
Dynamic Agent Manager for creating and managing agents at runtime.

This manager handles the lifecycle of dynamic agents, following the same patterns
as parallel agents for execution and communication.
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

from a2a.types import AgentCard

from mcp_agent.agents.agent import Agent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from mcp_agent.core.direct_factory import get_model_factory
from mcp_agent.core.prompt import Prompt
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.agents.base_agent import BaseAgent

logger = get_logger(__name__)


@dataclass
class DynamicAgentSpec:
    """Specification for creating a dynamic agent."""
    name: str
    instruction: str
    servers: List[str]
    tools: Optional[Dict[str, List[str]]] = None
    model: Optional[str] = None


def create_dynamic_agent_card(
    agent_id: str,
    name: str,
    description: str,
    servers: List[str],
    status: str = "active",
    context_tokens_used: int = 0,
    last_activity: Optional[str] = None
) -> AgentCard:
    """Create an AgentCard for a dynamic agent."""
    from a2a.types import AgentCapabilities, AgentSkill
    
    # Create skills from servers
    skills = []
    for server in servers:
        skills.append(AgentSkill(
            id=f"mcp_{server}",
            name=f"mcp_{server}",
            description=f"Access to {server} MCP server",
            tags=["mcp", "server", server]
        ))
    
    # Add status and metadata as additional skills
    skills.append(AgentSkill(
        id="agent_status",
        name="agent_status",
        description=f"Agent status: {status}",
        tags=["status", "metadata"]
    ))
    
    if context_tokens_used > 0:
        skills.append(AgentSkill(
            id="usage_info",
            name="usage_info",
            description=f"Context tokens used: {context_tokens_used}",
            tags=["usage", "metadata"]
        ))
    
    return AgentCard(
        name=name,
        description=description,
        url=f"fast-agent://dynamic-agents/{agent_id}/",
        version="0.1",
        capabilities=AgentCapabilities(
            supportsStreaming=False,
            supportsFunctionCalling=True,
            supportsToolUse=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=skills,
        provider=None,
        documentationUrl=None
    )


class DynamicAgentManager:
    """
    Manages dynamic agents for a parent agent.
    
    Follows the same patterns as ParallelAgent for execution and communication,
    but allows creating agents at runtime based on task needs.
    """
    
    def __init__(self, parent_agent: "BaseAgent") -> None:
        """
        Initialize the dynamic agent manager.
        
        Args:
            parent_agent: The agent that owns this manager
        """
        self.parent_agent = parent_agent
        self.dynamic_agents: Dict[str, Agent] = {}
        self.max_agents = parent_agent.config.max_dynamic_agents
        self.logger = get_logger(f"{__name__}.{parent_agent.name}")
        
    async def create_agent(self, spec: DynamicAgentSpec) -> str:
        """
        Create a new dynamic agent.
        
        Args:
            spec: Specification for the agent to create
            
        Returns:
            agent_id: Unique identifier for the created agent
            
        Raises:
            ValueError: If max agents limit reached or invalid specification
        """
        # Check limits
        if len(self.dynamic_agents) >= self.max_agents:
            raise ValueError(f"Maximum number of dynamic agents ({self.max_agents}) reached")
            
        # Validate servers exist in parent's context
        available_servers = self.parent_agent.server_names
        invalid_servers = set(spec.servers) - set(available_servers)
        if invalid_servers:
            raise ValueError(f"Invalid servers: {invalid_servers}. Available: {available_servers}")
        
        # Generate unique agent ID
        agent_id = f"{spec.name}_{uuid.uuid4().hex[:6]}"
        
        # Create agent config
        config = AgentConfig(
            name=spec.name,
            instruction=spec.instruction,
            servers=spec.servers,
            tools=spec.tools,
            model=spec.model or self.parent_agent.config.model,
            use_history=True,  # Each dynamic agent has its own context
            agent_type=AgentType.BASIC
        )
        
        # Create the agent using existing patterns
        agent = Agent(
            config=config,
            context=self.parent_agent._context  # Share context for MCP connections
        )
        
        # Initialize the agent
        await agent.initialize()
        
        # Attach LLM using the same process as factory
        model_factory = get_model_factory(
            context=self.parent_agent._context,
            model=config.model,
            default_model=self.parent_agent.config.model
        )
        
        await agent.attach_llm(
            model_factory,
            request_params=config.default_request_params,
            api_key=config.api_key
        )
        
        # Store the agent
        self.dynamic_agents[agent_id] = agent
        
        self.logger.info(
            f"Created dynamic agent '{spec.name}' with ID {agent_id}",
            data={
                "agent_id": agent_id,
                "name": spec.name,
                "servers": spec.servers,
                "total_agents": len(self.dynamic_agents)
            }
        )
        
        return agent_id
    
    async def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate a dynamic agent and clean up resources.
        
        Args:
            agent_id: ID of the agent to terminate
            
        Returns:
            success: True if terminated successfully
        """
        if agent_id not in self.dynamic_agents:
            return False
            
        agent = self.dynamic_agents[agent_id]
        
        try:
            await agent.shutdown()
            del self.dynamic_agents[agent_id]
            
            self.logger.info(
                f"Terminated dynamic agent {agent_id}",
                data={
                    "agent_id": agent_id,
                    "remaining_agents": len(self.dynamic_agents)
                }
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error terminating agent {agent_id}: {e}")
            return False
    
    async def send_to_agent(self, agent_id: str, message: str) -> str:
        """
        Send a message to a specific dynamic agent.
        
        Args:
            agent_id: ID of the agent to send to
            message: Message to send
            
        Returns:
            response: The agent's response
            
        Raises:
            ValueError: If agent_id not found
        """
        if agent_id not in self.dynamic_agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        agent = self.dynamic_agents[agent_id]
        response = await agent.send(message)
        
        self.logger.debug(
            f"Sent message to agent {agent_id}",
            data={"agent_id": agent_id, "message_length": len(message)}
        )
        
        return response
    
    async def broadcast_message(
        self, 
        message: str, 
        agent_ids: Optional[List[str]] = None,
        parallel: bool = True
    ) -> Dict[str, str]:
        """
        Send a message to multiple dynamic agents.
        
        Uses the EXACT same parallel execution pattern as ParallelAgent.
        
        Args:
            message: Message to send to agents
            agent_ids: Specific agents to send to (if None, sends to all)
            parallel: Execute in parallel (True) or sequential (False)
            
        Returns:
            responses: Dict mapping agent_id to response
        """
        # Get agents to execute
        if agent_ids:
            # Validate all agent IDs exist
            missing_ids = set(agent_ids) - set(self.dynamic_agents.keys())
            if missing_ids:
                raise ValueError(f"Agent IDs not found: {missing_ids}")
            agents_to_execute = [(id, self.dynamic_agents[id]) for id in agent_ids]
        else:
            agents_to_execute = list(self.dynamic_agents.items())
        
        if not agents_to_execute:
            return {}
        
        # Create prompt message
        prompt_message = [Prompt.user(message)]
        
        if parallel:
            # Execute in parallel - SAME as ParallelAgent
            responses = await asyncio.gather(
                *[agent.generate(prompt_message) for _, agent in agents_to_execute],
                return_exceptions=True
            )
        else:
            # Execute sequentially
            responses = []
            for _, agent in agents_to_execute:
                try:
                    response = await agent.generate(prompt_message)
                    responses.append(response)
                except Exception as e:
                    responses.append(e)
        
        # Process responses
        result = {}
        for i, (agent_id, agent) in enumerate(agents_to_execute):
            response = responses[i]
            if isinstance(response, Exception):
                result[agent_id] = f"Error: {str(response)}"
            else:
                result[agent_id] = response.all_text()
        
        self.logger.info(
            f"Broadcast message to {len(agents_to_execute)} agents",
            data={
                "agent_count": len(agents_to_execute),
                "parallel": parallel,
                "message_length": len(message)
            }
        )
        
        # Display results if console display is available
        if len(result) > 1:  # Only show tree view for multiple agents
            self._show_agent_results(result, message)
        
        return result
    
    def _show_agent_results(self, responses: Dict[str, str], original_message: str = None) -> None:
        """Show dynamic agent results using console display."""
        try:
            # Import here to avoid circular dependencies
            from mcp_agent.ui.console_display import ConsoleDisplay
            
            # Try to get display from parent agent
            display = None
            if hasattr(self.parent_agent, '_context') and self.parent_agent._context:
                display = getattr(self.parent_agent._context, 'display', None)
            
            # Create display if not available
            if not display:
                config = getattr(self.parent_agent, 'config', None)
                display = ConsoleDisplay(config)
            
            # Show results using the same pattern as parallel agents
            display.show_dynamic_agent_results(responses, original_message)
            
        except Exception as e:
            # Silently fail if display not available
            self.logger.debug(f"Could not display dynamic agent results: {e}")
    
    def list_agents(self) -> List[AgentCard]:
        """
        List all active dynamic agents as AgentCard objects.
        
        Returns:
            agents: List of agent cards
        """
        result = []
        for agent_id, agent in self.dynamic_agents.items():
            # Get token usage if available
            tokens_used = 0
            if hasattr(agent, 'usage_accumulator') and agent.usage_accumulator:
                summary = agent.usage_accumulator.get_summary()
                tokens_used = summary.get('cumulative_input_tokens', 0) + summary.get('cumulative_output_tokens', 0)
            
            card = create_dynamic_agent_card(
                agent_id=agent_id,
                name=agent.name,
                description=agent.instruction,
                servers=agent.config.servers,
                status="active",
                context_tokens_used=tokens_used
            )
            result.append(card)
        
        return result
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get a dynamic agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            agent: The agent instance or None if not found
        """
        return self.dynamic_agents.get(agent_id)
    
    async def shutdown_all(self) -> None:
        """
        Shutdown all dynamic agents and clean up resources.
        """
        agent_ids = list(self.dynamic_agents.keys())
        for agent_id in agent_ids:
            await self.terminate_agent(agent_id)
        
        self.logger.info("Shutdown all dynamic agents")
    
    def format_responses_for_aggregation(
        self, 
        responses: Dict[str, str], 
        original_message: Optional[str] = None
    ) -> str:
        """
        Format dynamic agent responses for aggregation - SAME format as ParallelAgent.
        
        Args:
            responses: Dict mapping agent_id to response
            original_message: The original message sent to agents
            
        Returns:
            formatted: Formatted string for aggregation
        """
        formatted = []
        
        # Include the original message if provided
        if original_message:
            formatted.append("The following request was sent to the dynamic agents:")
            formatted.append(f"<fastagent:request>\n{original_message}\n</fastagent:request>")
        
        # Format each agent's response - SAME format as ParallelAgent
        for agent_id, response in responses.items():
            agent = self.dynamic_agents.get(agent_id)
            agent_name = agent.name if agent else agent_id
            formatted.append(
                f'<fastagent:response agent="{agent_name}">\n{response}\n</fastagent:response>'
            )
        
        return "\n\n".join(formatted)