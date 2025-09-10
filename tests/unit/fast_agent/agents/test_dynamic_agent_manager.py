"""
Unit tests for DynamicAgentManager.

These tests verify the core functionality of dynamic agent creation,
lifecycle management, and communication.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_agent.agents.dynamic_agent_manager import (
    DynamicAgentInfo,
    DynamicAgentManager,
    DynamicAgentSpec,
)
from fast_agent.agents.agent_types import AgentConfig


class TestDynamicAgentSpec:
    """Test the DynamicAgentSpec dataclass."""
    
    def test_spec_creation(self):
        """Test creating a DynamicAgentSpec."""
        spec = DynamicAgentSpec(
            name="test_agent",
            instruction="You are a test agent",
            servers=["filesystem"]
        )
        
        assert spec.name == "test_agent"
        assert spec.instruction == "You are a test agent"
        assert spec.servers == ["filesystem"]
        assert spec.tools is None
        assert spec.model is None
    
    def test_spec_with_optional_fields(self):
        """Test creating a DynamicAgentSpec with optional fields."""
        spec = DynamicAgentSpec(
            name="test_agent",
            instruction="You are a test agent",
            servers=["filesystem", "fetch"],
            tools={"filesystem": ["read*", "write*"]},
            model="haiku"
        )
        
        assert spec.tools == {"filesystem": ["read*", "write*"]}
        assert spec.model == "haiku"


class TestDynamicAgentInfo:
    """Test the DynamicAgentInfo model."""
    
    def test_info_creation(self):
        """Test creating DynamicAgentInfo."""
        info = DynamicAgentInfo(
            agent_id="test_123",
            name="test_agent",
            status="active",
            servers=["filesystem"]
        )
        
        assert info.agent_id == "test_123"
        assert info.name == "test_agent"
        assert info.status == "active"
        assert info.servers == ["filesystem"]
        assert info.context_tokens_used == 0
        assert info.last_activity is None


class TestDynamicAgentManager:
    """Test the DynamicAgentManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock parent agent
        self.mock_parent = Mock()
        self.mock_parent.config = AgentConfig(
            name="parent_agent",
            max_dynamic_agents=3,
            model="haiku"
        )
        self.mock_parent.server_names = ["filesystem", "fetch"]
        self.mock_parent._context = Mock()
        self.mock_parent.name = "parent_agent"
        
        # Create the manager
        self.manager = DynamicAgentManager(self.mock_parent)
    
    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager.parent_agent == self.mock_parent
        assert self.manager.max_agents == 3
        assert len(self.manager.dynamic_agents) == 0
    
    @pytest.mark.asyncio
    async def test_create_agent_basic(self):
        """Test basic agent creation."""
        spec = DynamicAgentSpec(
            name="test_agent",
            instruction="You are a test agent",
            servers=["filesystem"]
        )
        
        with patch('mcp_agent.agents.dynamic_agent_manager.Agent') as mock_agent_class, \
             patch('mcp_agent.agents.dynamic_agent_manager.get_model_factory') as mock_factory:
            
            # Setup mocks
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_factory.return_value = Mock()
            
            # Create agent
            agent_id = await self.manager.create_agent(spec)
            
            # Verify
            assert agent_id.startswith("test_agent_")
            assert len(agent_id) == len("test_agent_") + 6  # name + underscore + 6 hex chars
            assert agent_id in self.manager.dynamic_agents
            
            # Verify agent was created with correct config
            mock_agent_class.assert_called_once()
            config_arg = mock_agent_class.call_args[1]['config']
            assert config_arg.name == "test_agent"
            assert config_arg.instruction == "You are a test agent"
            assert config_arg.servers == ["filesystem"]
            
            # Verify agent was initialized and LLM attached
            mock_agent.initialize.assert_called_once()
            mock_agent.attach_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_agent_max_limit(self):
        """Test agent creation fails when max limit reached."""
        # Fill up to max capacity
        for i in range(3):
            agent_id = f"agent_{i}_abcdef"
            mock_agent = Mock()
            self.manager.dynamic_agents[agent_id] = mock_agent
        
        # Try to create one more
        spec = DynamicAgentSpec(
            name="overflow_agent",
            instruction="This should fail",
            servers=["filesystem"]
        )
        
        with pytest.raises(ValueError, match="Maximum number of dynamic agents"):
            await self.manager.create_agent(spec)
    
    @pytest.mark.asyncio
    async def test_create_agent_invalid_servers(self):
        """Test agent creation fails with invalid servers."""
        spec = DynamicAgentSpec(
            name="test_agent",
            instruction="You are a test agent",
            servers=["invalid_server"]
        )
        
        with pytest.raises(ValueError, match="Invalid servers"):
            await self.manager.create_agent(spec)
    
    @pytest.mark.asyncio
    async def test_terminate_agent(self):
        """Test agent termination."""
        # Add a mock agent
        agent_id = "test_123"
        mock_agent = AsyncMock()
        self.manager.dynamic_agents[agent_id] = mock_agent
        
        # Terminate it
        result = await self.manager.terminate_agent(agent_id)
        
        # Verify
        assert result is True
        assert agent_id not in self.manager.dynamic_agents
        mock_agent.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_terminate_nonexistent_agent(self):
        """Test terminating a non-existent agent."""
        result = await self.manager.terminate_agent("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_to_agent(self):
        """Test sending message to specific agent."""
        # Add a mock agent
        agent_id = "test_123"
        mock_agent = AsyncMock()
        mock_agent.send.return_value = "Agent response"
        self.manager.dynamic_agents[agent_id] = mock_agent
        
        # Send message
        response = await self.manager.send_to_agent(agent_id, "Test message")
        
        # Verify
        assert response == "Agent response"
        mock_agent.send.assert_called_once_with("Test message")
    
    @pytest.mark.asyncio
    async def test_send_to_nonexistent_agent(self):
        """Test sending to non-existent agent fails."""
        with pytest.raises(ValueError, match="Agent nonexistent not found"):
            await self.manager.send_to_agent("nonexistent", "Test message")
    
    @pytest.mark.asyncio
    async def test_broadcast_message_parallel(self):
        """Test broadcasting message to multiple agents in parallel."""
        # Add mock agents
        agents = {}
        for i in range(3):
            agent_id = f"agent_{i}"
            mock_agent = AsyncMock()
            mock_response = Mock()
            mock_response.all_text.return_value = f"Response from agent {i}"
            mock_agent.generate.return_value = mock_response
            mock_agent.name = f"agent_{i}"
            agents[agent_id] = mock_agent
            self.manager.dynamic_agents[agent_id] = mock_agent
        
        # Broadcast message
        responses = await self.manager.broadcast_message("Test broadcast", parallel=True)
        
        # Verify all agents received the message
        assert len(responses) == 3
        for i, (agent_id, response) in enumerate(responses.items()):
            assert agent_id == f"agent_{i}"
            assert response == f"Response from agent {i}"
            agents[agent_id].generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_message_specific_agents(self):
        """Test broadcasting to specific agents only."""
        # Add mock agents
        for i in range(3):
            agent_id = f"agent_{i}"
            mock_agent = AsyncMock()
            mock_response = Mock()
            mock_response.all_text.return_value = f"Response from agent {i}"
            mock_agent.generate.return_value = mock_response
            mock_agent.name = f"agent_{i}"
            self.manager.dynamic_agents[agent_id] = mock_agent
        
        # Broadcast to specific agents only
        target_agents = ["agent_0", "agent_2"]
        responses = await self.manager.broadcast_message(
            "Test broadcast", 
            agent_ids=target_agents,
            parallel=True
        )
        
        # Verify only targeted agents received the message
        assert len(responses) == 2
        assert "agent_0" in responses
        assert "agent_2" in responses
        assert "agent_1" not in responses
    
    @pytest.mark.asyncio
    async def test_broadcast_empty_agents(self):
        """Test broadcasting with no agents."""
        responses = await self.manager.broadcast_message("Test broadcast")
        assert responses == {}
    
    def test_list_agents(self):
        """Test listing all agents."""
        # Add mock agents
        for i in range(2):
            agent_id = f"agent_{i}"
            mock_agent = Mock()
            mock_agent.name = f"agent_{i}"
            mock_agent.config = Mock()
            mock_agent.config.servers = ["filesystem"]
            mock_agent.usage_accumulator = None
            self.manager.dynamic_agents[agent_id] = mock_agent
        
        # List agents
        agents = self.manager.list_agents()
        
        # Verify
        assert len(agents) == 2
        for i, info in enumerate(agents):
            assert isinstance(info, DynamicAgentInfo)
            assert info.agent_id == f"agent_{i}"
            assert info.name == f"agent_{i}"
            assert info.status == "active"
            assert info.servers == ["filesystem"]
            assert info.context_tokens_used == 0
    
    def test_get_agent(self):
        """Test getting agent by ID."""
        # Add mock agent
        agent_id = "test_123"
        mock_agent = Mock()
        self.manager.dynamic_agents[agent_id] = mock_agent
        
        # Get agent
        result = self.manager.get_agent(agent_id)
        assert result == mock_agent
        
        # Get non-existent agent
        result = self.manager.get_agent("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """Test shutting down all agents."""
        # Add mock agents
        mock_agents = {}
        for i in range(3):
            agent_id = f"agent_{i}"
            mock_agent = AsyncMock()
            mock_agents[agent_id] = mock_agent
            self.manager.dynamic_agents[agent_id] = mock_agent
        
        # Shutdown all
        await self.manager.shutdown_all()
        
        # Verify all agents were terminated
        assert len(self.manager.dynamic_agents) == 0
        for mock_agent in mock_agents.values():
            mock_agent.shutdown.assert_called_once()
    
    def test_format_responses_for_aggregation(self):
        """Test formatting responses like ParallelAgent."""
        # Add mock agents for names
        mock_agent_1 = Mock()
        mock_agent_1.name = "frontend_dev"
        mock_agent_2 = Mock()
        mock_agent_2.name = "backend_dev"
        self.manager.dynamic_agents["agent_1"] = mock_agent_1
        self.manager.dynamic_agents["agent_2"] = mock_agent_2
        
        responses = {
            "agent_1": "Frontend component created",
            "agent_2": "API endpoints implemented"
        }
        
        formatted = self.manager.format_responses_for_aggregation(
            responses, "Build the application"
        )
        
        # Verify format matches ParallelAgent
        assert "The following request was sent to the dynamic agents:" in formatted
        assert "<fastagent:request>" in formatted
        assert "Build the application" in formatted
        assert "</fastagent:request>" in formatted
        assert '<fastagent:response agent="frontend_dev">' in formatted
        assert "Frontend component created" in formatted
        assert '<fastagent:response agent="backend_dev">' in formatted
        assert "API endpoints implemented" in formatted
        assert "</fastagent:response>" in formatted
    
    def test_format_responses_without_original_message(self):
        """Test formatting responses without original message."""
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        self.manager.dynamic_agents["agent_1"] = mock_agent
        
        responses = {"agent_1": "Task completed"}
        formatted = self.manager.format_responses_for_aggregation(responses)
        
        # Should not include original message section
        assert "The following request was sent" not in formatted
        assert "<fastagent:request>" not in formatted
        assert '<fastagent:response agent="test_agent">' in formatted
        assert "Task completed" in formatted