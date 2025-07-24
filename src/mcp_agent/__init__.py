"""fast-agent - (fast-agent-mcp) An MCP native agent application framework"""

# Import important MCP types
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    Prompt,
    PromptMessage,
    ReadResourceResult,
    Role,
    TextContent,
    Tool,
)

# Core agent components
from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.core.agent_app import AgentApp

# Workflow decorators
from mcp_agent.core.direct_decorators import (
    agent,
    chain,
    evaluator_optimizer,
    orchestrator,
    parallel,
    router,
)

# FastAgent components
from mcp_agent.core.fastagent import FastAgent

# MCP content creation utilities
from mcp_agent.core.mcp_content import (
    Assistant,
    MCPFile,
    MCPImage,
    MCPPrompt,
    MCPText,
    User,
    create_message,
)

# Request configuration
from mcp_agent.core.request_params import RequestParams

# MCP content helpers
from mcp_agent.mcp.helpers import (
    get_image_data,
    get_resource_text,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
)

# Core protocol interfaces
from mcp_agent.mcp.interfaces import AgentProtocol, AugmentedLLMProtocol
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

__all__ = [
    # MCP types
    "Prompt",
    "Tool",
    "CallToolResult",
    "TextContent",
    "ImageContent",
    "PromptMessage",
    "GetPromptResult",
    "ReadResourceResult",
    "EmbeddedResource",
    "Role",
    # Core protocols
    "AgentProtocol",
    "AugmentedLLMProtocol",
    # Core agent components
    "Agent",
    "AgentConfig",
    "MCPAggregator",
    "PromptMessageMultipart",
    # FastAgent components
    "FastAgent",
    "AgentApp",
    # Workflow decorators
    "agent",
    "orchestrator",
    "router",
    "chain",
    "parallel",
    "evaluator_optimizer",
    # Request configuration
    "RequestParams",
    # MCP content helpers
    "get_text",
    "get_image_data",
    "get_resource_uri",
    "is_text_content",
    "is_image_content",
    "is_resource_content",
    "is_resource_link",
    "get_resource_text",
    # MCP content creation utilities
    "MCPText",
    "MCPImage",
    "MCPFile",
    "MCPPrompt",
    "User",
    "Assistant",
    "create_message",
]
