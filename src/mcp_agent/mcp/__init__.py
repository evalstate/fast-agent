"""
MCP (Model Context Protocol) integration components.
"""

from mcp.types import PromptMessage

from .helpers import (
    get_image_data,
    get_resource_text,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
)
from .interfaces import (
    AgentProtocol,
    AugmentedLLMProtocol,
    MCPConnectionManagerProtocol,
    ModelFactoryClassProtocol,
    ModelT,
    ServerConnection,
    ServerRegistryProtocol,
)
from .prompt_message_multipart import PromptMessageMultipart

__all__ = [
    # Types from mcp.types
    "PromptMessage",
    # Multipart message handling
    "PromptMessageMultipart",
    # Protocol interfaces
    "AugmentedLLMProtocol",
    "AgentProtocol",
    "MCPConnectionManagerProtocol",
    "ServerRegistryProtocol",
    "ServerConnection",
    "ModelFactoryClassProtocol",
    "ModelT",
    # Helper functions
    "get_text",
    "get_image_data",
    "get_resource_uri",
    "is_text_content",
    "is_image_content",
    "is_resource_content",
    "is_resource_link",
    "get_resource_text",
]
