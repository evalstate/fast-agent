"""fast-agent - An MCP native agent application framework"""

# Import important MCP types for re-export (safe - external)
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

# Configuration and settings (safe - pure Pydantic models)
from fast_agent.config import (
    AnthropicSettings,
    AzureSettings,
    BedrockSettings,
    DeepSeekSettings,
    GenericSettings,
    GoogleSettings,
    GroqSettings,
    HuggingFaceSettings,
    LoggerSettings,
    MCPElicitationSettings,
    MCPRootSettings,
    MCPSamplingSettings,
    MCPServerAuthSettings,
    MCPServerSettings,
    MCPSettings,
    OpenAISettings,
    OpenRouterSettings,
    OpenTelemetrySettings,
    Settings,
    TensorZeroSettings,
    XAISettings,
)

# Type definitions and enums (safe - no dependencies)
from fast_agent.types import LlmStopReason


def __getattr__(name: str):
    """Lazy import heavy modules to avoid circular imports during package initialization."""
    if name == "Core":
        from fast_agent.core import Core

        return Core
    elif name == "Context":
        from fast_agent.context import Context

        return Context
    elif name == "ContextDependent":
        from fast_agent.context_dependent import ContextDependent

        return ContextDependent
    elif name == "ServerRegistry":
        from fast_agent.mcp_server_registry import ServerRegistry

        return ServerRegistry
    elif name == "ProgressAction":
        from fast_agent.event_progress import ProgressAction

        return ProgressAction
    elif name == "ProgressEvent":
        from fast_agent.event_progress import ProgressEvent

        return ProgressEvent
    elif name == "ToolAgentSynchronous":
        from fast_agent.agents.tool_agent import ToolAgent

        return ToolAgent
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Core fast-agent components (lazy loaded)
    "Core",
    "Context",
    "ContextDependent",
    "ServerRegistry",
    # Configuration and settings (eagerly loaded)
    "Settings",
    "MCPSettings",
    "MCPServerSettings",
    "MCPServerAuthSettings",
    "MCPSamplingSettings",
    "MCPElicitationSettings",
    "MCPRootSettings",
    "AnthropicSettings",
    "OpenAISettings",
    "DeepSeekSettings",
    "GoogleSettings",
    "XAISettings",
    "GenericSettings",
    "OpenRouterSettings",
    "AzureSettings",
    "GroqSettings",
    "OpenTelemetrySettings",
    "TensorZeroSettings",
    "BedrockSettings",
    "HuggingFaceSettings",
    "LoggerSettings",
    # Progress and event tracking (lazy loaded)
    "ProgressAction",
    "ProgressEvent",
    # Type definitions and enums (eagerly loaded)
    "LlmStopReason",
    # Agents (lazy loaded)
    "ToolAgentSynchronous",
    # MCP types (re-exported for convenience, eagerly loaded)
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
]
