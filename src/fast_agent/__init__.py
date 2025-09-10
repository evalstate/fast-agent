"""fast-agent - An MCP native agent application framework"""

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
from fast_agent.types import LlmStopReason, RequestParams


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
    elif name == "LlmAgent":
        from fast_agent.agents.llm_agent import LlmAgent

        return LlmAgent
    elif name == "LlmDecorator":
        from fast_agent.agents.llm_decorator import LlmDecorator

        return LlmDecorator
    elif name == "ToolAgent":
        from fast_agent.agents.tool_agent import ToolAgent

        return ToolAgent
    elif name == "McpAgent":
        from fast_agent.agents.mcp_agent import McpAgent

        return McpAgent
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
    "RequestParams",
    # Agents (lazy loaded)
    "ToolAgentSynchronous",
    "LlmAgent",
    "LlmDecorator",
    "ToolAgent",
    "McpAgent",
]
