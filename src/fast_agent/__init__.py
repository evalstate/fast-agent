"""fast-agent - An MCP native agent application framework"""

from typing import TYPE_CHECKING

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
    SkillsSettings,
    TensorZeroSettings,
    XAISettings,
)

# Type definitions and enums (safe - no dependencies)
from fast_agent.types import (
    ConversationSummary,
    LlmStopReason,
    PromptMessageExtended,
    RequestParams,
    ResourceLink,
    audio_link,
    extract_first,
    extract_last,
    find_matches,
    image_link,
    resource_link,
    search_messages,
    text_content,
    video_link,
)


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
        # Import directly from submodule to avoid package re-import cycles
        from fast_agent.agents.mcp_agent import McpAgent

        return McpAgent
    elif name == "FastAgent":
        # Import from the canonical implementation to avoid recursive imports
        from fast_agent.core.fastagent import FastAgent

        return FastAgent
    elif name == "Prompt":
        # Prompt helper relies on MCP types; load lazily to speed up import time.
        from fast_agent.mcp.prompt import Prompt

        return Prompt
    elif name == "load_prompt":
        # Prompt loader also depends on MCP; defer import until explicitly requested.
        from fast_agent.mcp.prompts.prompt_load import load_prompt

        return load_prompt
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Help static analyzers/IDEs resolve symbols and signatures without importing at runtime.
if TYPE_CHECKING:  # pragma: no cover - typing aid only
    # Provide a concrete import path for type checkers/IDEs
    from fast_agent.core.fastagent import FastAgent as FastAgent  # noqa: F401
    from fast_agent.mcp.prompt import Prompt as Prompt  # noqa: F401
    from fast_agent.types import ConversationSummary as ConversationSummary  # noqa: F401
    from fast_agent.types import PromptMessageExtended as PromptMessageExtended  # noqa: F401


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
    "SkillsSettings",
    # Progress and event tracking (lazy loaded)
    "ProgressAction",
    "ProgressEvent",
    # Type definitions and enums (eagerly loaded)
    "LlmStopReason",
    "RequestParams",
    "PromptMessageExtended",
    "ResourceLink",
    "ConversationSummary",
    # Content helpers (eagerly loaded)
    "text_content",
    "resource_link",
    "image_link",
    "video_link",
    "audio_link",
    # Search utilities (eagerly loaded)
    "search_messages",
    "find_matches",
    "extract_first",
    "extract_last",
    # Prompt helpers (eagerly loaded)
    "Prompt",
    "load_prompt",
    # Agents (lazy loaded)
    "LlmAgent",
    "LlmDecorator",
    "ToolAgent",
    "McpAgent",
    "FastAgent",
]
