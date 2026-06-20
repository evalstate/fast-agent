"""fast-agent - An MCP native agent application framework"""

from typing import TYPE_CHECKING

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AnthropicSettings": ("fast_agent.config", "AnthropicSettings"),
    "AzureSettings": ("fast_agent.config", "AzureSettings"),
    "BedrockSettings": ("fast_agent.config", "BedrockSettings"),
    "DeepSeekSettings": ("fast_agent.config", "DeepSeekSettings"),
    "GenericSettings": ("fast_agent.config", "GenericSettings"),
    "GoogleSettings": ("fast_agent.config", "GoogleSettings"),
    "GroqSettings": ("fast_agent.config", "GroqSettings"),
    "HuggingFaceSettings": ("fast_agent.config", "HuggingFaceSettings"),
    "LoggerSettings": ("fast_agent.config", "LoggerSettings"),
    "MCPElicitationSettings": ("fast_agent.config", "MCPElicitationSettings"),
    "MCPRootSettings": ("fast_agent.config", "MCPRootSettings"),
    "MCPSamplingSettings": ("fast_agent.config", "MCPSamplingSettings"),
    "MCPServerAuthSettings": ("fast_agent.config", "MCPServerAuthSettings"),
    "MCPServerSettings": ("fast_agent.config", "MCPServerSettings"),
    "MCPSettings": ("fast_agent.config", "MCPSettings"),
    "OpenAISettings": ("fast_agent.config", "OpenAISettings"),
    "OpenRouterSettings": ("fast_agent.config", "OpenRouterSettings"),
    "OpenTelemetrySettings": ("fast_agent.config", "OpenTelemetrySettings"),
    "Settings": ("fast_agent.config", "Settings"),
    "SkillsSettings": ("fast_agent.config", "SkillsSettings"),
    "TensorZeroSettings": ("fast_agent.config", "TensorZeroSettings"),
    "XAISettings": ("fast_agent.config", "XAISettings"),
    "ConversationSummary": ("fast_agent.types", "ConversationSummary"),
    "AgentAuth": ("fast_agent.types", "AgentAuth"),
    "AgentRequest": ("fast_agent.types", "AgentRequest"),
    "AgentResponse": ("fast_agent.types", "AgentResponse"),
    "LlmStopReason": ("fast_agent.types", "LlmStopReason"),
    "PromptMessageExtended": ("fast_agent.types", "PromptMessageExtended"),
    "ProgressReporter": ("fast_agent.types", "ProgressReporter"),
    "RequestParams": ("fast_agent.types", "RequestParams"),
    "ResourceLink": ("fast_agent.types", "ResourceLink"),
    "audio_link": ("fast_agent.types", "audio_link"),
    "extract_first": ("fast_agent.types", "extract_first"),
    "extract_last": ("fast_agent.types", "extract_last"),
    "find_matches": ("fast_agent.types", "find_matches"),
    "image_link": ("fast_agent.types", "image_link"),
    "resource_link": ("fast_agent.types", "resource_link"),
    "search_messages": ("fast_agent.types", "search_messages"),
    "text_content": ("fast_agent.types", "text_content"),
    "video_link": ("fast_agent.types", "video_link"),
    "Core": ("fast_agent.core", "Core"),
    "Context": ("fast_agent.context", "Context"),
    "ContextDependent": ("fast_agent.context_dependent", "ContextDependent"),
    "ServerRegistry": ("fast_agent.mcp_server_registry", "ServerRegistry"),
    "ProgressAction": ("fast_agent.event_progress", "ProgressAction"),
    "ProgressEvent": ("fast_agent.event_progress", "ProgressEvent"),
    "ToolAgentSynchronous": ("fast_agent.agents.tool_agent", "ToolAgent"),
    "LlmAgent": ("fast_agent.agents.llm_agent", "LlmAgent"),
    "LlmDecorator": ("fast_agent.agents.llm_decorator", "LlmDecorator"),
    "ToolAgent": ("fast_agent.agents.tool_agent", "ToolAgent"),
    "McpAgent": ("fast_agent.agents.mcp_agent", "McpAgent"),
    "FastAgent": ("fast_agent.core.fastagent", "FastAgent"),
    "AgentHarness": ("fast_agent.core.harness", "AgentHarness"),
    "HarnessSessions": ("fast_agent.core.harness", "HarnessSessions"),
    "HarnessSession": ("fast_agent.core.harness", "HarnessSession"),
    "Prompt": ("fast_agent.mcp.prompt", "Prompt"),
    "PromptExitError": ("fast_agent.core.exceptions", "PromptExitError"),
    "load_prompt": ("fast_agent.mcp.prompts.prompt_load", "load_prompt"),
}


def __getattr__(name: str):
    """Lazy import heavy modules to avoid circular imports during package initialization."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    from importlib import import_module

    return getattr(import_module(module_name), attr_name)


# Help static analyzers/IDEs resolve symbols and signatures without importing at runtime.
if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from fast_agent.config import (
        AnthropicSettings as AnthropicSettings,
    )
    from fast_agent.config import (
        AzureSettings as AzureSettings,
    )
    from fast_agent.config import (
        BedrockSettings as BedrockSettings,
    )
    from fast_agent.config import (
        DeepSeekSettings as DeepSeekSettings,
    )
    from fast_agent.config import (
        GenericSettings as GenericSettings,
    )
    from fast_agent.config import (
        GoogleSettings as GoogleSettings,
    )
    from fast_agent.config import (
        GroqSettings as GroqSettings,
    )
    from fast_agent.config import (
        HuggingFaceSettings as HuggingFaceSettings,
    )
    from fast_agent.config import (
        LoggerSettings as LoggerSettings,
    )
    from fast_agent.config import (
        MCPElicitationSettings as MCPElicitationSettings,
    )
    from fast_agent.config import (
        MCPRootSettings as MCPRootSettings,
    )
    from fast_agent.config import (
        MCPSamplingSettings as MCPSamplingSettings,
    )
    from fast_agent.config import (
        MCPServerAuthSettings as MCPServerAuthSettings,
    )
    from fast_agent.config import (
        MCPServerSettings as MCPServerSettings,
    )
    from fast_agent.config import (
        MCPSettings as MCPSettings,
    )
    from fast_agent.config import (
        OpenAISettings as OpenAISettings,
    )
    from fast_agent.config import (
        OpenRouterSettings as OpenRouterSettings,
    )
    from fast_agent.config import (
        OpenTelemetrySettings as OpenTelemetrySettings,
    )
    from fast_agent.config import (
        Settings as Settings,
    )
    from fast_agent.config import (
        SkillsSettings as SkillsSettings,
    )
    from fast_agent.config import (
        TensorZeroSettings as TensorZeroSettings,
    )
    from fast_agent.config import (
        XAISettings as XAISettings,
    )

    # Provide a concrete import path for type checkers/IDEs
    from fast_agent.core.fastagent import FastAgent as FastAgent
    from fast_agent.core.harness import AgentHarness as AgentHarness
    from fast_agent.core.harness import HarnessSession as HarnessSession
    from fast_agent.core.harness import HarnessSessions as HarnessSessions
    from fast_agent.mcp.prompt import Prompt as Prompt
    from fast_agent.types import AgentAuth as AgentAuth
    from fast_agent.types import AgentRequest as AgentRequest
    from fast_agent.types import AgentResponse as AgentResponse
    from fast_agent.types import ConversationSummary as ConversationSummary
    from fast_agent.types import LlmStopReason as LlmStopReason
    from fast_agent.types import ProgressReporter as ProgressReporter
    from fast_agent.types import PromptMessageExtended as PromptMessageExtended
    from fast_agent.types import RequestParams as RequestParams
    from fast_agent.types import ResourceLink as ResourceLink
    from fast_agent.types import audio_link as audio_link
    from fast_agent.types import extract_first as extract_first
    from fast_agent.types import extract_last as extract_last
    from fast_agent.types import find_matches as find_matches
    from fast_agent.types import image_link as image_link
    from fast_agent.types import resource_link as resource_link
    from fast_agent.types import search_messages as search_messages
    from fast_agent.types import text_content as text_content
    from fast_agent.types import video_link as video_link


__all__ = [
    "AnthropicSettings",
    "AgentAuth",
    "AgentHarness",
    "AgentRequest",
    "AgentResponse",
    "HarnessSessions",
    "HarnessSession",
    "AzureSettings",
    "BedrockSettings",
    "Context",
    "ContextDependent",
    "ConversationSummary",
    "Core",
    "DeepSeekSettings",
    "FastAgent",
    "GenericSettings",
    "GoogleSettings",
    "GroqSettings",
    "HuggingFaceSettings",
    "LlmAgent",
    "LlmDecorator",
    "LlmStopReason",
    "LoggerSettings",
    "MCPElicitationSettings",
    "MCPRootSettings",
    "MCPSamplingSettings",
    "MCPServerAuthSettings",
    "MCPServerSettings",
    "MCPSettings",
    "McpAgent",
    "OpenAISettings",
    "OpenRouterSettings",
    "OpenTelemetrySettings",
    "ProgressAction",
    "ProgressEvent",
    "Prompt",
    "PromptExitError",
    "PromptMessageExtended",
    "ProgressReporter",
    "RequestParams",
    "ResourceLink",
    "ServerRegistry",
    "Settings",
    "SkillsSettings",
    "TensorZeroSettings",
    "ToolAgent",
    "ToolAgentSynchronous",
    "XAISettings",
    "audio_link",
    "extract_first",
    "extract_last",
    "find_matches",
    "image_link",
    "load_prompt",
    "resource_link",
    "search_messages",
    "text_content",
    "video_link",
]
