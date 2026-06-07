"""fast-agent - An MCP native agent application framework"""

import importlib
from typing import TYPE_CHECKING

_CONFIG_EXPORTS = {
    "AnthropicSettings",
    "AzureSettings",
    "BedrockSettings",
    "DeepSeekSettings",
    "GenericSettings",
    "GoogleSettings",
    "GroqSettings",
    "HuggingFaceSettings",
    "LoggerSettings",
    "MCPElicitationSettings",
    "MCPRootSettings",
    "MCPSamplingSettings",
    "MCPServerAuthSettings",
    "MCPServerSettings",
    "MCPSettings",
    "OpenAISettings",
    "OpenRouterSettings",
    "OpenTelemetrySettings",
    "Settings",
    "SkillsSettings",
    "TensorZeroSettings",
    "XAISettings",
}

_TYPE_EXPORTS = {
    "ConversationSummary",
    "LlmStopReason",
    "PromptMessageExtended",
    "RequestParams",
    "ResourceLink",
    "audio_link",
    "extract_first",
    "extract_last",
    "find_matches",
    "image_link",
    "resource_link",
    "search_messages",
    "text_content",
    "video_link",
}

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
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
    if name in _CONFIG_EXPORTS:
        return getattr(importlib.import_module("fast_agent.config"), name)
    if name in _TYPE_EXPORTS:
        return getattr(importlib.import_module("fast_agent.types"), name)
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    from importlib import import_module

    return getattr(import_module(module_name), attr_name)


# Help static analyzers/IDEs resolve symbols and signatures without importing at runtime.
if TYPE_CHECKING:  # pragma: no cover - typing aid only
    # Provide a concrete import path for type checkers/IDEs
    from fast_agent.core.fastagent import FastAgent as FastAgent
    from fast_agent.core.harness import AgentHarness as AgentHarness
    from fast_agent.core.harness import HarnessSession as HarnessSession
    from fast_agent.core.harness import HarnessSessions as HarnessSessions
    from fast_agent.mcp.prompt import Prompt as Prompt
    from fast_agent.types import ConversationSummary as ConversationSummary
    from fast_agent.types import PromptMessageExtended as PromptMessageExtended


__all__ = [
    "AnthropicSettings",
    "AgentHarness",
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
