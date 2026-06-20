"""Plugin slash-command actions."""

from fast_agent.command_actions.config import parse_plugin_command_action_specs
from fast_agent.command_actions.models import (
    FAST_AGENT_AUDIT_CHANNEL,
    PluginCommandAction,
    PluginCommandActionContext,
    PluginCommandActionFunction,
    PluginCommandActionImage,
    PluginCommandActionResult,
    PluginCommandActionSpec,
    PluginCommandCompletion,
    PluginCommandCompletionContext,
    PluginCommandCompletionFunction,
)
from fast_agent.command_actions.registry import (
    PluginCommandActionRegistry,
    normalize_plugin_command_action_result,
)
from fast_agent.command_actions.runtime import PluginRuntime, PluginRuntimeFacade

__all__ = [
    "FAST_AGENT_AUDIT_CHANNEL",
    "PluginCommandAction",
    "PluginCommandActionContext",
    "PluginCommandActionFunction",
    "PluginCommandActionImage",
    "PluginCommandActionRegistry",
    "PluginCommandActionResult",
    "PluginCommandActionSpec",
    "PluginCommandCompletion",
    "PluginCommandCompletionContext",
    "PluginCommandCompletionFunction",
    "PluginRuntime",
    "PluginRuntimeFacade",
    "normalize_plugin_command_action_result",
    "parse_plugin_command_action_specs",
]
