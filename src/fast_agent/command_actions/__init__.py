"""Plugin slash-command actions."""

from fast_agent.command_actions.config import parse_plugin_command_action_specs
from fast_agent.command_actions.models import (
    FAST_AGENT_AUDIT_CHANNEL,
    PluginCommandAction,
    PluginCommandActionContext,
    PluginCommandActionFunction,
    PluginCommandActionResult,
    PluginCommandActionSpec,
)
from fast_agent.command_actions.registry import (
    PluginCommandActionRegistry,
    normalize_plugin_command_action_result,
)

__all__ = [
    "FAST_AGENT_AUDIT_CHANNEL",
    "PluginCommandAction",
    "PluginCommandActionContext",
    "PluginCommandActionFunction",
    "PluginCommandActionRegistry",
    "PluginCommandActionResult",
    "PluginCommandActionSpec",
    "parse_plugin_command_action_specs",
    "normalize_plugin_command_action_result",
]
