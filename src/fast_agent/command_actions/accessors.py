"""Optional capability accessors for plugin command actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.command_actions.models import PluginCommandActionSpec


@runtime_checkable
class _PluginCommandConfig(Protocol):
    commands: dict[str, PluginCommandActionSpec] | None


@runtime_checkable
class _PluginCommandAgent(Protocol):
    config: _PluginCommandConfig


@runtime_checkable
class _PluginCommandProvider(Protocol):
    plugin_commands: dict[str, PluginCommandActionSpec] | None
    plugin_command_base_path: Path | None


@runtime_checkable
class _AgentLookupProvider(Protocol):
    def get_agent(self, name: str) -> object | None: ...


@runtime_checkable
class _PrivateAgentLookupProvider(Protocol):
    def _agent(self, name: str) -> object | None: ...


def plugin_commands_for_agent(agent: object | None) -> dict[str, PluginCommandActionSpec] | None:
    if not isinstance(agent, _PluginCommandAgent):
        return None
    config = agent.config
    if not isinstance(config, _PluginCommandConfig):
        return None
    return config.commands


def plugin_commands_for_provider(
    provider: object | None,
) -> dict[str, PluginCommandActionSpec] | None:
    if not isinstance(provider, _PluginCommandProvider):
        return None
    return provider.plugin_commands


def plugin_command_base_path_for_provider(provider: object | None) -> Path | None:
    if not isinstance(provider, _PluginCommandProvider):
        return None
    return provider.plugin_command_base_path


def lookup_agent(provider: object | None, name: str) -> object | None:
    if isinstance(provider, _AgentLookupProvider):
        return provider.get_agent(name)
    if isinstance(provider, _PrivateAgentLookupProvider):
        return provider._agent(name)
    return None
