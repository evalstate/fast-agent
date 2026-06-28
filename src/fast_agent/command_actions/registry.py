"""Registry and execution helpers for plugin command actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fast_agent.command_actions.config import normalize_plugin_command_name
from fast_agent.command_actions.loader import load_plugin_command_action_function
from fast_agent.command_actions.models import (
    PluginCommandAction,
    PluginCommandActionContext,
    PluginCommandActionResult,
    PluginCommandActionSpec,
)

if TYPE_CHECKING:
    from pathlib import Path


def normalize_plugin_command_action_result(
    value: PluginCommandActionResult | str | None,
) -> PluginCommandActionResult:
    if value is None:
        return PluginCommandActionResult()
    if isinstance(value, str):
        return PluginCommandActionResult(message=value)
    return value


@dataclass(slots=True)
class PluginCommandActionRegistry:
    """Loaded command actions for one agent/app scope."""

    _actions: dict[str, PluginCommandAction] = field(default_factory=dict)

    @classmethod
    def from_specs(
        cls,
        specs: dict[str, PluginCommandActionSpec] | None,
        *,
        base_path: Path | None = None,
    ) -> "PluginCommandActionRegistry":
        registry = cls()
        for spec in (specs or {}).values():
            handler = load_plugin_command_action_function(spec.handler, base_path=base_path)
            registry.register(PluginCommandAction(spec=spec, handler=handler))
        return registry

    def register(self, action: PluginCommandAction) -> None:
        self._actions[normalize_plugin_command_name(action.spec.name)] = action

    def resolve(self, name: str) -> PluginCommandAction | None:
        return self._actions.get(normalize_plugin_command_name(name))

    async def execute(
        self,
        name: str,
        ctx: PluginCommandActionContext,
    ) -> PluginCommandActionResult | None:
        action = self.resolve(name)
        if action is None:
            return None
        return normalize_plugin_command_action_result(await action.handler(ctx))
