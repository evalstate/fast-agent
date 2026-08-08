"""Shared command infrastructure for TUI and ACP adapters."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "CommandContext",
    "CommandHandler",
    "CommandIO",
    "CommandMessage",
    "CommandOutcome",
    "CommandRegistry",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CommandContext": ("fast_agent.commands.context", "CommandContext"),
    "CommandIO": ("fast_agent.commands.context", "CommandIO"),
    "CommandHandler": ("fast_agent.commands.registry", "CommandHandler"),
    "CommandRegistry": ("fast_agent.commands.registry", "CommandRegistry"),
    "CommandMessage": ("fast_agent.commands.results", "CommandMessage"),
    "CommandOutcome": ("fast_agent.commands.results", "CommandOutcome"),
}

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext as CommandContext
    from fast_agent.commands.context import CommandIO as CommandIO
    from fast_agent.commands.registry import CommandHandler as CommandHandler
    from fast_agent.commands.registry import CommandRegistry as CommandRegistry
    from fast_agent.commands.results import CommandMessage as CommandMessage
    from fast_agent.commands.results import CommandOutcome as CommandOutcome


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
