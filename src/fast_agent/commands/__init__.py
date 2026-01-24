"""Shared command infrastructure for TUI and ACP adapters."""

from __future__ import annotations

__all__ = [
    "CommandContext",
    "CommandIO",
    "CommandMessage",
    "CommandOutcome",
    "CommandHandler",
    "CommandRegistry",
]

from fast_agent.commands.context import CommandContext, CommandIO
from fast_agent.commands.registry import CommandHandler, CommandRegistry
from fast_agent.commands.results import CommandMessage, CommandOutcome
