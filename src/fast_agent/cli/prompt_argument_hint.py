"""Explain that fast-agent does not take a prompt as a positional argument.

Coding agents such as Claude Code accept ``claude "task"``, and terminal managers
(``hc run --bg fast-agent "task"``) assume that convention. fast-agent instead
reports a confusing "No such command" error; replace it with what to do.
"""

from __future__ import annotations

from typing import Any

from typer._click.exceptions import UsageError
from typer.core import TyperGroup


def prompt_argument_message(argument: str) -> str:
    preview = argument if len(argument) <= 60 else argument[:57] + "..."
    return (
        f"fast-agent does not take a prompt as an argument: {preview!r}.\n"
        "Start fast-agent and send the message once it is ready (for example "
        "`hc wait <t>` then `hc send <t> ...`, or an hc template), or run it once "
        "with --message:\n"
        f"  fast-agent go --message {argument!r}"
    )


def looks_like_prompt(argument: str) -> bool:
    return any(character.isspace() for character in argument.strip())


class PromptArgumentHintGroup(TyperGroup):
    """A Typer group whose unknown-command error recognises a prompt argument."""

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        try:
            return super().resolve_command(ctx, args)
        except UsageError as exc:
            if args and looks_like_prompt(args[0]):
                raise UsageError(prompt_argument_message(args[0]), ctx) from exc
            raise
