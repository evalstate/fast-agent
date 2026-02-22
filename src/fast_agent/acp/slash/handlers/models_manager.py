"""Model onboarding slash command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import models_manager as models_handlers

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_models(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    tokens = (arguments or "").strip().split(maxsplit=1)
    action = tokens[0].lower() if tokens else "doctor"
    remainder = tokens[1] if len(tokens) > 1 else ""

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    try:
        outcome = await models_handlers.handle_models_command(
            ctx,
            agent_name=handler.current_agent_name,
            action=action,
            argument=remainder or None,
        )
    except Exception as exc:  # noqa: BLE001
        return f"# models\n\nFailed to execute /models: {exc}"

    heading = "models" if action in {"", "doctor", "list"} else f"models {action}"
    return handler._format_outcome_as_markdown(outcome, heading, io=io)
