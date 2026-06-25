"""Model slash command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.command_discovery import render_direct_command_help
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.handlers import models_manager as models_manager_handlers
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.shared_command_intents import parse_model_command_intent

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_model(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    return await _handle_model_like(handler, arguments, heading_prefix="model")


async def _handle_model_like(
    handler: "SlashCommandHandler",
    arguments: str | None,
    *,
    heading_prefix: str,
) -> str:
    direct_help = render_direct_command_help(heading_prefix, arguments)
    if direct_help is not None:
        return direct_help

    default_action = "reasoning" if heading_prefix == "model" else "doctor"
    intent = parse_model_command_intent(arguments, default_action=default_action)
    if intent.error is not None:
        return f"Invalid /{heading_prefix} arguments: {intent.error}"
    if intent.action == "unknown":
        return handler._model_usage_text()

    ctx = handler._build_command_context()
    if intent.action == "doctor":
        return models_manager_handlers.render_models_doctor_markdown(ctx)

    if models_manager_handlers.is_model_manager_action(intent.action):
        outcome = await models_manager_handlers.handle_models_command(
            ctx,
            agent_name=handler.current_agent_name,
            action=intent.action,
            argument=intent.argument,
        )
    else:
        model_action_handler = model_handlers.get_model_action_handler(intent.action)
        if model_action_handler is None:
            return handler._model_usage_text()
        outcome = await model_action_handler(
            ctx,
            agent_name=handler.current_agent_name,
            value=intent.argument,
        )

    heading = (
        heading_prefix
        if intent.action == "reasoning" and intent.argument is None
        else f"{heading_prefix}.{intent.action}"
    )
    return render_command_outcome_markdown(outcome, heading=heading)
