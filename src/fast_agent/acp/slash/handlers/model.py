"""Model slash command handlers."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, cast

from fast_agent.acp.command_io import ACPCommandIO
from fast_agent.commands.context import CommandContext, StaticAgentProvider
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.handlers import models_manager as models_manager_handlers
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown

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
    remainder = (arguments or "").strip()
    value = None
    command_kind = "reasoning" if heading_prefix == "model" else "doctor"
    if remainder:
        try:
            tokens = shlex.split(remainder)
        except ValueError:
            tokens = remainder.split(maxsplit=1)

        if tokens:
            subcmd = tokens[0].lower()
            argument = remainder[len(tokens[0]) :].strip()
            if subcmd == "verbosity":
                command_kind = "verbosity"
                value = argument or None
            elif subcmd == "task_budget":
                command_kind = "task_budget"
                value = argument or None
            elif subcmd == "fast":
                command_kind = "fast"
                value = argument or None
            elif subcmd == "reasoning":
                value = argument or None
            elif subcmd == "web_search":
                command_kind = "web_search"
                value = argument or None
            elif subcmd == "web_fetch":
                command_kind = "web_fetch"
                value = argument or None
            elif subcmd == "switch":
                command_kind = "switch"
                value = argument or None
            elif subcmd in {"doctor", "references", "catalog", "help"}:
                command_kind = subcmd
                value = argument or None
            else:
                return handler._model_usage_text()

    io = ACPCommandIO()
    ctx = CommandContext(
        agent_provider=StaticAgentProvider(
            cast("dict[str, object]", dict(handler.instance.agents))
        ),
        current_agent_name=handler.current_agent_name,
        io=io,
        noenv=handler._noenv,
    )
    if command_kind == "doctor":
        return models_manager_handlers.render_models_doctor_markdown(ctx)
    elif command_kind == "references":
        outcome = await models_manager_handlers.handle_models_command(
            ctx,
            agent_name=handler.current_agent_name,
            action="references",
            argument=value,
        )
    elif command_kind == "catalog":
        outcome = await models_manager_handlers.handle_models_command(
            ctx,
            agent_name=handler.current_agent_name,
            action="catalog",
            argument=value,
        )
    elif command_kind == "help":
        outcome = await models_manager_handlers.handle_models_command(
            ctx,
            agent_name=handler.current_agent_name,
            action="help",
            argument=value,
        )
    elif command_kind == "verbosity":
        outcome = await model_handlers.handle_model_verbosity(
            ctx,
            agent_name=handler.current_agent_name,
            value=value,
        )
    elif command_kind == "task_budget":
        outcome = await model_handlers.handle_model_task_budget(
            ctx,
            agent_name=handler.current_agent_name,
            value=value,
        )
    elif command_kind == "fast":
        outcome = await model_handlers.handle_model_fast(
            ctx,
            agent_name=handler.current_agent_name,
            value=value,
        )
    elif command_kind == "web_search":
        outcome = await model_handlers.handle_model_web_search(
            ctx,
            agent_name=handler.current_agent_name,
            value=value,
        )
    elif command_kind == "web_fetch":
        outcome = await model_handlers.handle_model_web_fetch(
            ctx,
            agent_name=handler.current_agent_name,
            value=value,
        )
    elif command_kind == "switch":
        outcome = await model_handlers.handle_model_switch(
            ctx,
            agent_name=handler.current_agent_name,
            value=value,
        )
    else:
        outcome = await model_handlers.handle_model_reasoning(
            ctx,
            agent_name=handler.current_agent_name,
            value=value,
        )

    if outcome.reset_session:
        if not handler._noenv:
            outcome.add_message(
                "Model switch starts a new session to avoid mixing histories.",
                channel="info",
            )
            session_outcome = await sessions_handlers.handle_create_session(
                ctx,
                session_name=None,
            )
            outcome.messages.extend(session_outcome.messages)
        else:
            outcome.add_message(
                "Model switch cleared in-memory history (--noenv disables session persistence).",
                channel="info",
            )
        cleared = clear_agent_histories(handler.instance.agents, handler._logger)
        if cleared:
            outcome.add_message(
                f"Cleared agent history: {', '.join(sorted(cleared))}",
                channel="info",
            )
    heading = (
        heading_prefix
        if command_kind == "reasoning" and value is None
        else f"{heading_prefix}.{command_kind}"
    )
    return render_command_outcome_markdown(outcome, heading=heading)
