"""Model slash command handlers."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, cast

from fast_agent.acp.command_io import ACPCommandIO
from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown


class _SimpleAgentProvider:
    def __init__(self, agents: dict[str, object]) -> None:
        self._agents = agents

    def _agent(self, name: str):
        return self._agents[name]

    def agent_names(self):
        return list(self._agents.keys())

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None) -> object:
        return {}


if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_model(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    remainder = (arguments or "").strip()
    value = None
    command_kind = "reasoning"
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
            elif subcmd == "reasoning":
                value = argument or None
            elif subcmd == "web_search":
                command_kind = "web_search"
                value = argument or None
            elif subcmd == "web_fetch":
                command_kind = "web_fetch"
                value = argument or None
            else:
                return handler._model_usage_text()

    io = ACPCommandIO()
    ctx = CommandContext(
        agent_provider=_SimpleAgentProvider(cast("dict[str, object]", dict(handler.instance.agents))),
        current_agent_name=handler.current_agent_name,
        io=io,
        noenv=handler._noenv,
    )
    if command_kind == "verbosity":
        outcome = await model_handlers.handle_model_verbosity(
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
    else:
        outcome = await model_handlers.handle_model_reasoning(
            ctx,
            agent_name=handler.current_agent_name,
            value=value,
        )
    return render_command_outcome_markdown(outcome, heading="model")
