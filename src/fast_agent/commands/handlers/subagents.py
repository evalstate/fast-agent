"""Shared handlers for subagent runtime controls and persisted runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.agents.subagent_tool import (
    set_subagent_tool_enabled,
    subagent_tool_enabled,
)
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.command_actions.accessors import lookup_agent
from fast_agent.commands.results import CommandOutcome
from fast_agent.session import subagent_run_from_session
from fast_agent.utils.markdown import escape_markdown_table_cell

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.session import Session

_USAGE = "Usage: /subagents [list|status|on|off|toggle|help]"


def _active_root_session(ctx: "CommandContext") -> "Session | None":
    if not ctx.sessions_enabled:
        return None
    manager = ctx.resolve_session_manager()
    session = manager.current_session
    if ctx.acp_session_id is not None and (
        session is None or session.info.name != ctx.acp_session_id
    ):
        return manager.get_session(ctx.acp_session_id)
    return session


def _runtime_agent(ctx: "CommandContext", agent_name: str) -> ToolAgent | None:
    agent = lookup_agent(ctx.agent_provider, agent_name)
    return agent if isinstance(agent, ToolAgent) else None


def _status_outcome(ctx: "CommandContext", *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = _runtime_agent(ctx, agent_name)
    if agent is None:
        outcome.add_message(
            f"Subagents are unavailable for agent '{agent_name}'.",
            channel="warning",
            right_info="subagents",
        )
        return outcome

    state = "enabled" if subagent_tool_enabled(agent) else "disabled"
    source = agent.config.subagent_activation_source or "configuration"
    model = agent.config.subagent_model or "inherit parent model"
    outcome.add_message(
        f"Subagents: {state}\nActivation: {source}\nModel: {model}",
        right_info="subagents",
    )
    return outcome


def _list_outcome(ctx: "CommandContext", *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    parent = _active_root_session(ctx)
    if parent is None:
        outcome.add_message(
            "No active persisted session; no subagent runs to list.",
            channel="warning",
            right_info="subagents",
        )
        return outcome

    manager = ctx.resolve_session_manager()
    runs = [
        run
        for child in manager.list_child_sessions(parent)
        if (run := subagent_run_from_session(parent, child)) is not None
    ]
    runs.sort(key=lambda run: (run.ordinal == 0, run.ordinal))
    if not runs:
        state = (
            "enabled"
            if (agent := _runtime_agent(ctx, agent_name)) is not None
            and subagent_tool_enabled(agent)
            else "disabled"
        )
        outcome.add_message(
            f"No subagent runs in this session. Tool: {state}.",
            right_info="subagents",
        )
        return outcome

    lines = [
        "| Alias | Status | Agent | Task |",
        "| --- | --- | --- | --- |",
    ]
    for run in runs:
        task = run.task_preview or run.label or ""
        lines.append(
            "| "
            f"{escape_markdown_table_cell(run.alias)} | "
            f"{escape_markdown_table_cell(run.status)} | "
            f"{escape_markdown_table_cell(run.parent_agent_name)} | "
            f"{escape_markdown_table_cell(task)} |"
        )
    outcome.add_message(
        "\n".join(lines),
        right_info="subagents",
        render_markdown=True,
    )
    return outcome


def _toggle_outcome(
    ctx: "CommandContext",
    *,
    agent_name: str,
    enabled: bool,
) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = _runtime_agent(ctx, agent_name)
    if agent is None or agent.config.tool_only or agent.config.subagent_child:
        outcome.add_message(
            f"Subagent runtime controls are unavailable for agent '{agent_name}'.",
            channel="warning",
            right_info="subagents",
        )
        return outcome

    if not set_subagent_tool_enabled(agent, enabled):
        source = agent.config.subagent_activation_source
        reason = (
            f" Subagents are disabled by {source}."
            if enabled and source in {"configuration", "cli"}
            else ""
        )
        outcome.add_message(
            f"Failed to {'enable' if enabled else 'disable'} subagents for '{agent_name}'.{reason}",
            channel="error",
            right_info="subagents",
        )
        return outcome

    source = agent.config.subagent_activation_source or "configuration"
    outcome.add_message(
        f"Subagents {'enabled' if enabled else 'disabled'} for '{agent_name}' ({source}).",
        right_info="subagents",
    )
    return outcome


async def handle_subagents_command(
    ctx: "CommandContext",
    *,
    agent_name: str,
    action: str,
) -> CommandOutcome:
    if action == "list":
        return _list_outcome(ctx, agent_name=agent_name)
    if action == "status":
        return _status_outcome(ctx, agent_name=agent_name)
    if action == "help":
        outcome = CommandOutcome()
        outcome.add_message(_USAGE, right_info="subagents")
        return outcome
    if action not in {"on", "off", "toggle"}:
        outcome = CommandOutcome()
        outcome.add_message(_USAGE, channel="error", right_info="subagents")
        return outcome

    agent = _runtime_agent(ctx, agent_name)
    if action == "toggle":
        return _toggle_outcome(
            ctx,
            agent_name=agent_name,
            enabled=not subagent_tool_enabled(agent),
        )
    return _toggle_outcome(ctx, agent_name=agent_name, enabled=action == "on")
