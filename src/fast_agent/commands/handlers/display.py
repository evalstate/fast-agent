"""Shared display command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich import print as rich_print

from fast_agent.commands.results import CommandOutcome
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.ui.usage_display import collect_agents_from_provider

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.core.agent_app import AgentApp


async def handle_show_usage(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    agents_to_show = collect_agents_from_provider(ctx.agent_provider, agent_name)
    if not agents_to_show:
        outcome.add_message("No usage data available", channel="warning", right_info="usage")
        return outcome

    await ctx.io.display_usage_report(agents_to_show)
    return outcome


async def handle_show_system(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = ctx.agent_provider._agent(agent_name)
    system_prompt = getattr(agent, "instruction", None)
    if not system_prompt:
        outcome.add_message("No system prompt available", channel="warning", right_info="system")
        return outcome

    server_count = 0
    if isinstance(agent, McpAgentProtocol):
        server_names = agent.aggregator.server_names
        server_count = len(server_names) if server_names else 0

    await ctx.io.display_system_prompt(
        agent_name,
        system_prompt,
        server_count=server_count,
    )

    return outcome


async def handle_show_markdown(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = ctx.agent_provider._agent(agent_name)
    if not agent.llm:
        outcome.add_message("No message history available", channel="warning")
        return outcome

    message_history = agent.message_history
    if not message_history:
        outcome.add_message("No messages in history", channel="warning")
        return outcome

    last_assistant_msg = None
    for msg in reversed(message_history):
        if msg.role == "assistant":
            last_assistant_msg = msg
            break

    if not last_assistant_msg:
        outcome.add_message("No assistant messages found", channel="warning")
        return outcome

    content = last_assistant_msg.last_text()

    rich_print("\n[bold blue]Last Assistant Response (Plain Text):[/bold blue]")
    rich_print("─" * 60)
    from fast_agent.ui import console

    console.console.print(content, markup=False)
    rich_print("─" * 60)
    rich_print()

    return outcome


async def handle_show_mcp_status(
    ctx: CommandContext, *, agent_name: str
) -> CommandOutcome:
    outcome = CommandOutcome()
    from fast_agent.ui.enhanced_prompt import show_mcp_status

    await show_mcp_status(agent_name, cast("AgentApp", ctx.agent_provider))
    return outcome
