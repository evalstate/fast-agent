"""Mid-turn concurrent execution for read-only slash commands.

Shell commands and other deferred input stay queued in ``TurnInputReader`` and
follow the normal prompt path when the active turn completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape as escape_markup

from fast_agent.ui.command_payloads import (
    A2ACommand,
    CheckCommand,
    CommandPayload,
    CommandsCommand,
    EnvironmentCommand,
    EventsCommand,
    HistoryViewCommand,
    ListPromptsCommand,
    ListSessionsCommand,
    ListSkillsCommand,
    ListToolsCommand,
    McpListCommand,
    ModelStatusCommand,
    ProcessCommand,
    ShowMarkdownCommand,
    ShowMcpStatusCommand,
    ShowSystemCommand,
    ShowUsageCommand,
    SkillsCommand,
    SubagentsCommand,
    WarmCommand,
    is_command_payload,
)
from fast_agent.ui.console import rich_print
from fast_agent.ui.prompt.parser import parse_special_input

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.session.session_manager import SessionManager
    from fast_agent.ui.interactive_prompt import InteractivePrompt

_CONCURRENT_SAFE_STRINGS = frozenset({"HELP", "HELP:STATUS"})


def is_concurrent_safe_command(parsed: str | CommandPayload) -> bool:
    """Return True if ``parsed`` is safe to execute concurrently during an active turn."""
    if isinstance(parsed, str):
        return parsed in _CONCURRENT_SAFE_STRINGS

    if isinstance(
        parsed,
        (
            ShowUsageCommand,
            ShowSystemCommand,
            ShowMarkdownCommand,
            ShowMcpStatusCommand,
            McpListCommand,
            EnvironmentCommand,
            CommandsCommand,
            CheckCommand,
            ModelStatusCommand,
            ListPromptsCommand,
            ListToolsCommand,
            ListSkillsCommand,
            ListSessionsCommand,
            HistoryViewCommand,
        ),
    ):
        return True

    if isinstance(parsed, SkillsCommand):
        return parsed.action == "list"

    if isinstance(parsed, ProcessCommand):
        return parsed.attach_process_id is None and parsed.terminate_process_id is None

    if isinstance(parsed, WarmCommand):
        return parsed.action == "status"

    if isinstance(parsed, EventsCommand):
        arg = (parsed.argument or "").strip().lower()
        return arg in ("", "status", "list")

    if isinstance(parsed, A2ACommand):
        return parsed.action in ("status", "tasks") and parsed.error is None

    if isinstance(parsed, SubagentsCommand):
        return parsed.action in ("", "list", "status", "help") and parsed.error is None

    return False


def parse_concurrent_command(text: str) -> str | CommandPayload | None:
    """Parse ``text`` and return it if it can run immediately while a turn is active."""
    if not text.lstrip().startswith("/"):
        return None
    parsed = parse_special_input(text)
    if is_concurrent_safe_command(parsed):
        return parsed
    return None


async def execute_concurrent_command(
    parsed: str | CommandPayload,
    *,
    raw_text: str,
    prompt_prefix: str,
    interactive_prompt: InteractivePrompt,
    prompt_provider: AgentApp,
    agent_name: str,
    available_agents: list[str],
    session_manager: SessionManager | None = None,
    shell_working_dir: Path | None = None,
) -> None:
    """Execute a concurrent-safe command and print its output above the live dock."""
    first_line = raw_text.lstrip().splitlines()[0]
    rich_print(f"[dim]{escape_markup(prompt_prefix)} {escape_markup(first_line)}[/dim]")

    from fast_agent.ui.prompt.special_commands import handle_special_commands

    command_result = handle_special_commands(parsed, prompt_provider)
    if not is_command_payload(command_result):
        return

    from fast_agent.ui.interactive.command_dispatch import dispatch_command_payload

    await dispatch_command_payload(
        interactive_prompt,
        command_result,
        prompt_provider=prompt_provider,
        agent=agent_name,
        available_agents=available_agents,
        available_agents_set=set(available_agents),
        merge_pinned_agents=lambda names: list(names),
        buffer_prefill="",
        shell_working_dir=shell_working_dir,
        session_manager=session_manager,
    )
