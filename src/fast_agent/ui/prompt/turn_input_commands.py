"""Mid-turn concurrent execution for read-only slash commands and non-interactive shell commands.

Mutating commands (such as ``/clear``, ``/compact``, ``/model <name>``, ``@agent``,
``/session new``, or interactive ``!`` shells) are intentionally NOT allowed here;
they remain queued in ``TurnInputReader`` and run in FIFO order once the active
turn completes.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from rich.markup import escape as escape_markup
from rich.text import Text

from fast_agent.core.logging.logger import get_logger
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
    ShellCommand,
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

logger = get_logger(__name__)

_SHELL_OUTPUT_MAX_CHARS = 8_000
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

    if isinstance(parsed, ShellCommand):
        return not parsed.output_to_prompt and bool(parsed.command.strip())

    return False


def parse_concurrent_command(text: str) -> str | CommandPayload | None:
    """Parse ``text`` and return it if it can run immediately while a turn is active."""
    stripped = text.lstrip()
    if not stripped or not stripped.startswith(("/", "!")):
        return None
    parsed = parse_special_input(text)
    if is_concurrent_safe_command(parsed):
        return parsed
    return None


def is_foreground_shell_command(parsed: str | CommandPayload) -> bool:
    """Return True when ``parsed`` requires suspending live UI for foreground TTY access."""
    return isinstance(parsed, ShellCommand) and not parsed.output_to_prompt


async def _run_foreground_shell_command(
    command: str,
    *,
    prompt_prefix: str,
    first_line: str,
) -> None:
    """Suspend live UI, hand over the TTY to ``command``, and flush buffered output on exit."""
    from fast_agent.ui.console import suspend_console_output
    from fast_agent.ui.interactive_shell import run_interactive_shell_command
    from fast_agent.ui.progress_display import progress_display
    from fast_agent.ui.prompt.turn_input import suspend_turn_input
    from fast_agent.ui.streaming.display import suspend_streaming_displays

    is_tty = False
    try:
        is_tty = bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        is_tty = False

    with suspend_streaming_displays(), progress_display.paused():
        rich_print(
            f"[bold red]{escape_markup(prompt_prefix)}[/bold red] "
            f"[red]{escape_markup(first_line)}[/red]"
        )
        with suspend_turn_input(), suspend_console_output():
            result = await asyncio.to_thread(
                run_interactive_shell_command,
                command,
                show_output=is_tty,
                echo_command=False,
                restore_scroll_region=is_tty,
            )
        if not is_tty:
            stdout_text = result.stdout.rstrip()
            if stdout_text:
                if len(stdout_text) > _SHELL_OUTPUT_MAX_CHARS:
                    stdout_text = stdout_text[:_SHELL_OUTPUT_MAX_CHARS] + "\n… (output truncated)"
                rich_print(Text(stdout_text))
        if result.exit_code not in (0, None):
            rich_print(f"[dim red]exit {result.exit_code}[/dim red]")


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

    if isinstance(parsed, ShellCommand):
        await _run_foreground_shell_command(
            parsed.command,
            prompt_prefix=prompt_prefix,
            first_line=first_line,
        )
        return

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
