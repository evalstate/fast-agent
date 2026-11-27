"""Mid-turn concurrent execution for read-only slash commands and non-interactive shell commands.

Mutating commands (such as ``/clear``, ``/compact``, ``/model <name>``, ``@agent``,
``/session new``, or interactive ``!`` shells) are intentionally NOT allowed here;
they remain queued in ``TurnInputReader`` and run in FIFO order once the active
turn completes.
"""

from __future__ import annotations

import asyncio
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
_SHELL_TIMEOUT_SECONDS = 30.0
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
        return (
            not parsed.interactive and not parsed.output_to_prompt and bool(parsed.command.strip())
        )

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


async def _run_concurrent_shell_command(
    command: str,
    *,
    working_dir: Path | None,
) -> None:
    """Run a non-interactive shell command with stdin closed and print output above the dock."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(working_dir) if working_dir is not None else None,
        )
    except OSError as exc:
        rich_print(f"[red]Shell command failed to start: {escape_markup(str(exc))}[/red]")
        return

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=_SHELL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        rich_print(
            f"[yellow]Shell command timed out after {int(_SHELL_TIMEOUT_SECONDS)}s: "
            f"{escape_markup(command)}[/yellow]"
        )
        return
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise

    stdout_text = stdout_bytes.decode("utf-8", errors="replace").rstrip()
    stderr_text = stderr_bytes.decode("utf-8", errors="replace").rstrip()

    if stdout_text:
        if len(stdout_text) > _SHELL_OUTPUT_MAX_CHARS:
            stdout_text = stdout_text[:_SHELL_OUTPUT_MAX_CHARS] + "\n… (output truncated)"
        rich_print(Text(stdout_text))
    if stderr_text:
        if len(stderr_text) > _SHELL_OUTPUT_MAX_CHARS:
            stderr_text = stderr_text[:_SHELL_OUTPUT_MAX_CHARS] + "\n… (stderr truncated)"
        rich_print(Text(stderr_text, style="red"))
    if proc.returncode not in (0, None):
        rich_print(f"[dim red]exit {proc.returncode}[/dim red]")


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

    if isinstance(parsed, ShellCommand):
        await _run_concurrent_shell_command(parsed.command, working_dir=shell_working_dir)
        return

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
