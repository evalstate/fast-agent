"""
Interactive prompt functionality for agents.

This module provides interactive command-line functionality for agents,
extracted from the original AgentApp implementation to support the new DirectAgentApp.

Usage:
    prompt = InteractivePrompt()
    await prompt.prompt_loop(
        send_func=agent_app.send,
        default_agent="default_agent",
        available_agents=["agent1", "agent2"],
        prompt_provider=agent_app
    )
"""

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Union, cast, runtime_checkable

from mcp.types import PromptMessage
from rich import print as rich_print
from rich.text import Text

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.console_display import ConsoleDisplay

from fast_agent.agents.agent_types import AgentType
from fast_agent.agents.tool_runner import HistoryRollbackState
from fast_agent.cli.runtime.shell_cwd_policy import (
    can_prompt_for_missing_cwd,
    collect_shell_cwd_issues_from_runtime_agents,
    create_missing_shell_cwd_directories,
    effective_missing_shell_cwd_policy,
    resolve_missing_shell_cwd_policy,
)
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.protocols import HistoryEditableAgent
from fast_agent.config import get_settings
from fast_agent.core.exceptions import PromptExitError
from fast_agent.interfaces import AgentProtocol, TurnCancellationStateCapable
from fast_agent.tools.session_environment import ShellExecutionResult
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui import enhanced_prompt
from fast_agent.ui.command_payloads import (
    CommandPayload,
    InterruptCommand,
    ShellCommand,
    is_command_payload,
)
from fast_agent.ui.console import configure_console_stream
from fast_agent.ui.display_suppression import suppress_interactive_display
from fast_agent.ui.enhanced_prompt import (
    get_enhanced_input,
    get_selection_input,
    handle_special_commands,
    parse_special_input,
    set_last_copyable_output,
)
from fast_agent.ui.interactive_diagnostics import write_interactive_trace
from fast_agent.ui.interactive_shell import run_interactive_shell_command
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.prompt.input import resolve_shell_working_dir
from fast_agent.ui.prompt.resource_mentions import (
    build_prompt_with_resources,
    parse_mentions,
    resolve_mentions,
)
from fast_agent.ui.prompt_marks import emit_prompt_mark
from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import strip_casefold

_MCP_RUNTIME_HANDLERS_COMPAT = mcp_runtime_handlers

# Type alias for the send function
SendFunc = Callable[[Union[str, PromptMessage, PromptMessageExtended], str], Awaitable[str]]
type PromptLoopResult = str | ShellExecutionResult

# Type alias for the agent getter function
AgentGetter = Callable[[str], object | None]


@runtime_checkable
class DisplayCapable(Protocol):
    @property
    def display(self) -> "ConsoleDisplay": ...


@dataclass(frozen=True, slots=True)
class HashSendExecution:
    """Result of executing a delegated hash-send request."""

    buffer_prefill: str | None


@dataclass(slots=True)
class PromptLoopAgents:
    """Mutable agent roster tracked across interactive turns."""

    current_agent: str
    available_agents: list[str]
    available_agents_set: set[str]


@dataclass(slots=True)
class PendingCommandExecution:
    """Post-dispatch work that should happen outside command handling."""

    hash_send_target: str | None = None
    hash_send_message: str | None = None
    hash_send_quiet: bool = False
    shell_execute_cmd: str | None = None

    def has_pending_execution(self) -> bool:
        return (
            self.hash_send_target is not None
            and self.hash_send_message is not None
            or self.shell_execute_cmd is not None
        )


@dataclass(slots=True)
class PromptTurnPreparation:
    """Outcome of pre-input prompt-turn orchestration."""

    agent_state: PromptLoopAgents | None
    should_continue: bool = False
    should_return: bool = False


@dataclass(slots=True)
class PromptInputPhase:
    """Collected input and post-input refresh state for a turn."""

    user_input: str | CommandPayload | None
    agent_state: PromptLoopAgents | None
    buffer_prefill: str
    should_continue: bool = False
    should_return: bool = False


@dataclass(slots=True)
class PromptCommandPhase:
    """Result of command handling prior to send execution."""

    agent_state: PromptLoopAgents
    pending: PendingCommandExecution
    buffer_prefill: str
    should_continue: bool = False
    should_return: bool = False


@dataclass(slots=True)
class DispatchApplicationResult:
    """Prompt-loop state after applying a command dispatch result."""

    agent_state: PromptLoopAgents
    pending: PendingCommandExecution
    buffer_prefill: str
    should_continue: bool


@dataclass(slots=True)
class PendingExecutionResult:
    """Result of running post-command pending work."""

    result: PromptLoopResult
    buffer_prefill: str | None = None
    handled: bool = False


@dataclass(slots=True)
class PromptLoopRuntimeState:
    """Cross-turn interactive prompt control state."""

    ctrl_c_deadline: float | None = None
    startup_warning_digest_checked: bool = False
    shell_cwd_startup_prompt_checked: bool = False


def _turn_preparation_ready(turn_preparation: PromptTurnPreparation) -> bool:
    return not turn_preparation.should_continue and turn_preparation.agent_state is not None


def _input_phase_ready(input_phase: PromptInputPhase) -> bool:
    return (
        not input_phase.should_continue
        and input_phase.user_input is not None
        and input_phase.agent_state is not None
    )


def _hash_send_status_text(prefix: str, target_agent: str, suffix: str, *, style: str) -> Text:
    text = Text(prefix, style=style)
    text.append(target_agent)
    text.append(suffix, style=style)
    return text


def _warning_text(message: str) -> Text:
    return Text(message, style="yellow")


def _clear_current_task_cancellation_requests() -> int:
    """Clear pending cancellation requests on the current interactive task.

    Interactive turn cancellation is user-facing recoverable behavior: after a
    cancelled turn we return to the prompt instead of tearing down the whole
    session. Some provider/streaming paths propagate cancellation by marking the
    current task as cancelling even when the turn is later normalized into a
    cancelled result. Clear that latent state before the next turn so a
    subsequent Ctrl+C is handled as a fresh cancel instead of an immediate exit.
    """
    task = asyncio.current_task()
    if task is None:
        return 0

    cleared = 0
    while task.cancelling() > 0:
        task.uncancel()
        cleared += 1

    if cleared:
        write_interactive_trace("prompt.task_uncancelled", cleared=cleared)
    return cleared


class InteractivePrompt:
    """
    Provides interactive prompt functionality that works with any agent implementation.
    This is extracted from the original AgentApp implementation to support DirectAgentApp.
    """

    def __init__(self, agent_types: dict[str, AgentType] | None = None) -> None:
        """
        Initialize the interactive prompt.

        Args:
            agent_types: Dictionary mapping agent names to their types for display
        """
        self.agent_types: dict[str, AgentType] = agent_types or {}

    def _get_agent_or_warn(self, prompt_provider: "AgentApp", agent_name: str) -> Any | None:
        try:
            return prompt_provider._agent(agent_name)
        except Exception:
            message = Text("Unable to load agent '", style="red")
            message.append(agent_name)
            message.append("'", style="red")
            rich_print(message)
            return None

    def _get_history_agent_or_warn(
        self,
        prompt_provider: "AgentApp",
        agent_name: str,
    ) -> HistoryEditableAgent | None:
        agent = self._get_agent_or_warn(prompt_provider, agent_name)
        if agent is None:
            return None
        return cast("HistoryEditableAgent", agent)

    async def _get_all_prompts(
        self,
        prompt_provider: "AgentApp",
        agent_name: str | None = None,
    ) -> list[prompt_handlers.PromptSummary]:
        from fast_agent.ui.interactive.command_context import build_command_context

        target_agent = prompt_provider.resolve_target_agent_name(agent_name) or ""
        context = build_command_context(prompt_provider, target_agent)
        return await prompt_handlers._get_all_prompts(context, agent_name)

    def _merge_pinned_agents(
        self,
        *,
        prompt_provider: "AgentApp",
        agent_names: list[str],
        pinned_agent: str | None,
    ) -> list[str]:
        if not pinned_agent or pinned_agent in agent_names:
            return agent_names
        try:
            known_agents = set(prompt_provider.registered_agent_names())
        except Exception:
            return agent_names
        if pinned_agent in known_agents:
            return [pinned_agent, *agent_names]
        return agent_names

    def _sync_enhanced_prompt_agents(
        self,
        *,
        prompt_provider: "AgentApp",
        available_agents: list[str],
    ) -> None:
        force_include = available_agents[0] if available_agents else None
        self.agent_types = prompt_provider.visible_agent_types(force_include=force_include)
        enhanced_prompt.available_agents = set(available_agents)

    def _build_initial_agent_state(
        self,
        *,
        default_agent: str,
        available_agents: list[str],
        prompt_provider: "AgentApp",
        pinned_agent: str | None,
    ) -> PromptLoopAgents:
        agent = default_agent
        if not agent:
            if available_agents:
                agent = available_agents[0]
            else:
                raise ValueError("No default agent available")

        next_available_agents = self._merge_pinned_agents(
            prompt_provider=prompt_provider,
            agent_names=list(available_agents),
            pinned_agent=pinned_agent,
        )
        if agent not in next_available_agents:
            raise ValueError(f"No agent named '{agent}'")

        self._sync_enhanced_prompt_agents(
            prompt_provider=prompt_provider,
            available_agents=next_available_agents,
        )
        return PromptLoopAgents(
            current_agent=agent,
            available_agents=next_available_agents,
            available_agents_set=set(next_available_agents),
        )

    def _current_agent_roster(
        self,
        *,
        prompt_provider: "AgentApp",
        pinned_agent: str | None,
    ) -> tuple[list[str], set[str]]:
        base_agent_names = list(prompt_provider.visible_agent_names())
        available_agents = self._merge_pinned_agents(
            prompt_provider=prompt_provider,
            agent_names=base_agent_names,
            pinned_agent=pinned_agent,
        )
        return available_agents, set(available_agents)

    def _select_active_agent_or_exit(
        self,
        *,
        preferred_agent: str | None,
        current_agent: str,
        available_agents: list[str],
        available_agents_set: set[str],
        no_agents_message: str,
    ) -> str | None:
        if preferred_agent and preferred_agent in available_agents_set:
            return preferred_agent
        if current_agent in available_agents_set:
            return current_agent
        if available_agents:
            return available_agents[0]
        rich_print(no_agents_message)
        return None

    async def _refresh_agents_if_needed(
        self,
        *,
        prompt_provider: "AgentApp",
        state: PromptLoopAgents,
        pinned_agent: str | None,
        skip_refresh: bool = False,
        no_agents_message: str = "[red]No agents available.[/red]",
    ) -> PromptLoopAgents | None:
        refreshed = False
        if not skip_refresh:
            refreshed = await prompt_provider.refresh_if_needed()
        refresh_result = prompt_provider.latest_refresh_result()

        next_available_agents, next_available_agents_set = self._current_agent_roster(
            prompt_provider=prompt_provider,
            pinned_agent=pinned_agent,
        )
        if (
            not refreshed
            and next_available_agents == state.available_agents
            and next_available_agents_set == state.available_agents_set
        ):
            return state

        next_agent = self._select_active_agent_or_exit(
            preferred_agent=refresh_result.active_agent if refreshed else None,
            current_agent=state.current_agent,
            available_agents=next_available_agents,
            available_agents_set=next_available_agents_set,
            no_agents_message=no_agents_message,
        )
        if next_agent is None:
            return None

        self._sync_enhanced_prompt_agents(
            prompt_provider=prompt_provider,
            available_agents=next_available_agents,
        )
        if refreshed:
            for warning in refresh_result.warnings:
                rich_print(_warning_text(warning))
            rich_print("[green]AgentCards reloaded.[/green]")

        return PromptLoopAgents(
            current_agent=next_agent,
            available_agents=next_available_agents,
            available_agents_set=next_available_agents_set,
        )

    def _describe_cancelled_history_state(self, history_state: HistoryRollbackState | None) -> str:
        if history_state is None:
            return "History reconciliation completed."

        status = history_state.status
        removed_messages = history_state.removed_messages

        if status == "history_disabled":
            return (
                "Agent history is configured with use_history=false, so no per-turn "
                "history was persisted."
            )

        if status == "history_empty":
            return "History was already empty."

        if status == "appended_interrupted_tool_result":
            return (
                "Added an interrupted tool-result marker "
                "('**The user interrupted this tool call**')."
            )

        if status == "history_unchanged":
            return "No dangling tool call was found; history was left unchanged."

        return (
            f"History reconciliation completed (removed "
            f"{format_count(removed_messages, 'message')})."
        )

    def _report_previous_turn_cancellation(
        self,
        *,
        prompt_provider: "AgentApp",
        agent_name: str,
        clear_progress_for_agent: Callable[[str | None], None],
    ) -> None:
        try:
            agent_obj = prompt_provider._agent(agent_name)
        except Exception:
            agent_obj = None

        if (
            not isinstance(agent_obj, TurnCancellationStateCapable)
            or not agent_obj.last_turn_cancelled
        ):
            return

        _clear_current_task_cancellation_requests()
        clear_progress_for_agent(agent_name)
        reason = agent_obj.last_turn_cancel_reason
        history_state = agent_obj.last_turn_history_state
        agent_obj.clear_last_turn_cancellation()
        state_message = self._describe_cancelled_history_state(history_state)
        write_interactive_trace(
            "prompt.previous_turn_cancelled",
            agent=agent_name,
            reason=reason,
            state=state_message,
        )
        rich_print(
            "[yellow]Previous turn was {reason}. {state} "
            "Use /history to inspect or manipulate history.[/yellow]".format(
                reason=reason,
                state=state_message,
            )
        )

    def _handle_ctrl_c_interrupt(
        self,
        *,
        runtime_state: PromptLoopRuntimeState,
        exit_window_seconds: float,
    ) -> None:
        now = time.monotonic()
        if runtime_state.ctrl_c_deadline is not None and now <= runtime_state.ctrl_c_deadline:
            rich_print("[red]Second Ctrl+C received; exiting fast-agent session.[/red]")
            raise PromptExitError("User requested to exit fast-agent session")

        runtime_state.ctrl_c_deadline = now + exit_window_seconds
        rich_print(
            "[yellow]Interrupted operation; returning to prompt. "
            "Press Ctrl+C again within 2 seconds to exit.[/yellow]"
        )

    def _clear_ctrl_c_interrupt(self, *, runtime_state: PromptLoopRuntimeState) -> None:
        runtime_state.ctrl_c_deadline = None

    def _handle_inflight_cancel(self, *, runtime_state: PromptLoopRuntimeState) -> None:
        """Handle user cancellation while generation or tool calling is active."""
        runtime_state.ctrl_c_deadline = None
        _clear_current_task_cancellation_requests()
        write_interactive_trace("prompt.inflight_cancel")
        rich_print("[yellow]Generation cancelled by user.[/yellow]")

    def _clear_progress_for_agent(self, agent_name: str | None) -> None:
        """Remove stale progress rows after an interrupted/cancelled send."""
        with suppress(Exception):
            progress_display.clear_agent_tasks(agent_name)

    def _last_assistant_message_cancelled(
        self,
        *,
        prompt_provider: "AgentApp",
        agent_name: str | None,
    ) -> bool:
        """Return True when an agent's latest history message is assistant CANCELLED."""
        if not agent_name:
            return False

        try:
            agent_obj = prompt_provider._agent(agent_name)
        except Exception:
            return False

        try:
            history = agent_obj.message_history
            last_message = history[-1] if history else None
            return bool(
                last_message is not None
                and last_message.role == "assistant"
                and last_message.stop_reason == LlmStopReason.CANCELLED
            )
        except Exception:
            return False

    def _emit_startup_warning_digest_once(
        self,
        *,
        runtime_state: PromptLoopRuntimeState,
    ) -> None:
        if runtime_state.startup_warning_digest_checked:
            return
        runtime_state.startup_warning_digest_checked = True

        try:
            from fast_agent.ui import notification_tracker

            startup_warnings = notification_tracker.pop_startup_warnings()
        except Exception:
            return

        if not startup_warnings:
            return

        header = Text("Startup warning", style="yellow")
        if len(startup_warnings) > 1:
            header.append(f"s ({len(startup_warnings)})")
        header.append(":")
        rich_print(header)

        prefix = "  • " if len(startup_warnings) > 1 else "  "
        for warning in startup_warnings:
            rich_print(Text(f"{prefix}{warning}"))

    @staticmethod
    def _shell_cwd_startup_issues(
        prompt_provider: "AgentApp",
    ) -> tuple[Any, Any] | None:
        runtime_agents = prompt_provider.registered_agents()
        if not isinstance(runtime_agents, dict):
            return None

        issues = collect_shell_cwd_issues_from_runtime_agents(runtime_agents, cwd=Path.cwd())
        if not issues:
            return None
        return runtime_agents, issues

    @staticmethod
    async def _confirm_shell_cwd_creation() -> bool:
        selection = await get_selection_input(
            "Create missing shell directories now? [Y/n] ",
            default="y",
            allow_cancel=False,
            complete_options=False,
        )
        answer = strip_casefold(selection or "")
        return answer in {"", "y", "yes"}

    @staticmethod
    def _print_shell_cwd_creation_result(creation_result: Any) -> None:
        if creation_result.created_paths:
            rich_print("[green]Created missing shell cwd directories:[/green]")
            for path in creation_result.created_paths:
                rich_print(Text(f"  • {path}"))

        if creation_result.errors:
            rich_print("[red]Failed to create one or more shell cwd directories:[/red]")
            for item in creation_result.errors:
                rich_print(Text(f"  • {item.path}: {item.message}"))

    @staticmethod
    def _clear_shell_cwd_warnings_if_resolved(runtime_agents: Any) -> None:
        remaining_issues = collect_shell_cwd_issues_from_runtime_agents(
            runtime_agents,
            cwd=Path.cwd(),
        )
        if remaining_issues:
            return
        try:
            from fast_agent.ui import notification_tracker

            notification_tracker.remove_startup_warnings_containing("shell cwd")
        except Exception:
            pass

    async def _maybe_prompt_for_shell_cwd_startup_once(
        self,
        *,
        runtime_state: PromptLoopRuntimeState,
        prompt_provider: "AgentApp",
        shell_cwd_policy: str,
    ) -> None:
        if runtime_state.shell_cwd_startup_prompt_checked:
            return
        runtime_state.shell_cwd_startup_prompt_checked = True

        if shell_cwd_policy != "ask":
            return

        issue_context = self._shell_cwd_startup_issues(prompt_provider)
        if issue_context is None:
            return
        runtime_agents, issues = issue_context

        rich_print(
            f"[yellow]Shell cwd startup check:[/yellow] {format_count(len(issues), 'issue')} found."
        )

        if not await self._confirm_shell_cwd_creation():
            return

        creation_result = create_missing_shell_cwd_directories(issues)
        self._print_shell_cwd_creation_result(creation_result)
        self._clear_shell_cwd_warnings_if_resolved(runtime_agents)

    def _apply_dispatch_result(
        self,
        *,
        state: PromptLoopAgents,
        dispatch_result: Any,
        buffer_prefill: str,
    ) -> DispatchApplicationResult:
        next_available_agents = dispatch_result.available_agents or state.available_agents
        next_available_agents_set = (
            dispatch_result.available_agents_set or state.available_agents_set
        )
        next_state = PromptLoopAgents(
            current_agent=dispatch_result.next_agent or state.current_agent,
            available_agents=next_available_agents,
            available_agents_set=next_available_agents_set,
        )
        pending = PendingCommandExecution(
            hash_send_target=dispatch_result.hash_send_target,
            hash_send_message=dispatch_result.hash_send_message,
            hash_send_quiet=dispatch_result.hash_send_quiet,
            shell_execute_cmd=dispatch_result.shell_execute_cmd,
        )
        next_buffer_prefill = (
            dispatch_result.buffer_prefill
            if dispatch_result.buffer_prefill is not None
            else buffer_prefill
        )
        should_continue = dispatch_result.handled and not pending.has_pending_execution()
        return DispatchApplicationResult(
            agent_state=next_state,
            pending=pending,
            buffer_prefill=next_buffer_prefill,
            should_continue=should_continue,
        )

    async def _prepare_prompt_turn(
        self,
        *,
        prompt_provider: "AgentApp",
        agent_state: PromptLoopAgents,
        pinned_agent: str | None,
        runtime_state: PromptLoopRuntimeState,
        ctrl_c_exit_window_seconds: float,
        shell_cwd_policy: str,
    ) -> PromptTurnPreparation:
        progress_display.pause(cancel_deferred_on_noop=True)
        self._report_previous_turn_cancellation(
            prompt_provider=prompt_provider,
            agent_name=agent_state.current_agent,
            clear_progress_for_agent=self._clear_progress_for_agent,
        )
        try:
            refreshed_state = await self._refresh_agents_if_needed(
                prompt_provider=prompt_provider,
                state=agent_state,
                pinned_agent=pinned_agent,
                no_agents_message="[red]No agents available after refresh.[/red]",
            )
        except KeyboardInterrupt:
            self._handle_ctrl_c_interrupt(
                runtime_state=runtime_state,
                exit_window_seconds=ctrl_c_exit_window_seconds,
            )
            return PromptTurnPreparation(agent_state=agent_state, should_continue=True)

        if refreshed_state is None:
            return PromptTurnPreparation(agent_state=None, should_return=True)

        await self._maybe_prompt_for_shell_cwd_startup_once(
            runtime_state=runtime_state,
            prompt_provider=prompt_provider,
            shell_cwd_policy=shell_cwd_policy,
        )
        self._emit_startup_warning_digest_once(runtime_state=runtime_state)
        return PromptTurnPreparation(agent_state=refreshed_state)

    async def _collect_turn_input(
        self,
        *,
        prompt_provider: "AgentApp",
        agent_state: PromptLoopAgents,
        pinned_agent: str | None,
        default: str,
        buffer_prefill: str,
        runtime_state: PromptLoopRuntimeState,
        ctrl_c_exit_window_seconds: float,
    ) -> PromptInputPhase:
        noenv_mode = prompt_provider.noenv_mode
        try:
            user_input = await get_enhanced_input(
                agent_name=agent_state.current_agent,
                default=default,
                show_default=(default != ""),
                show_stop_hint=True,
                multiline=False,
                available_agent_names=agent_state.available_agents,
                agent_types=self.agent_types,
                agent_provider=prompt_provider,
                noenv_mode=noenv_mode,
                pre_populate_buffer=buffer_prefill,
            )
        except KeyboardInterrupt:
            self._handle_ctrl_c_interrupt(
                runtime_state=runtime_state,
                exit_window_seconds=ctrl_c_exit_window_seconds,
            )
            return PromptInputPhase(
                user_input=None,
                agent_state=agent_state,
                buffer_prefill=buffer_prefill,
                should_continue=True,
            )

        next_buffer_prefill = ""
        if isinstance(user_input, str):
            user_input = parse_special_input(user_input)

        if not isinstance(user_input, InterruptCommand):
            self._clear_ctrl_c_interrupt(runtime_state=runtime_state)

        if isinstance(user_input, ShellCommand):
            return PromptInputPhase(
                user_input=user_input,
                agent_state=agent_state,
                buffer_prefill=next_buffer_prefill,
            )

        try:
            refreshed_state = await self._refresh_agents_if_needed(
                prompt_provider=prompt_provider,
                state=agent_state,
                pinned_agent=pinned_agent,
                no_agents_message="[red]No agents available after refresh.[/red]",
            )
        except KeyboardInterrupt:
            self._handle_ctrl_c_interrupt(
                runtime_state=runtime_state,
                exit_window_seconds=ctrl_c_exit_window_seconds,
            )
            return PromptInputPhase(
                user_input=None,
                agent_state=agent_state,
                buffer_prefill=next_buffer_prefill,
                should_continue=True,
            )

        if refreshed_state is None:
            return PromptInputPhase(
                user_input=None,
                agent_state=None,
                buffer_prefill=next_buffer_prefill,
                should_return=True,
            )

        return PromptInputPhase(
            user_input=user_input,
            agent_state=refreshed_state,
            buffer_prefill=next_buffer_prefill,
        )

    async def _process_turn_command_phase(
        self,
        *,
        prompt_provider: "AgentApp",
        agent_state: PromptLoopAgents,
        user_input: str | CommandPayload,
        buffer_prefill: str,
        pinned_agent: str | None,
        runtime_state: PromptLoopRuntimeState,
        ctrl_c_exit_window_seconds: float,
    ) -> PromptCommandPhase:
        pending = PendingCommandExecution()
        command_result = await handle_special_commands(user_input, prompt_provider)

        if is_command_payload(command_result):
            from fast_agent.ui.interactive.command_dispatch import dispatch_command_payload

            try:
                dispatch_result = await dispatch_command_payload(
                    self,
                    command_result,
                    prompt_provider=prompt_provider,
                    agent=agent_state.current_agent,
                    available_agents=agent_state.available_agents,
                    available_agents_set=agent_state.available_agents_set,
                    merge_pinned_agents=lambda agent_names: self._merge_pinned_agents(
                        prompt_provider=prompt_provider,
                        agent_names=agent_names,
                        pinned_agent=pinned_agent,
                    ),
                    buffer_prefill=buffer_prefill,
                    shell_working_dir=resolve_shell_working_dir(
                        agent_name=agent_state.current_agent,
                        agent_provider=prompt_provider,
                    ),
                )
            except KeyboardInterrupt:
                self._handle_ctrl_c_interrupt(
                    runtime_state=runtime_state,
                    exit_window_seconds=ctrl_c_exit_window_seconds,
                )
                return PromptCommandPhase(
                    agent_state=agent_state,
                    pending=pending,
                    buffer_prefill=buffer_prefill,
                    should_continue=True,
                )

            dispatch_application = self._apply_dispatch_result(
                state=agent_state,
                dispatch_result=dispatch_result,
                buffer_prefill=buffer_prefill,
            )
            agent_state = dispatch_application.agent_state
            pending = dispatch_application.pending
            buffer_prefill = dispatch_application.buffer_prefill
            if dispatch_result.should_return:
                return PromptCommandPhase(
                    agent_state=agent_state,
                    pending=pending,
                    buffer_prefill=buffer_prefill,
                    should_return=True,
                )
            if dispatch_application.should_continue:
                return PromptCommandPhase(
                    agent_state=agent_state,
                    pending=pending,
                    buffer_prefill=buffer_prefill,
                    should_continue=True,
                )

        if (
            not pending.has_pending_execution()
            and isinstance(user_input, str)
            and user_input.strip().upper() == "STOP"
        ):
            return PromptCommandPhase(
                agent_state=agent_state,
                pending=pending,
                buffer_prefill=buffer_prefill,
                should_return=True,
            )

        if self._should_continue_after_command(
            user_input=user_input,
            command_result=command_result,
            pending=pending,
        ):
            return PromptCommandPhase(
                agent_state=agent_state,
                pending=pending,
                buffer_prefill=buffer_prefill,
                should_continue=True,
            )

        return PromptCommandPhase(
            agent_state=agent_state,
            pending=pending,
            buffer_prefill=buffer_prefill,
        )

    def _should_continue_after_command(
        self,
        *,
        user_input: object,
        command_result: object,
        pending: PendingCommandExecution,
    ) -> bool:
        if pending.has_pending_execution():
            return False
        if command_result or is_command_payload(user_input) or is_command_payload(command_result):
            return True
        if not isinstance(user_input, str):
            return True
        return user_input == ""

    async def _handle_pending_execution(
        self,
        *,
        pending: PendingCommandExecution,
        send_func: SendFunc,
        quiet_send_func: SendFunc | None,
        prompt_provider: "AgentApp",
        display: "ConsoleDisplay",
        current_result: PromptLoopResult,
        runtime_state: PromptLoopRuntimeState,
    ) -> PendingExecutionResult:
        if pending.hash_send_target is not None and pending.hash_send_message is not None:
            active_send_func = (
                quiet_send_func if pending.hash_send_quiet and quiet_send_func else send_func
            )
            hash_send_execution = await self._execute_hash_send(
                send_func=active_send_func,
                target_agent=pending.hash_send_target,
                message=pending.hash_send_message,
                quiet=pending.hash_send_quiet,
                clear_progress_for_agent=self._clear_progress_for_agent,
                clear_ctrl_c_interrupt=lambda: self._clear_ctrl_c_interrupt(
                    runtime_state=runtime_state
                ),
                handle_inflight_cancel=lambda: self._handle_inflight_cancel(
                    runtime_state=runtime_state
                ),
                last_assistant_message_cancelled=lambda agent_name: (
                    self._last_assistant_message_cancelled(
                        prompt_provider=prompt_provider,
                        agent_name=agent_name,
                    )
                ),
            )
            return PendingExecutionResult(
                result=current_result,
                buffer_prefill=hash_send_execution.buffer_prefill,
                handled=True,
            )

        if pending.shell_execute_cmd:
            print(f"$ {pending.shell_execute_cmd}", flush=True)
            emit_prompt_mark("C")
            result = run_interactive_shell_command(
                pending.shell_execute_cmd,
                echo_command=False,
            )
            emit_prompt_mark(f"D;{result.exit_code}")

            if result.stdout.strip():
                set_last_copyable_output(result.stdout.rstrip())

            if result.exit_code != 0:
                display.show_shell_exit_code(result.exit_code)

            return PendingExecutionResult(result=result, handled=True)

        return PendingExecutionResult(result=current_result)

    async def _resolve_prompt_payload(
        self,
        *,
        prompt_provider: "AgentApp",
        agent_name: str,
        user_input: str,
    ) -> str | PromptMessageExtended | None:
        prompt_payload: str | PromptMessageExtended = user_input
        parsed_mentions = parse_mentions(
            user_input,
            cwd=resolve_shell_working_dir(agent_name=agent_name, agent_provider=prompt_provider),
        )
        for warning in parsed_mentions.warnings:
            rich_print(_warning_text(warning))

        if not parsed_mentions.mentions:
            return prompt_payload

        try:
            agent_for_mentions = prompt_provider._agent(agent_name)
        except Exception:
            message = Text("Unable to resolve resource mentions: agent '", style="red")
            message.append(agent_name)
            message.append("' unavailable", style="red")
            rich_print(message)
            return user_input

        try:
            resolved_mentions = await resolve_mentions(agent_for_mentions, parsed_mentions)
            return build_prompt_with_resources(user_input, resolved_mentions)
        except Exception as exc:
            message = Text("Failed to resolve resource mentions: ", style="red")
            message.append(str(exc))
            rich_print(message)
            return user_input

    async def _send_regular_message(
        self,
        *,
        send_func: SendFunc,
        prompt_payload: str | PromptMessageExtended,
        prompt_provider: "AgentApp",
        agent_name: str,
        runtime_state: PromptLoopRuntimeState,
    ) -> PromptLoopResult | None:
        emit_prompt_mark("C")
        write_interactive_trace("prompt.send.start", agent=agent_name)
        progress_display.resume()
        try:
            result = await send_func(prompt_payload, agent_name)
        except KeyboardInterrupt:
            write_interactive_trace("prompt.send.keyboard_interrupt", agent=agent_name)
            self._clear_progress_for_agent(agent_name)
            self._handle_inflight_cancel(runtime_state=runtime_state)
            return None
        except asyncio.CancelledError:
            write_interactive_trace("prompt.send.cancelled_error", agent=agent_name)
            self._clear_progress_for_agent(agent_name)
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            self._handle_inflight_cancel(runtime_state=runtime_state)
            return None
        finally:
            write_interactive_trace("prompt.send.finally_pause", agent=agent_name)
            progress_display.pause(cancel_deferred_on_noop=True)
            emit_prompt_mark("D")

        if result and result.startswith("▲ **System Error:**"):
            print(result)

        if self._last_assistant_message_cancelled(
            prompt_provider=prompt_provider,
            agent_name=agent_name,
        ):
            _clear_current_task_cancellation_requests()
            self._clear_progress_for_agent(agent_name)
            self._clear_ctrl_c_interrupt(runtime_state=runtime_state)

        if result:
            set_last_copyable_output(result)

        return result

    @staticmethod
    def _apply_pending_execution_result(
        *,
        pending_result: PendingExecutionResult,
        buffer_prefill: str,
    ) -> tuple[PromptLoopResult, str, bool]:
        if not pending_result.handled:
            return pending_result.result, buffer_prefill, False
        return (
            pending_result.result,
            pending_result.buffer_prefill or buffer_prefill,
            True,
        )

    async def _send_user_input_if_prompt_payload(
        self,
        *,
        send_func: SendFunc,
        prompt_provider: "AgentApp",
        agent_name: str,
        user_input: str,
        runtime_state: PromptLoopRuntimeState,
    ) -> PromptLoopResult | None:
        prompt_payload = await self._resolve_prompt_payload(
            prompt_provider=prompt_provider,
            agent_name=agent_name,
            user_input=user_input,
        )
        if prompt_payload is None:
            return None
        return await self._send_regular_message(
            send_func=send_func,
            prompt_payload=prompt_payload,
            prompt_provider=prompt_provider,
            agent_name=agent_name,
            runtime_state=runtime_state,
        )

    async def prompt_loop(
        self,
        send_func: SendFunc,
        default_agent: str,
        available_agents: list[str],
        prompt_provider: "AgentApp",
        pinned_agent: str | None = None,
        default: str = "",
        quiet_send_func: SendFunc | None = None,
    ) -> PromptLoopResult:
        """
        Start an interactive prompt session.

        Args:
            send_func: Function to send messages to agents
            quiet_send_func: Optional function used for quiet delegated sends
            default_agent: Name of the default agent to use
            available_agents: List of available agent names
            prompt_provider: AgentApp instance for accessing agents and prompts
            pinned_agent: Explicitly targeted agent name to preserve across refreshes
            default: Default message to use when user presses enter

        Returns:
            The result of the interactive session
        """
        configure_console_stream("stdout")

        agent_state = self._build_initial_agent_state(
            default_agent=default_agent,
            available_agents=available_agents,
            prompt_provider=prompt_provider,
            pinned_agent=pinned_agent,
        )

        from fast_agent.ui.console_display import ConsoleDisplay

        display = ConsoleDisplay(config=get_settings())

        result: PromptLoopResult = ""
        buffer_prefill = ""  # One-off buffer content for # command results
        ctrl_c_exit_window_seconds = 2.0
        runtime_state = PromptLoopRuntimeState()
        configured_shell_cwd_policy = get_settings().shell_execution.missing_cwd_policy
        resolved_shell_cwd_policy = resolve_missing_shell_cwd_policy(
            cli_override=prompt_provider.missing_shell_cwd_policy_override,
            configured_policy=configured_shell_cwd_policy,
        )
        shell_cwd_policy = effective_missing_shell_cwd_policy(
            resolved_shell_cwd_policy,
            can_prompt=can_prompt_for_missing_cwd(
                mode="interactive",
                execution_mode="repl",
                stdin_is_tty=sys.stdin.isatty(),
                tty_device_available=False,
            ),
        )

        while True:
            turn_preparation = await self._prepare_prompt_turn(
                prompt_provider=prompt_provider,
                agent_state=agent_state,
                pinned_agent=pinned_agent,
                runtime_state=runtime_state,
                ctrl_c_exit_window_seconds=ctrl_c_exit_window_seconds,
                shell_cwd_policy=shell_cwd_policy,
            )
            if turn_preparation.should_return:
                return result
            if not _turn_preparation_ready(turn_preparation):
                continue
            assert turn_preparation.agent_state is not None
            agent_state = turn_preparation.agent_state

            input_phase = await self._collect_turn_input(
                prompt_provider=prompt_provider,
                agent_state=agent_state,
                pinned_agent=pinned_agent,
                default=default,
                buffer_prefill=buffer_prefill,
                runtime_state=runtime_state,
                ctrl_c_exit_window_seconds=ctrl_c_exit_window_seconds,
            )
            if input_phase.should_return:
                return result
            if not _input_phase_ready(input_phase):
                continue
            assert input_phase.agent_state is not None
            assert input_phase.user_input is not None

            agent_state = input_phase.agent_state
            buffer_prefill = input_phase.buffer_prefill
            user_input = input_phase.user_input

            command_phase = await self._process_turn_command_phase(
                prompt_provider=prompt_provider,
                agent_state=agent_state,
                user_input=user_input,
                buffer_prefill=buffer_prefill,
                pinned_agent=pinned_agent,
                runtime_state=runtime_state,
                ctrl_c_exit_window_seconds=ctrl_c_exit_window_seconds,
            )
            if command_phase.should_return:
                return result

            agent_state = command_phase.agent_state
            buffer_prefill = command_phase.buffer_prefill
            if command_phase.should_continue:
                continue

            pending_result = await self._handle_pending_execution(
                pending=command_phase.pending,
                send_func=send_func,
                quiet_send_func=quiet_send_func,
                prompt_provider=prompt_provider,
                display=display,
                current_result=result,
                runtime_state=runtime_state,
            )
            result, buffer_prefill, pending_handled = self._apply_pending_execution_result(
                pending_result=pending_result,
                buffer_prefill=buffer_prefill,
            )
            if pending_handled:
                continue

            # Send the message to the agent
            # Type narrowing: by this point user_input is str (non-str inputs continue above)
            assert isinstance(user_input, str)
            send_result = await self._send_user_input_if_prompt_payload(
                send_func=send_func,
                prompt_provider=prompt_provider,
                agent_name=agent_state.current_agent,
                user_input=user_input,
                runtime_state=runtime_state,
            )
            if send_result is not None:
                result = send_result

        return result

    async def _execute_hash_send(
        self,
        *,
        send_func: SendFunc,
        target_agent: str,
        message: str,
        quiet: bool,
        clear_progress_for_agent: Callable[[str | None], None],
        clear_ctrl_c_interrupt: Callable[[], None],
        handle_inflight_cancel: Callable[[], None],
        last_assistant_message_cancelled: Callable[[str | None], bool],
    ) -> HashSendExecution:
        if not quiet:
            rich_print(_hash_send_status_text("Asking ", target_agent, "...", style="dim"))

        try:
            emit_prompt_mark("C")
            write_interactive_trace("prompt.hash_send.start", agent=target_agent, quiet=quiet)
            progress_display.resume()
            display_context = suppress_interactive_display() if quiet else nullcontext()
            with display_context:
                response_text = await send_func(message, target_agent)
        except KeyboardInterrupt:
            write_interactive_trace(
                "prompt.hash_send.keyboard_interrupt",
                agent=target_agent,
                quiet=quiet,
            )
            clear_progress_for_agent(target_agent)
            handle_inflight_cancel()
            return HashSendExecution(buffer_prefill=None)
        except asyncio.CancelledError:
            write_interactive_trace(
                "prompt.hash_send.cancelled_error",
                agent=target_agent,
                quiet=quiet,
            )
            clear_progress_for_agent(target_agent)
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            handle_inflight_cancel()
            return HashSendExecution(buffer_prefill=None)
        except Exception as exc:
            status_text = _hash_send_status_text("Error asking ", target_agent, ": ", style="red")
            status_text.append(str(exc))
            rich_print(status_text)
            return HashSendExecution(buffer_prefill=None)
        finally:
            write_interactive_trace(
                "prompt.hash_send.finally_pause",
                agent=target_agent,
                quiet=quiet,
            )
            progress_display.pause(cancel_deferred_on_noop=True)
            emit_prompt_mark("D")

        if last_assistant_message_cancelled(target_agent):
            _clear_current_task_cancellation_requests()
            clear_progress_for_agent(target_agent)
            clear_ctrl_c_interrupt()

        if response_text:
            if not quiet:
                rich_print(
                    _hash_send_status_text(
                        "Response from ",
                        target_agent,
                        " loaded into input buffer",
                        style="blue",
                    )
                )
            return HashSendExecution(buffer_prefill=response_text)

        status_text = _hash_send_status_text(
            "No response received from ",
            target_agent,
            "",
            style="dim" if quiet else "yellow",
        )
        if quiet:
            rich_print(status_text)
        else:
            rich_print(status_text)
        return HashSendExecution(buffer_prefill=None)

    def _resolve_display(
        self, prompt_provider: "AgentApp", agent_name: str | None
    ) -> "ConsoleDisplay":
        from fast_agent.ui.console_display import ConsoleDisplay

        agent = None
        if agent_name:
            try:
                agent = prompt_provider._agent(agent_name)
            except Exception:
                agent = None

        if isinstance(agent, DisplayCapable):
            return agent.display

        config = None
        if isinstance(agent, AgentProtocol):
            agent_context = agent.context
            config = agent_context.config if agent_context else None

        if config is None:
            config = get_settings()

        return ConsoleDisplay(config=config)
