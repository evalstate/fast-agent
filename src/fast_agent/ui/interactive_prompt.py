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
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union

from mcp.types import PromptMessage
from rich import print as rich_print

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.console_display import ConsoleDisplay

from fast_agent.agents.agent_types import AgentType
from fast_agent.cli.runtime.shell_cwd_policy import (
    can_prompt_for_missing_cwd,
    collect_shell_cwd_issues_from_runtime_agents,
    create_missing_shell_cwd_directories,
    effective_missing_shell_cwd_policy,
    resolve_missing_shell_cwd_policy,
)
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers  # noqa: F401
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.config import get_settings
from fast_agent.core.exceptions import PromptExitError
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui import enhanced_prompt
from fast_agent.ui.command_payloads import (
    CommandPayload,
    InterruptCommand,
    ShellCommand,
    is_command_payload,
)
from fast_agent.ui.display_suppression import suppress_interactive_display
from fast_agent.ui.enhanced_prompt import (
    get_enhanced_input,
    get_selection_input,
    handle_special_commands,
    parse_special_input,
    set_last_copyable_output,
)
from fast_agent.ui.interactive_diagnostics import write_interactive_trace
from fast_agent.ui.interactive_shell import ShellExecutionResult, run_interactive_shell_command
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.prompt.input import resolve_shell_working_dir
from fast_agent.ui.prompt.resource_mentions import (
    build_prompt_with_resources,
    parse_mentions,
    resolve_mentions,
)
from fast_agent.ui.prompt_marks import emit_prompt_mark

# Type alias for the send function
SendFunc = Callable[[Union[str, PromptMessage, PromptMessageExtended], str], Awaitable[str]]
type PromptLoopResult = str | ShellExecutionResult

# Type alias for the agent getter function
AgentGetter = Callable[[str], object | None]


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
class PromptLoopRuntimeState:
    """Cross-turn interactive prompt control state."""

    ctrl_c_deadline: float | None = None
    startup_warning_digest_checked: bool = False
    shell_cwd_startup_prompt_checked: bool = False


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
            rich_print(f"[red]Unable to load agent '{agent_name}'[/red]")
            return None

    async def _get_all_prompts(
        self,
        prompt_provider: "AgentApp",
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
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
        current_agent: str,
        available_agents: list[str],
        available_agents_set: set[str],
        no_agents_message: str,
    ) -> str | None:
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
            rich_print("[green]AgentCards reloaded.[/green]")

        return PromptLoopAgents(
            current_agent=next_agent,
            available_agents=next_available_agents,
            available_agents_set=next_available_agents_set,
        )

    def _describe_cancelled_history_state(self, history_state: object | None) -> str:
        status = getattr(history_state, "status", None)
        removed_messages = getattr(history_state, "removed_messages", 0)

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
            f"History reconciliation completed (removed {removed_messages} "
            "message(s))."
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

        if agent_obj is None or not getattr(agent_obj, "_last_turn_cancelled", False):
            return

        _clear_current_task_cancellation_requests()
        clear_progress_for_agent(agent_name)
        reason = getattr(agent_obj, "_last_turn_cancel_reason", "cancelled")
        setattr(agent_obj, "_last_turn_cancelled", False)
        history_state = getattr(agent_obj, "_last_turn_history_state", None)
        setattr(agent_obj, "_last_turn_history_state", None)
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
        try:
            progress_display.clear_agent_tasks(agent_name)
        except Exception:
            pass

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
            history = getattr(agent_obj, "message_history", [])
            last_message = history[-1] if history else None
            return bool(
                last_message is not None
                and getattr(last_message, "role", None) == "assistant"
                and getattr(last_message, "stop_reason", None) == LlmStopReason.CANCELLED
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

        count = len(startup_warnings)
        if count == 1:
            rich_print("[yellow]Startup warning:[/yellow]")
            rich_print(f"  {startup_warnings[0]}")
            return

        rich_print(f"[yellow]Startup warnings ({count}):[/yellow]")
        for warning in startup_warnings:
            rich_print(f"  • {warning}")

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

        runtime_agents = prompt_provider.registered_agents()
        if not isinstance(runtime_agents, dict):
            return

        issues = collect_shell_cwd_issues_from_runtime_agents(runtime_agents, cwd=Path.cwd())
        if not issues:
            return

        issue_word = "issue" if len(issues) == 1 else "issues"
        rich_print(f"[yellow]Shell cwd startup check:[/yellow] {len(issues)} {issue_word} found.")

        selection = await get_selection_input(
            "Create missing shell directories now? [Y/n] ",
            default="y",
            allow_cancel=False,
            complete_options=False,
        )
        answer = (selection or "").strip().lower()
        if answer not in {"", "y", "yes"}:
            return

        created_paths, creation_errors = create_missing_shell_cwd_directories(issues)
        if created_paths:
            rich_print("[green]Created missing shell cwd directories:[/green]")
            for path in created_paths:
                rich_print(f"  • {path}")

        if creation_errors:
            rich_print("[red]Failed to create one or more shell cwd directories:[/red]")
            for item in creation_errors:
                rich_print(f"  • {item.path}: {item.message}")

        remaining_issues = collect_shell_cwd_issues_from_runtime_agents(
            runtime_agents,
            cwd=Path.cwd(),
        )
        if not remaining_issues:
            try:
                from fast_agent.ui import notification_tracker

                notification_tracker.remove_startup_warnings_containing("shell cwd")
            except Exception:
                pass

    def _apply_dispatch_result(
        self,
        *,
        state: PromptLoopAgents,
        dispatch_result: Any,
        buffer_prefill: str,
    ) -> tuple[PromptLoopAgents, PendingCommandExecution, str, bool]:
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
        return next_state, pending, next_buffer_prefill, should_continue

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
        noenv_mode = bool(getattr(prompt_provider, "_noenv_mode", False))
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

            agent_state, pending, buffer_prefill, should_continue = self._apply_dispatch_result(
                state=agent_state,
                dispatch_result=dispatch_result,
                buffer_prefill=buffer_prefill,
            )
            if dispatch_result.should_return:
                return PromptCommandPhase(
                    agent_state=agent_state,
                    pending=pending,
                    buffer_prefill=buffer_prefill,
                    should_return=True,
                )
            if should_continue:
                return PromptCommandPhase(
                    agent_state=agent_state,
                    pending=pending,
                    buffer_prefill=buffer_prefill,
                    should_continue=True,
                )

        if not pending.has_pending_execution() and isinstance(user_input, str):
            if user_input.strip().upper() == "STOP":
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
    ) -> tuple[PromptLoopResult, str | None, bool]:
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
            return current_result, hash_send_execution.buffer_prefill, True

        if pending.shell_execute_cmd:
            emit_prompt_mark("C")
            result = run_interactive_shell_command(pending.shell_execute_cmd)
            emit_prompt_mark("D")

            if result.output.strip():
                set_last_copyable_output(result.output.rstrip())

            if result.return_code != 0:
                display.show_shell_exit_code(result.return_code)

            return result, None, True

        return current_result, None, False

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
            rich_print(f"[yellow]{warning}[/yellow]")

        if not parsed_mentions.mentions:
            return prompt_payload

        try:
            agent_for_mentions = prompt_provider._agent(agent_name)
        except Exception:
            rich_print(f"[red]Unable to resolve resource mentions: agent '{agent_name}' unavailable[/red]")
            return user_input

        try:
            resolved_mentions = await resolve_mentions(agent_for_mentions, parsed_mentions)
            return build_prompt_with_resources(user_input, resolved_mentions)
        except Exception as exc:
            rich_print(f"[red]Failed to resolve resource mentions: {exc}[/red]")
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
        configured_shell_cwd_policy = getattr(
            getattr(get_settings(), "shell_execution", None),
            "missing_cwd_policy",
            None,
        )
        resolved_shell_cwd_policy = resolve_missing_shell_cwd_policy(
            cli_override=getattr(prompt_provider, "_missing_shell_cwd_policy_override", None),
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
            if turn_preparation.should_continue or turn_preparation.agent_state is None:
                continue
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
            if (
                input_phase.should_continue
                or input_phase.user_input is None
                or input_phase.agent_state is None
            ):
                continue

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

            result, hash_buffer_prefill, handled_pending = await self._handle_pending_execution(
                pending=command_phase.pending,
                send_func=send_func,
                quiet_send_func=quiet_send_func,
                prompt_provider=prompt_provider,
                display=display,
                current_result=result,
                runtime_state=runtime_state,
            )
            if handled_pending:
                if hash_buffer_prefill:
                    buffer_prefill = hash_buffer_prefill
                continue

            # Send the message to the agent
            # Type narrowing: by this point user_input is str (non-str inputs continue above)
            assert isinstance(user_input, str)
            prompt_payload = await self._resolve_prompt_payload(
                prompt_provider=prompt_provider,
                agent_name=agent_state.current_agent,
                user_input=user_input,
            )
            if prompt_payload is None:
                continue

            send_result = await self._send_regular_message(
                send_func=send_func,
                prompt_payload=prompt_payload,
                prompt_provider=prompt_provider,
                agent_name=agent_state.current_agent,
                runtime_state=runtime_state,
            )
            if send_result is None:
                continue
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
            rich_print(f"[dim]Asking {target_agent}...[/dim]")

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
            rich_print(f"[red]Error asking {target_agent}: {exc}[/red]")
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
                rich_print(f"[blue]Response from {target_agent} loaded into input buffer[/blue]")
            return HashSendExecution(buffer_prefill=response_text)

        if quiet:
            rich_print(f"[dim]No response received from {target_agent}[/dim]")
        else:
            rich_print(f"[yellow]No response received from {target_agent}[/yellow]")
        return HashSendExecution(buffer_prefill=None)

    def _resolve_display(
        self, prompt_provider: "AgentApp", agent_name: str | None
    ) -> "ConsoleDisplay":
        from fast_agent.ui.console_display import ConsoleDisplay

        agent = None
        if agent_name and hasattr(prompt_provider, "_agent"):
            try:
                agent = prompt_provider._agent(agent_name)
            except Exception:
                agent = None

        display = getattr(agent, "display", None) if agent is not None else None
        if display is not None:
            return display

        config = None
        if agent is not None:
            agent_context = getattr(agent, "context", None)
            config = getattr(agent_context, "config", None) if agent_context else None

        if config is None:
            config = get_settings()

        return ConsoleDisplay(config=config)
