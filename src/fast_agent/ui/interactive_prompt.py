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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union, cast

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
from fast_agent.ui.enhanced_prompt import (
    get_enhanced_input,
    get_selection_input,
    handle_special_commands,
    parse_special_input,
    set_last_copyable_output,
)
from fast_agent.ui.interactive_shell import run_interactive_shell_command
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.prompt.resource_mentions import (
    build_prompt_with_resources,
    parse_mentions,
    resolve_mentions,
)
from fast_agent.ui.prompt_marks import emit_prompt_mark

# Type alias for the send function
SendFunc = Callable[[Union[str, PromptMessage, PromptMessageExtended], str], Awaitable[str]]

# Type alias for the agent getter function
AgentGetter = Callable[[str], object | None]


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

        target_agent = agent_name
        if not target_agent:
            agent_names = list(prompt_provider.agent_names())
            target_agent = agent_names[0] if agent_names else ""
        context = build_command_context(prompt_provider, target_agent)
        return await prompt_handlers._get_all_prompts(context, agent_name)

    async def prompt_loop(
        self,
        send_func: SendFunc,
        default_agent: str,
        available_agents: list[str],
        prompt_provider: "AgentApp",
        pinned_agent: str | None = None,
        default: str = "",
    ) -> str:
        """
        Start an interactive prompt session.

        Args:
            send_func: Function to send messages to agents
            default_agent: Name of the default agent to use
            available_agents: List of available agent names
            prompt_provider: AgentApp instance for accessing agents and prompts
            pinned_agent: Explicitly targeted agent name to preserve across refreshes
            default: Default message to use when user presses enter

        Returns:
            The result of the interactive session
        """
        def _merge_pinned_agents(agent_names: list[str]) -> list[str]:
            if not pinned_agent or pinned_agent in agent_names:
                return agent_names
            try:
                agent_types = prompt_provider.agent_types()
            except Exception:
                return agent_names
            if pinned_agent in agent_types:
                return [pinned_agent, *agent_names]
            return agent_names

        agent = default_agent
        if not agent:
            if available_agents:
                agent = available_agents[0]
            else:
                raise ValueError("No default agent available")

        available_agents = _merge_pinned_agents(list(available_agents))
        if agent not in available_agents:
            raise ValueError(f"No agent named '{agent}'")

        # Ensure we track available agents in a set for fast lookup
        available_agents_set = set(available_agents)

        from fast_agent.ui.console_display import ConsoleDisplay

        display = ConsoleDisplay(config=get_settings())

        result = ""
        buffer_prefill = ""  # One-off buffer content for # command results
        ctrl_c_exit_window_seconds = 2.0
        ctrl_c_deadline: float | None = None
        startup_warning_digest_checked = False
        shell_cwd_startup_prompt_checked = False
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
                message=None,
                prompt_file=None,
                stdin_is_tty=sys.stdin.isatty(),
                tty_device_available=False,
            ),
        )

        def _handle_ctrl_c_interrupt() -> None:
            nonlocal ctrl_c_deadline

            now = time.monotonic()
            if ctrl_c_deadline is not None and now <= ctrl_c_deadline:
                rich_print("[red]Second Ctrl+C received; exiting fast-agent session.[/red]")
                raise PromptExitError("User requested to exit fast-agent session")

            ctrl_c_deadline = now + ctrl_c_exit_window_seconds
            rich_print(
                "[yellow]Interrupted operation; returning to prompt. "
                "Press Ctrl+C again within 2 seconds to exit.[/yellow]"
            )

        def _clear_ctrl_c_interrupt() -> None:
            nonlocal ctrl_c_deadline

            ctrl_c_deadline = None

        def _handle_inflight_cancel() -> None:
            """Handle Ctrl+C/Cancel while generation or tool calling is active."""
            nonlocal ctrl_c_deadline

            ctrl_c_deadline = None
            rich_print("[yellow]Generation cancelled by user.[/yellow]")

        def _emit_startup_warning_digest_once() -> None:
            nonlocal startup_warning_digest_checked

            if startup_warning_digest_checked:
                return
            startup_warning_digest_checked = True

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

        async def _maybe_prompt_for_shell_cwd_startup_once() -> None:
            nonlocal shell_cwd_startup_prompt_checked

            if shell_cwd_startup_prompt_checked:
                return
            shell_cwd_startup_prompt_checked = True

            if shell_cwd_policy != "ask":
                return

            runtime_agents = getattr(prompt_provider, "_agents", None)
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

        def _auto_fix_pending_tool_call(agent_obj: Any) -> bool:
            """Remove a dangling assistant TOOL_USE message at end of history."""
            history = getattr(agent_obj, "message_history", None)
            if not isinstance(history, list) or not history:
                return False

            last_message = history[-1]
            if not (
                getattr(last_message, "role", None) == "assistant"
                and getattr(last_message, "tool_calls", None)
                and getattr(last_message, "stop_reason", None) == LlmStopReason.TOOL_USE
            ):
                return False

            trimmed_history = history[:-1]
            load_history = getattr(agent_obj, "load_message_history", None)
            if callable(load_history):
                load_history(trimmed_history)
                return True

            existing_history = getattr(agent_obj, "message_history", None)
            if isinstance(existing_history, list):
                existing_history.clear()
                existing_history.extend(trimmed_history)
                return True

            return False

        while True:
            # Variables for hash command - sent after input handling
            hash_send_target: str | None = None
            hash_send_message: str | None = None
            # Variable for shell command - executed after input handling
            shell_execute_cmd: str | None = None

            progress_display.pause(cancel_deferred_on_noop=True)

            try:
                refreshed = await prompt_provider.refresh_if_needed()
            except KeyboardInterrupt:
                _handle_ctrl_c_interrupt()
                continue
            try:
                agent_obj = prompt_provider._agent(agent)
            except Exception:
                agent_obj = None

            if agent_obj is not None and getattr(agent_obj, "_last_turn_cancelled", False):
                reason = getattr(agent_obj, "_last_turn_cancel_reason", "cancelled")
                setattr(agent_obj, "_last_turn_cancelled", False)
                pending_tool_call_removed = False
                try:
                    pending_tool_call_removed = _auto_fix_pending_tool_call(agent_obj)
                except Exception:
                    pending_tool_call_removed = False

                if pending_tool_call_removed:
                    rich_print(
                        "[yellow]Previous turn was {reason}. Removed pending tool call from history.[/yellow]".format(
                            reason=reason
                        )
                    )
                else:
                    rich_print("[yellow]Previous turn was {reason}.[/yellow]".format(reason=reason))

            if refreshed:
                available_agents = _merge_pinned_agents(list(prompt_provider.agent_names()))
                available_agents_set = set(available_agents)
                self.agent_types = prompt_provider.agent_types()
                enhanced_prompt.available_agents = set(available_agents)

                if agent not in available_agents_set:
                    if available_agents:
                        agent = available_agents[0]
                    else:
                        rich_print("[red]No agents available after refresh.[/red]")
                        return result

                rich_print("[green]AgentCards reloaded.[/green]")

            current_agents = _merge_pinned_agents(list(prompt_provider.agent_names()))
            if current_agents and set(current_agents) != available_agents_set:
                available_agents = current_agents
                available_agents_set = set(available_agents)
                enhanced_prompt.available_agents = set(available_agents)
            if agent not in available_agents_set:
                if available_agents:
                    agent = available_agents[0]
                else:
                    rich_print("[red]No agents available.[/red]")
                    return result

            await _maybe_prompt_for_shell_cwd_startup_once()
            _emit_startup_warning_digest_once()

            # Use the enhanced input method with advanced features
            noenv_mode = bool(getattr(prompt_provider, "_noenv_mode", False))
            try:
                user_input = await get_enhanced_input(
                    agent_name=agent,
                    default=default,
                    show_default=(default != ""),
                    show_stop_hint=True,
                    multiline=False,  # Default to single-line mode
                    available_agent_names=available_agents,
                    agent_types=self.agent_types,  # Pass agent types for display
                    agent_provider=prompt_provider,  # Pass agent provider for info display
                    noenv_mode=noenv_mode,
                    pre_populate_buffer=buffer_prefill,  # One-off buffer content
                )
            except KeyboardInterrupt:
                _handle_ctrl_c_interrupt()
                continue
            buffer_prefill = ""  # Clear after use - it's one-off

            if isinstance(user_input, str):
                user_input = parse_special_input(user_input)

            if not isinstance(user_input, InterruptCommand):
                _clear_ctrl_c_interrupt()

            # Avoid blocking quick shell commands on agent refresh.
            skip_refresh = isinstance(user_input, ShellCommand)
            if not skip_refresh:
                try:
                    refreshed = await prompt_provider.refresh_if_needed()
                except KeyboardInterrupt:
                    _handle_ctrl_c_interrupt()
                    continue
                if refreshed:
                    available_agents = _merge_pinned_agents(list(prompt_provider.agent_names()))
                    available_agents_set = set(available_agents)
                    self.agent_types = prompt_provider.agent_types()
                    enhanced_prompt.available_agents = set(available_agents)

                    if agent not in available_agents_set:
                        if available_agents:
                            agent = available_agents[0]
                        else:
                            rich_print("[red]No agents available after refresh.[/red]")
                            return result

                    rich_print("[green]AgentCards reloaded.[/green]")

            # Handle special commands with access to the agent provider
            command_result = await handle_special_commands(user_input, prompt_provider)

            # Check if we should switch agents
            if is_command_payload(command_result):
                command_payload: CommandPayload = cast("CommandPayload", command_result)
                from fast_agent.ui.interactive.command_dispatch import dispatch_command_payload

                try:
                    dispatch_result = await dispatch_command_payload(
                        self,
                        command_payload,
                        prompt_provider=prompt_provider,
                        agent=agent,
                        available_agents=available_agents,
                        available_agents_set=available_agents_set,
                        merge_pinned_agents=_merge_pinned_agents,
                    )
                except KeyboardInterrupt:
                    _handle_ctrl_c_interrupt()
                    continue

                if dispatch_result.available_agents is not None:
                    available_agents = dispatch_result.available_agents
                if dispatch_result.available_agents_set is not None:
                    available_agents_set = dispatch_result.available_agents_set

                if dispatch_result.next_agent is not None:
                    agent = dispatch_result.next_agent

                if dispatch_result.buffer_prefill:
                    buffer_prefill = dispatch_result.buffer_prefill

                if dispatch_result.hash_send_target and dispatch_result.hash_send_message:
                    hash_send_target = dispatch_result.hash_send_target
                    hash_send_message = dispatch_result.hash_send_message

                if dispatch_result.shell_execute_cmd:
                    shell_execute_cmd = dispatch_result.shell_execute_cmd

                if dispatch_result.should_return:
                    return result

                if dispatch_result.handled and not hash_send_target and not shell_execute_cmd:
                    continue

            # Skip further processing if:
            # 1. The command was handled (command_result is truthy)
            # 2. The original input was a command payload (special command like /prompt)
            # 3. The command result itself is a command payload (special command handling result)
            # This fixes the issue where /prompt without arguments gets sent to the LLM
            # Skip these checks if we have a pending hash or shell command to handle outside
            if not hash_send_target and not shell_execute_cmd:
                if (
                    command_result
                    or is_command_payload(user_input)
                    or is_command_payload(command_result)
                ):
                    continue

                if not isinstance(user_input, str):
                    continue

                if user_input.strip().upper() == "STOP":
                    return result

                if user_input == "":
                    continue

            # Handle hash command after input handling; resume progress display only for the send.
            if hash_send_target and hash_send_message:
                rich_print(f"[dim]Asking {hash_send_target}...[/dim]")

                try:
                    # Use the return value from send_func directly - this works even
                    # when use_history=False (e.g., for agents loaded as tools)
                    emit_prompt_mark("C")
                    progress_display.resume()
                    response_text = await send_func(hash_send_message, hash_send_target)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    _handle_inflight_cancel()
                    continue
                except Exception as exc:
                    rich_print(f"[red]Error asking {hash_send_target}: {exc}[/red]")
                    continue
                finally:
                    progress_display.pause(cancel_deferred_on_noop=True)
                    emit_prompt_mark("D")

                # Status messages after send completes
                if response_text:
                    buffer_prefill = response_text
                    rich_print(
                        f"[blue]Response from {hash_send_target} loaded into input buffer[/blue]"
                    )
                else:
                    rich_print(f"[yellow]No response received from {hash_send_target}[/yellow]")
                continue

            # Handle shell command after input handling
            if shell_execute_cmd:
                emit_prompt_mark("C")
                result = run_interactive_shell_command(shell_execute_cmd)
                emit_prompt_mark("D")

                if result.output.strip():
                    set_last_copyable_output(result.output.rstrip())

                if result.return_code != 0:
                    display.show_shell_exit_code(result.return_code)

                shell_execute_cmd = None
                continue

            # Send the message to the agent
            # Type narrowing: by this point user_input is str (non-str inputs continue above)
            assert isinstance(user_input, str)

            prompt_payload: str | PromptMessageExtended = user_input
            parsed_mentions = parse_mentions(user_input)
            for warning in parsed_mentions.warnings:
                rich_print(f"[yellow]{warning}[/yellow]")

            if parsed_mentions.mentions:
                try:
                    agent_for_mentions = prompt_provider._agent(agent)
                except Exception:
                    rich_print(
                        f"[red]Unable to resolve resource mentions: agent '{agent}' unavailable[/red]"
                    )
                    continue

                try:
                    resolved_mentions = await resolve_mentions(agent_for_mentions, parsed_mentions)
                    prompt_payload = build_prompt_with_resources(user_input, resolved_mentions)
                except Exception as exc:
                    rich_print(f"[red]Failed to resolve resource mentions: {exc}[/red]")
                    continue

            emit_prompt_mark("C")
            progress_display.resume()
            try:
                result = await send_func(prompt_payload, agent)
            except (KeyboardInterrupt, asyncio.CancelledError):
                _handle_inflight_cancel()
                continue
            finally:
                progress_display.pause(cancel_deferred_on_noop=True)
                emit_prompt_mark("D")

            if result and result.startswith("▲ **System Error:**"):
                # rich_print(result)
                print(result)

            try:
                agent_after_send = prompt_provider._agent(agent)
            except Exception:
                agent_after_send = None

            if agent_after_send is not None:
                try:
                    history = getattr(agent_after_send, "message_history", [])
                    last_message = history[-1] if history else None
                    if (
                        last_message is not None
                        and getattr(last_message, "role", None) == "assistant"
                        and getattr(last_message, "stop_reason", None) == LlmStopReason.CANCELLED
                    ):
                        ctrl_c_deadline = None
                except Exception:
                    pass

            # Update last copyable output with assistant response for Ctrl+Y
            if result:
                set_last_copyable_output(result)

        return result

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
