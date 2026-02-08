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

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union, cast

from mcp.types import PromptMessage
from rich import print as rich_print

if TYPE_CHECKING:
    from fast_agent.commands.context import AgentProvider
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.console_display import ConsoleDisplay

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import agent_cards as agent_card_handlers
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.commands.results import CommandOutcome
from fast_agent.config import get_settings
from fast_agent.types import PromptMessageExtended
from fast_agent.ui import enhanced_prompt
from fast_agent.ui.adapters import TuiCommandIO
from fast_agent.ui.command_payloads import (
    AgentCommand,
    ClearCommand,
    ClearSessionsCommand,
    CommandPayload,
    CreateSessionCommand,
    ForkSessionCommand,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryReviewCommand,
    HistoryRewindCommand,
    ListPromptsCommand,
    ListSessionsCommand,
    ListSkillsCommand,
    ListToolsCommand,
    LoadAgentCardCommand,
    LoadPromptCommand,
    ModelReasoningCommand,
    ModelVerbosityCommand,
    PinSessionCommand,
    ReloadAgentsCommand,
    ResumeSessionCommand,
    SelectPromptCommand,
    ShellCommand,
    ShowHistoryCommand,
    ShowMarkdownCommand,
    ShowMcpStatusCommand,
    ShowSystemCommand,
    ShowUsageCommand,
    SkillsCommand,
    SwitchAgentCommand,
    TitleSessionCommand,
    UnknownCommand,
    is_command_payload,
)
from fast_agent.ui.enhanced_prompt import (
    _display_agent_info_helper,
    get_enhanced_input,
    handle_special_commands,
    parse_special_input,
    set_last_copyable_output,
)
from fast_agent.ui.interactive_shell import run_interactive_shell_command
from fast_agent.ui.progress_display import progress_display
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

    def _build_command_context(
        self, prompt_provider: "AgentApp", agent_name: str
    ) -> CommandContext:
        settings = get_settings()
        noenv_mode = bool(getattr(prompt_provider, "_noenv_mode", False))
        io = TuiCommandIO(
            prompt_provider=cast("AgentProvider", prompt_provider),
            agent_name=agent_name,
            settings=settings,
        )
        return CommandContext(
            agent_provider=cast("AgentProvider", prompt_provider),
            current_agent_name=agent_name,
            io=io,
            settings=settings,
            noenv=noenv_mode,
        )

    async def _emit_command_outcome(self, context: CommandContext, outcome: CommandOutcome) -> None:
        for message in outcome.messages:
            await context.io.emit(message)

    async def _get_all_prompts(
        self,
        prompt_provider: "AgentApp",
        agent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        target_agent = agent_name
        if not target_agent:
            agent_names = list(prompt_provider.agent_names())
            target_agent = agent_names[0] if agent_names else ""
        context = self._build_command_context(prompt_provider, target_agent)
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
        while True:
            # Variables for hash command - sent after input handling
            hash_send_target: str | None = None
            hash_send_message: str | None = None
            # Variable for shell command - executed after input handling
            shell_execute_cmd: str | None = None

            progress_display.pause()

            refreshed = await prompt_provider.refresh_if_needed()
            try:
                agent_obj = prompt_provider._agent(agent)
            except Exception:
                agent_obj = None

            if agent_obj is not None and getattr(agent_obj, "_last_turn_cancelled", False):
                reason = getattr(agent_obj, "_last_turn_cancel_reason", "cancelled")
                setattr(agent_obj, "_last_turn_cancelled", False)
                rich_print(
                    "[yellow]Previous turn was {reason}. If the session now shows a pending tool call, "
                    "run /history fix or /history clear all.[/yellow]".format(reason=reason)
                )

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

            # Use the enhanced input method with advanced features
            noenv_mode = bool(getattr(prompt_provider, "_noenv_mode", False))
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
            buffer_prefill = ""  # Clear after use - it's one-off

            if isinstance(user_input, str):
                user_input = parse_special_input(user_input)

            # Avoid blocking quick shell commands on agent refresh.
            skip_refresh = isinstance(user_input, ShellCommand)
            if not skip_refresh:
                refreshed = await prompt_provider.refresh_if_needed()
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
                match command_payload:
                    case SwitchAgentCommand(agent_name=new_agent):
                        if new_agent in available_agents_set:
                            agent = new_agent
                            # Display new agent info immediately when switching
                            rich_print()  # Add spacing
                            await _display_agent_info_helper(agent, prompt_provider)
                            continue
                        rich_print(f"[red]Agent '{new_agent}' not found[/red]")
                        continue
                    case HashAgentCommand(agent_name=target_agent, message=hash_message):
                        # Validate, but send after input handling.
                        if target_agent not in available_agents_set:
                            rich_print(f"[red]Agent '{target_agent}' not found[/red]")
                            continue
                        if not hash_message:
                            rich_print(f"[yellow]Usage: #{target_agent} <message>[/yellow]")
                            continue
                        # Set up for sending outside the paused context
                        hash_send_target = target_agent
                        hash_send_message = hash_message
                        # Don't continue here - fall through to execution
                    case ShellCommand(command=shell_cmd):
                        # Store for execution after input handling
                        shell_execute_cmd = shell_cmd
                        # Don't continue here - fall through to execution
                    # Keep the existing list_prompts handler for backward compatibility
                    case ListPromptsCommand():
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await prompt_handlers.handle_list_prompts(
                            context,
                            agent_name=agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case SelectPromptCommand(prompt_name=prompt_name, prompt_index=prompt_index):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await prompt_handlers.handle_select_prompt(
                            context,
                            agent_name=agent,
                            requested_name=prompt_name,
                            prompt_index=prompt_index,
                        )
                        await self._emit_command_outcome(context, outcome)
                        if outcome.buffer_prefill:
                            buffer_prefill = outcome.buffer_prefill
                        continue
                    case LoadPromptCommand(filename=filename, error=error):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await prompt_handlers.handle_load_prompt(
                            context,
                            agent_name=agent,
                            filename=filename,
                            error=error,
                        )
                        await self._emit_command_outcome(context, outcome)
                        if outcome.buffer_prefill:
                            buffer_prefill = outcome.buffer_prefill
                        continue
                    case ListToolsCommand():
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await tools_handlers.handle_list_tools(
                            context,
                            agent_name=agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ListSkillsCommand():
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await skills_handlers.handle_list_skills(
                            context,
                            agent_name=agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case SkillsCommand(action=action, argument=argument):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await skills_handlers.handle_skills_command(
                            context,
                            agent_name=agent,
                            action=action,
                            argument=argument,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ShowUsageCommand():
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await display_handlers.handle_show_usage(
                            context,
                            agent_name=agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ShowHistoryCommand(agent=target_agent):
                        if (
                            target_agent
                            and self._get_agent_or_warn(prompt_provider, target_agent) is None
                        ):
                            continue
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await history_handlers.handle_show_history(
                            context,
                            agent_name=agent,
                            target_agent=target_agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case HistoryRewindCommand(turn_index=turn_index, error=error):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await history_handlers.handle_history_rewind(
                            context,
                            agent_name=agent,
                            turn_index=turn_index,
                            error=error,
                        )
                        await self._emit_command_outcome(context, outcome)
                        if outcome.buffer_prefill:
                            buffer_prefill = outcome.buffer_prefill
                        continue
                    case HistoryReviewCommand(turn_index=turn_index, error=error):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await history_handlers.handle_history_review(
                            context,
                            agent_name=agent,
                            turn_index=turn_index,
                            error=error,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case HistoryFixCommand(agent=target_agent):
                        if (
                            target_agent
                            and self._get_agent_or_warn(prompt_provider, target_agent) is None
                        ):
                            continue
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await history_handlers.handle_history_fix(
                            context,
                            agent_name=agent,
                            target_agent=target_agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ClearCommand(kind="clear_last", agent=target_agent):
                        if (
                            target_agent
                            and self._get_agent_or_warn(prompt_provider, target_agent) is None
                        ):
                            continue
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await history_handlers.handle_history_clear_last(
                            context,
                            agent_name=agent,
                            target_agent=target_agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ClearCommand(kind="clear_history", agent=target_agent):
                        if (
                            target_agent
                            and self._get_agent_or_warn(prompt_provider, target_agent) is None
                        ):
                            continue
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await history_handlers.handle_history_clear_all(
                            context,
                            agent_name=agent,
                            target_agent=target_agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ShowSystemCommand():
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await display_handlers.handle_show_system(
                            context,
                            agent_name=agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ShowMarkdownCommand():
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await display_handlers.handle_show_markdown(
                            context,
                            agent_name=agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ShowMcpStatusCommand():
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await display_handlers.handle_show_mcp_status(
                            context,
                            agent_name=agent,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ModelReasoningCommand(value=value):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await model_handlers.handle_model_reasoning(
                            context,
                            agent_name=agent,
                            value=value,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ModelVerbosityCommand(value=value):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await model_handlers.handle_model_verbosity(
                            context,
                            agent_name=agent,
                            value=value,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case CreateSessionCommand(session_name=session_name):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await sessions_handlers.handle_create_session(
                            context,
                            session_name=session_name,
                        )
                        # Clear agent histories for new session
                        cleared = clear_agent_histories(prompt_provider._agents)
                        if cleared:
                            cleared_list = ", ".join(sorted(cleared))
                            outcome.add_message(
                                f"Cleared agent history: {cleared_list}",
                                channel="info",
                            )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ListSessionsCommand(show_help=show_help):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await sessions_handlers.handle_list_sessions(
                            context,
                            show_help=show_help,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ClearSessionsCommand(target=target):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await sessions_handlers.handle_clear_sessions(
                            context,
                            target=target,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case PinSessionCommand(value=value, target=target):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await sessions_handlers.handle_pin_session(
                            context,
                            value=value,
                            target=target,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ResumeSessionCommand(session_id=session_id):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await sessions_handlers.handle_resume_session(
                            context,
                            agent_name=agent,
                            session_id=session_id,
                        )
                        await self._emit_command_outcome(context, outcome)
                        if outcome.switch_agent:
                            agent = outcome.switch_agent
                        continue
                    case TitleSessionCommand(title=title):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await sessions_handlers.handle_title_session(
                            context,
                            title=title,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ForkSessionCommand(title=title):
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await sessions_handlers.handle_fork_session(
                            context,
                            title=title,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue

                    case LoadAgentCardCommand(
                        filename=filename,
                        add_tool=add_tool,
                        remove_tool=remove_tool,
                        error=error,
                    ):
                        if error:
                            rich_print(f"[red]{error}[/red]")
                            continue

                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await agent_card_handlers.handle_card_load(
                            context,
                            manager=prompt_provider,
                            filename=filename,
                            add_tool=add_tool,
                            remove_tool=remove_tool,
                            current_agent=agent,
                        )
                        await self._emit_command_outcome(context, outcome)

                        if outcome.requires_refresh:
                            available_agents = _merge_pinned_agents(
                                list(prompt_provider.agent_names())
                            )
                            available_agents_set = set(available_agents)
                            self.agent_types = prompt_provider.agent_types()

                            if agent not in available_agents_set:
                                if available_agents:
                                    agent = available_agents[0]
                                else:
                                    rich_print("[red]No agents available after load.[/red]")
                                    return result
                        continue
                    case AgentCommand(
                        agent_name=agent_name,
                        add_tool=add_tool,
                        remove_tool=remove_tool,
                        dump=dump,
                        error=error,
                    ):
                        if error:
                            rich_print(f"[red]{error}[/red]")
                            continue

                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await agent_card_handlers.handle_agent_command(
                            context,
                            manager=prompt_provider,
                            current_agent=agent,
                            target_agent=agent_name,
                            add_tool=add_tool,
                            remove_tool=remove_tool,
                            dump=dump,
                        )
                        await self._emit_command_outcome(context, outcome)
                        continue
                    case ReloadAgentsCommand():
                        context = self._build_command_context(prompt_provider, agent)
                        outcome = await agent_card_handlers.handle_reload_agents(
                            context,
                            manager=prompt_provider,
                        )
                        await self._emit_command_outcome(context, outcome)

                        if outcome.requires_refresh:
                            available_agents = _merge_pinned_agents(
                                list(prompt_provider.agent_names())
                            )
                            available_agents_set = set(available_agents)
                            self.agent_types = prompt_provider.agent_types()

                            if agent not in available_agents_set:
                                if available_agents:
                                    agent = available_agents[0]
                                else:
                                    rich_print("[red]No agents available after reload.[/red]")
                                    return result
                        continue
                    case UnknownCommand(command=command):
                        rich_print(f"[red]Command not found: {command}[/red]")
                        continue
                    case _:
                        pass

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

                if user_input.upper() == "STOP":
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
                except Exception as exc:
                    rich_print(f"[red]Error asking {hash_send_target}: {exc}[/red]")
                    continue
                finally:
                    progress_display.pause()
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
            emit_prompt_mark("C")
            progress_display.resume()
            try:
                result = await send_func(user_input, agent)
            finally:
                progress_display.pause()
                emit_prompt_mark("D")

            if result and result.startswith("▲ **System Error:**"):
                # rich_print(result)
                print(result)

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
