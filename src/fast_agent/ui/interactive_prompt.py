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

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union, cast

from fast_agent.constants import CONTROL_MESSAGE_SAVE_HISTORY

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
from mcp.types import Prompt, PromptMessage
from rich import print as rich_print

from fast_agent.agents.agent_types import AgentType
from fast_agent.config import get_settings
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.mcp.mcp_aggregator import SEP
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.skills.manager import (
    DEFAULT_SKILL_REGISTRIES,
    fetch_marketplace_skills,
    fetch_marketplace_skills_with_source,
    format_marketplace_display_url,
    get_manager_directory,
    get_marketplace_url,
    install_marketplace_skill,
    list_local_skills,
    reload_skill_manifests,
    remove_local_skill,
    resolve_skill_directories,
    select_manifest_by_name_or_index,
    select_skill_by_name_or_index,
)
from fast_agent.skills.registry import SkillManifest, format_skills_for_prompt
from fast_agent.types import PromptMessageExtended
from fast_agent.ui import enhanced_prompt
from fast_agent.ui.command_payloads import (
    AgentCommand,
    ClearCommand,
    CommandPayload,
    HashAgentCommand,
    ListPromptsCommand,
    ListSkillsCommand,
    ListToolsCommand,
    LoadAgentCardCommand,
    LoadHistoryCommand,
    ReloadAgentsCommand,
    SaveHistoryCommand,
    SelectPromptCommand,
    ShowHistoryCommand,
    ShowMarkdownCommand,
    ShowMcpStatusCommand,
    ShowSystemCommand,
    ShowUsageCommand,
    SkillsCommand,
    SwitchAgentCommand,
    is_command_payload,
)
from fast_agent.ui.enhanced_prompt import (
    _display_agent_info_helper,
    get_argument_input,
    get_enhanced_input,
    get_selection_input,
    handle_special_commands,
    parse_special_input,
    show_mcp_status,
)
from fast_agent.ui.history_display import display_history_overview
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.usage_display import collect_agents_from_provider, display_usage_report

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

    async def prompt_loop(
        self,
        send_func: SendFunc,
        default_agent: str,
        available_agents: list[str],
        prompt_provider: "AgentApp",
        default: str = "",
    ) -> str:
        """
        Start an interactive prompt session.

        Args:
            send_func: Function to send messages to agents
            default_agent: Name of the default agent to use
            available_agents: List of available agent names
            prompt_provider: AgentApp instance for accessing agents and prompts
            default: Default message to use when user presses enter

        Returns:
            The result of the interactive session
        """
        agent = default_agent
        if not agent:
            if available_agents:
                agent = available_agents[0]
            else:
                raise ValueError("No default agent available")

        if agent not in available_agents:
            raise ValueError(f"No agent named '{agent}'")

        # Ensure we track available agents in a set for fast lookup
        available_agents_set = set(available_agents)

        result = ""
        buffer_prefill = ""  # One-off buffer content for # command results
        while True:
            # Variables for hash command - must be sent OUTSIDE paused context
            hash_send_target: str | None = None
            hash_send_message: str | None = None

            refreshed = await prompt_provider.refresh_if_needed()
            if refreshed:
                available_agents = list(prompt_provider.agent_names())
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

            current_agents = list(prompt_provider.agent_names())
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

            with progress_display.paused():
                # Use the enhanced input method with advanced features
                user_input = await get_enhanced_input(
                    agent_name=agent,
                    default=default,
                    show_default=(default != ""),
                    show_stop_hint=True,
                    multiline=False,  # Default to single-line mode
                    available_agent_names=available_agents,
                    agent_types=self.agent_types,  # Pass agent types for display
                    agent_provider=prompt_provider,  # Pass agent provider for info display
                    pre_populate_buffer=buffer_prefill,  # One-off buffer content
                )
                buffer_prefill = ""  # Clear after use - it's one-off

                if isinstance(user_input, str):
                    user_input = parse_special_input(user_input)

                refreshed = await prompt_provider.refresh_if_needed()
                if refreshed:
                    available_agents = list(prompt_provider.agent_names())
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
                            # Validate, but send OUTSIDE paused context to avoid
                            # nested paused() issues with progress display
                            if target_agent not in available_agents_set:
                                rich_print(f"[red]Agent '{target_agent}' not found[/red]")
                                continue
                            if not hash_message:
                                rich_print(
                                    f"[yellow]Usage: #{target_agent} <message>[/yellow]"
                                )
                                continue
                            # Set up for sending outside the paused context
                            hash_send_target = target_agent
                            hash_send_message = hash_message
                            # Don't continue here - fall through to exit paused context
                        # Keep the existing list_prompts handler for backward compatibility
                        case ListPromptsCommand():
                            # Use the prompt_provider directly
                            await self._list_prompts(prompt_provider, agent)
                            continue
                        case SelectPromptCommand(
                            prompt_name=prompt_name, prompt_index=prompt_index
                        ):
                            # Handle prompt selection, using both list_prompts and apply_prompt
                            # If a specific index was provided (from /prompt <number>)
                            if prompt_index is not None:
                                # First get a list of all prompts to look up the index
                                all_prompts = await self._get_all_prompts(prompt_provider, agent)
                                if not all_prompts:
                                    rich_print("[yellow]No prompts available[/yellow]")
                                    continue

                                # Check if the index is valid
                                if 1 <= prompt_index <= len(all_prompts):
                                    # Get the prompt at the specified index (1-based to 0-based)
                                    selected_prompt = all_prompts[prompt_index - 1]
                                    # Use the already created namespaced_name to ensure consistency
                                    await self._select_prompt(
                                        prompt_provider,
                                        agent,
                                        selected_prompt["namespaced_name"],
                                    )
                                else:
                                    rich_print(
                                        f"[red]Invalid prompt number: {prompt_index}. Valid range is 1-{len(all_prompts)}[/red]"
                                    )
                                    # Show the prompt list for convenience
                                    await self._list_prompts(prompt_provider, agent)
                            else:
                                # Use the name-based selection
                                await self._select_prompt(prompt_provider, agent, prompt_name)
                            continue
                        case ListToolsCommand():
                            # Handle tools list display
                            await self._list_tools(prompt_provider, agent)
                            continue
                        case ListSkillsCommand():
                            await self._list_skills(prompt_provider, agent)
                            continue
                        case SkillsCommand(action=action, argument=argument):
                            payload = {"action": action, "argument": argument}
                            await self._handle_skills_command(prompt_provider, agent, payload)
                            continue
                        case ShowUsageCommand():
                            # Handle usage display
                            await self._show_usage(prompt_provider, agent)
                            continue
                        case ShowHistoryCommand(agent=target_agent):
                            target_agent = target_agent or agent
                            try:
                                agent_obj = prompt_provider._agent(target_agent)
                            except Exception:
                                rich_print(f"[red]Unable to load agent '{target_agent}'[/red]")
                                continue

                            history = getattr(agent_obj, "message_history", [])
                            usage = getattr(agent_obj, "usage_accumulator", None)
                            display_history_overview(target_agent, history, usage)
                            continue
                        case ClearCommand(kind="clear_last", agent=target_agent):
                            target_agent = target_agent or agent
                            try:
                                agent_obj = prompt_provider._agent(target_agent)
                            except Exception:
                                rich_print(f"[red]Unable to load agent '{target_agent}'[/red]")
                                continue

                            removed_message = None
                            pop_callable = getattr(agent_obj, "pop_last_message", None)
                            if callable(pop_callable):
                                removed_message = pop_callable()
                            else:
                                history = getattr(agent_obj, "message_history", [])
                                if history:
                                    try:
                                        removed_message = history.pop()
                                    except Exception:
                                        removed_message = None

                            if removed_message:
                                role = getattr(removed_message, "role", "message")
                                role_display = (
                                    role.capitalize() if isinstance(role, str) else "Message"
                                )
                                rich_print(
                                    f"[green]Removed last {role_display} for agent '{target_agent}'.[/green]"
                                )
                            else:
                                rich_print(
                                    f"[yellow]No messages to remove for agent '{target_agent}'.[/yellow]"
                                )
                            continue
                        case ClearCommand(kind="clear_history", agent=target_agent):
                            target_agent = target_agent or agent
                            try:
                                agent_obj = prompt_provider._agent(target_agent)
                            except Exception:
                                rich_print(f"[red]Unable to load agent '{target_agent}'[/red]")
                                continue

                            if hasattr(agent_obj, "clear"):
                                try:
                                    agent_obj.clear()
                                    rich_print(
                                        f"[green]History cleared for agent '{target_agent}'.[/green]"
                                    )
                                except Exception as exc:
                                    rich_print(
                                        f"[red]Failed to clear history for '{target_agent}': {exc}[/red]"
                                    )
                            else:
                                rich_print(
                                    f"[yellow]Agent '{target_agent}' does not support clearing history.[/yellow]"
                                )
                            continue
                        case ShowSystemCommand():
                            # Handle system prompt display
                            await self._show_system(prompt_provider, agent)
                            continue
                        case ShowMarkdownCommand():
                            # Handle markdown display
                            await self._show_markdown(prompt_provider, agent)
                            continue
                        case ShowMcpStatusCommand():
                            rich_print()
                            await show_mcp_status(agent, prompt_provider)
                            continue
                        case SaveHistoryCommand(filename=filename):
                            # Save history for the current agent
                            try:
                                agent_obj = prompt_provider._agent(agent)

                                # Prefer type-safe exporter over magic string
                                saved_path = await HistoryExporter.save(agent_obj, filename)
                                rich_print(f"[green]History saved to {saved_path}[/green]")
                            except Exception:
                                # Fallback to magic string path for maximum compatibility
                                control = CONTROL_MESSAGE_SAVE_HISTORY + (
                                    f" {filename}" if filename else ""
                                )
                                result = await send_func(control, agent)
                                if result:
                                    rich_print(f"[green]{result}[/green]")
                            continue
                        case LoadHistoryCommand(filename=filename, error=error):
                            # Load history for the current agent
                            if error:
                                rich_print(f"[red]{error}[/red]")
                                continue

                            if filename is None:
                                rich_print("[red]Filename required for load_history[/red]")
                                continue

                            try:
                                from fast_agent.mcp.prompts.prompt_load import (
                                    load_history_into_agent,
                                )

                                # Get the agent object and its underlying LLM
                                agent_obj = prompt_provider._agent(agent)

                                # Load history directly without triggering LLM call
                                load_history_into_agent(agent_obj, Path(filename))

                                msg_count = len(agent_obj.message_history)
                                rich_print(
                                    f"[green]Loaded {msg_count} messages from {filename}[/green]"
                                )
                            except FileNotFoundError:
                                rich_print(f"[red]File not found: {filename}[/red]")
                            except Exception as e:
                                rich_print(f"[red]Error loading history: {e}[/red]")
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

                            if filename is None:
                                rich_print("[red]Filename required for /card[/red]")
                                continue

                            if not prompt_provider.can_load_agent_cards():
                                rich_print(
                                    "[yellow]AgentCard loading is not available in this session.[/yellow]"
                                )
                                continue

                            try:
                                if add_tool and not remove_tool:
                                    loaded_names, attached_names = (
                                        await prompt_provider.load_agent_card(
                                            filename, agent
                                        )
                                    )
                                else:
                                    loaded_names, attached_names = (
                                        await prompt_provider.load_agent_card(filename)
                                    )
                            except Exception as exc:
                                rich_print(f"[red]AgentCard load failed: {exc}[/red]")
                                continue

                            available_agents = list(prompt_provider.agent_names())
                            available_agents_set = set(available_agents)
                            self.agent_types = prompt_provider.agent_types()

                            if agent not in available_agents_set:
                                if available_agents:
                                    agent = available_agents[0]
                                else:
                                    rich_print("[red]No agents available after load.[/red]")
                                    return result

                            if not loaded_names:
                                rich_print("[green]AgentCard loaded.[/green]")
                            else:
                                name_list = ", ".join(loaded_names)
                                rich_print(f"[green]Loaded AgentCard(s): {name_list}[/green]")

                            if add_tool and remove_tool:
                                if not prompt_provider.can_detach_agent_tools():
                                    rich_print(
                                        "[yellow]Agent tool detachment is not available in this session.[/yellow]"
                                    )
                                    continue
                                try:
                                    removed = await prompt_provider.detach_agent_tools(
                                        agent, loaded_names
                                    )
                                except Exception as exc:
                                    rich_print(
                                        f"[red]Agent tool detach failed: {exc}[/red]"
                                    )
                                    continue
                                if removed:
                                    removed_list = ", ".join(removed)
                                    rich_print(
                                        f"[green]Detached agent tool(s): {removed_list}[/green]"
                                    )
                                else:
                                    rich_print(
                                        "[yellow]No agent tools detached.[/yellow]"
                                    )
                                continue
                            if add_tool:
                                if attached_names:
                                    attached_list = ", ".join(attached_names)
                                    rich_print(
                                        f"[green]Attached agent tool(s): {attached_list}[/green]"
                                    )
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

                            target_agent = agent_name or agent

                            if dump:
                                if not prompt_provider.can_dump_agent_cards():
                                    rich_print(
                                        "[yellow]AgentCard dumping is not available in this session.[/yellow]"
                                    )
                                    continue
                                try:
                                    card_text = await prompt_provider.dump_agent_card(
                                        target_agent
                                    )
                                except Exception as exc:
                                    rich_print(f"[red]AgentCard dump failed: {exc}[/red]")
                                    continue
                                print(card_text)
                                continue

                            if add_tool and remove_tool:
                                if not prompt_provider.can_detach_agent_tools():
                                    rich_print(
                                        "[yellow]Agent tool detachment is not available in this session.[/yellow]"
                                    )
                                    continue
                                try:
                                    removed = await prompt_provider.detach_agent_tools(
                                        agent, [target_agent]
                                    )
                                except Exception as exc:
                                    rich_print(
                                        f"[red]Agent tool detach failed: {exc}[/red]"
                                    )
                                    continue
                                if removed:
                                    removed_list = ", ".join(removed)
                                    rich_print(
                                        f"[green]Detached agent tool(s): {removed_list}[/green]"
                                    )
                                else:
                                    rich_print(
                                        "[yellow]No agent tools detached.[/yellow]"
                                    )
                                continue
                            if add_tool:
                                if target_agent == agent:
                                    rich_print(
                                        "[yellow]Can't attach agent to itself.[/yellow]"
                                    )
                                    continue
                                if target_agent not in available_agents_set:
                                    rich_print(
                                        f"[red]Agent '{target_agent}' not found[/red]"
                                    )
                                    continue
                                if not prompt_provider.can_attach_agent_tools():
                                    rich_print(
                                        "[yellow]Agent tool attachment is not available in this session.[/yellow]"
                                    )
                                    continue
                                try:
                                    attached = await prompt_provider.attach_agent_tools(
                                        agent, [target_agent]
                                    )
                                except Exception as exc:
                                    rich_print(
                                        f"[red]Agent tool attach failed: {exc}[/red]"
                                    )
                                    continue

                                if attached:
                                    attached_list = ", ".join(attached)
                                    rich_print(
                                        f"[green]Attached agent tool(s): {attached_list}[/green]"
                                    )
                                else:
                                    rich_print(
                                        "[yellow]No agent tools attached.[/yellow]"
                                    )
                                continue

                            rich_print("[red]Invalid /agent command.[/red]")
                            continue
                        case ReloadAgentsCommand():
                            if not prompt_provider.can_reload_agents():
                                rich_print(
                                    "[yellow]Reload is not available in this session.[/yellow]"
                                )
                                continue

                            reloadable = prompt_provider.reload_agents
                            try:
                                changed = await reloadable()
                            except Exception as exc:
                                rich_print(f"[red]Reload failed: {exc}[/red]")
                                continue

                            if not changed:
                                rich_print("[dim]No AgentCard changes detected.[/dim]")
                                continue

                            available_agents = list(prompt_provider.agent_names())
                            available_agents_set = set(available_agents)
                            self.agent_types = prompt_provider.agent_types()

                            if agent not in available_agents_set:
                                if available_agents:
                                    agent = available_agents[0]
                                else:
                                    rich_print(
                                        "[red]No agents available after reload.[/red]"
                                    )
                                    return result

                            rich_print("[green]AgentCards reloaded.[/green]")
                            continue
                        case _:
                            pass

                # Skip further processing if:
                # 1. The command was handled (command_result is truthy)
                # 2. The original input was a command payload (special command like /prompt)
                # 3. The command result itself is a command payload (special command handling result)
                # This fixes the issue where /prompt without arguments gets sent to the LLM
                # Skip these checks if we have a pending hash command to handle outside
                if not hash_send_target:
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

            # Handle hash command OUTSIDE paused context so progress display works correctly
            if hash_send_target and hash_send_message:
                rich_print(f"[dim]Asking {hash_send_target}...[/dim]")
                try:
                    await send_func(hash_send_message, hash_send_target)
                except Exception as exc:
                    with progress_display.paused():
                        rich_print(f"[red]Error asking {hash_send_target}: {exc}[/red]")
                    continue

                # Pause progress display for status messages after send completes
                with progress_display.paused():
                    # Get the last assistant message from the target agent
                    target_agent_obj = prompt_provider._agent(hash_send_target)
                    response_text = ""
                    for msg in reversed(target_agent_obj.message_history):
                        if msg.role == "assistant":
                            response_text = msg.last_text()
                            break

                    if response_text:
                        buffer_prefill = response_text
                        rich_print(
                            f"[green]Response from {hash_send_target} loaded into input buffer[/green]"
                        )
                    else:
                        rich_print(
                            f"[yellow]No response received from {hash_send_target}[/yellow]"
                        )
                continue

            # Send the message to the agent
            # Type narrowing: by this point user_input is str (non-str inputs continue above)
            assert isinstance(user_input, str)
            result = await send_func(user_input, agent)

        return result

    def _create_combined_separator_status(
        self, left_content: str, right_info: str, console
    ) -> None:
        """
        Create a combined separator and status line using the new visual style.

        Args:
            left_content: The main content (block, arrow, name) - left justified with color
            right_info: Supplementary information to show in brackets - right aligned
            console: Rich console instance to use
        """
        from rich.text import Text

        width = console.size.width

        # Create left text
        left_text = Text.from_markup(left_content)

        # Create right text if we have info
        if right_info and right_info.strip():
            # Add dim brackets around the right info
            right_text = Text()
            right_text.append("[", style="dim")
            right_text.append_text(Text.from_markup(right_info))
            right_text.append("]", style="dim")
            # Calculate separator count
            separator_count = width - left_text.cell_len - right_text.cell_len
            if separator_count < 1:
                separator_count = 1  # Always at least 1 separator
        else:
            right_text = Text("")
            separator_count = width - left_text.cell_len

        # Build the combined line
        combined = Text()
        combined.append_text(left_text)
        combined.append(" ", style="default")
        combined.append("─" * (separator_count - 1), style="dim")
        combined.append_text(right_text)

        # Print with empty line before
        rich_print()
        console.print(combined)
        rich_print()

    async def _get_all_prompts(self, prompt_provider: "AgentApp", agent_name: str | None = None):
        """
        Get a list of all available prompts.

        Args:
            prompt_provider: Provider that implements list_prompts
            agent_name: Optional agent name (for multi-agent apps)

        Returns:
            List of prompt info dictionaries, sorted by server and name
        """
        try:
            # Call list_prompts on the provider
            prompt_servers = await prompt_provider.list_prompts(
                namespace=None, agent_name=agent_name
            )

            all_prompts = []

            # Process the returned prompt servers
            if prompt_servers:
                # First collect all prompts
                for server_name, prompts_info in prompt_servers.items():
                    if prompts_info and hasattr(prompts_info, "prompts") and prompts_info.prompts:
                        for prompt in prompts_info.prompts:
                            # Use the SEP constant for proper namespacing
                            all_prompts.append(
                                {
                                    "server": server_name,
                                    "name": prompt.name,
                                    "namespaced_name": f"{server_name}{SEP}{prompt.name}",
                                    "title": prompt.title or None,
                                    "description": prompt.description or "No description",
                                    "arg_count": len(prompt.arguments or []),
                                    "arguments": prompt.arguments or [],
                                }
                            )
                    elif isinstance(prompts_info, list) and prompts_info:
                        for prompt in prompts_info:
                            if isinstance(prompt, dict) and "name" in prompt:
                                all_prompts.append(
                                    {
                                        "server": server_name,
                                        "name": prompt["name"],
                                        "namespaced_name": f"{server_name}{SEP}{prompt['name']}",
                                        "title": prompt.get("title", None),
                                        "description": prompt.get("description", "No description"),
                                        "arg_count": len(prompt.get("arguments", [])),
                                        "arguments": prompt.get("arguments", []),
                                    }
                                )
                            else:
                                # Handle Prompt objects from mcp.types
                                prompt_obj = cast("Prompt", prompt)
                                all_prompts.append(
                                    {
                                        "server": server_name,
                                        "name": prompt_obj.name,
                                        "namespaced_name": f"{server_name}{SEP}{prompt_obj.name}",
                                        "title": prompt_obj.title or None,
                                        "description": prompt_obj.description or "No description",
                                        "arg_count": len(prompt_obj.arguments or []),
                                        "arguments": prompt_obj.arguments or [],
                                    }
                                )

                # Sort prompts by server and name for consistent ordering
                all_prompts.sort(key=lambda p: (p["server"], p["name"]))

            return all_prompts

        except Exception as e:
            import traceback

            from rich import print as rich_print

            rich_print(f"[red]Error getting prompts: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")
            return []

    async def _list_prompts(self, prompt_provider: "AgentApp", agent_name: str) -> None:
        """
        List available prompts for an agent.

        Args:
            prompt_provider: Provider that implements list_prompts
            agent_name: Name of the agent
        """
        try:
            # Get all prompts using the helper function
            all_prompts = await self._get_all_prompts(prompt_provider, agent_name)

            rich_print(f"\n[bold]Prompts for agent [cyan]{agent_name}[/cyan]:[/bold]")

            if not all_prompts:
                rich_print("[yellow]No prompts available for this agent[/yellow]")
                return

            rich_print()

            # Display prompts using clean compact format
            for i, prompt in enumerate(all_prompts, 1):
                # Main line: [ 1] server•prompt_name Title
                from rich.text import Text

                prompt_line = Text()
                prompt_line.append(f"[{i:2}] ", style="dim cyan")
                prompt_line.append(f"{prompt['server']}•", style="dim green")
                prompt_line.append(prompt["name"], style="bright_blue bold")

                # Add title if available
                if prompt["title"] and prompt["title"].strip():
                    prompt_line.append(f" {prompt['title']}", style="default")

                rich_print(prompt_line)

                # Description lines - show 2-3 rows if needed
                if prompt["description"] and prompt["description"].strip():
                    description = prompt["description"].strip()
                    # Calculate rough character limit for 2-3 lines (assuming ~80 chars per line with indent)
                    char_limit = 240  # About 3 lines worth

                    if len(description) > char_limit:
                        # Find a good break point near the limit (prefer sentence/word boundaries)
                        truncate_pos = char_limit
                        # Look back for sentence end
                        sentence_break = description.rfind(". ", 0, char_limit + 20)
                        if sentence_break > char_limit - 50:  # If we found a nearby sentence break
                            truncate_pos = sentence_break + 1
                        else:
                            # Look for word boundary
                            word_break = description.rfind(" ", 0, char_limit + 10)
                            if word_break > char_limit - 30:  # If we found a nearby word break
                                truncate_pos = word_break

                        description = description[:truncate_pos].rstrip() + "..."

                    # Split into lines and wrap
                    import textwrap

                    wrapped_lines = textwrap.wrap(description, width=72, subsequent_indent="     ")
                    for line in wrapped_lines:
                        if line.startswith("     "):  # Already indented continuation line
                            rich_print(f"     [white]{line[5:]}[/white]")
                        else:  # First line needs indent
                            rich_print(f"     [white]{line}[/white]")

                # Arguments line - show argument names if available
                if prompt["arg_count"] > 0:
                    arg_names = prompt.get("arg_names", [])
                    required_args = prompt.get("required_args", [])

                    if arg_names:
                        arg_list = []
                        for arg_name in arg_names:
                            if arg_name in required_args:
                                arg_list.append(f"{arg_name}*")
                            else:
                                arg_list.append(arg_name)

                        args_text = ", ".join(arg_list)
                        if len(args_text) > 80:
                            args_text = args_text[:77] + "..."
                        rich_print(f"     [dim magenta]args: {args_text}[/dim magenta]")
                    else:
                        rich_print(
                            f"     [dim magenta]args: {prompt['arg_count']} parameter{'s' if prompt['arg_count'] != 1 else ''}[/dim magenta]"
                        )

                rich_print()  # Space between prompts

            # Add usage instructions
            rich_print(
                "[dim]Usage: /prompt <number> to select by number, or /prompts for interactive selection[/dim]"
            )

        except Exception as e:
            import traceback

            rich_print(f"[red]Error listing prompts: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

    async def _select_prompt(
        self,
        prompt_provider: "AgentApp",
        agent_name: str,
        requested_name: str | None = None,
        send_func: SendFunc | None = None,
    ) -> None:
        """
        Select and apply a prompt.

        Args:
            prompt_provider: Provider that implements list_prompts and get_prompt
            agent_name: Name of the agent
            requested_name: Optional name of the prompt to apply
        """
        try:
            # Get all available prompts directly from the prompt provider
            rich_print(f"\n[bold]Fetching prompts for agent [cyan]{agent_name}[/cyan]...[/bold]")

            # Call list_prompts on the provider
            prompt_servers = await prompt_provider.list_prompts(
                namespace=None, agent_name=agent_name
            )

            if not prompt_servers:
                rich_print("[yellow]No prompts available for this agent[/yellow]")
                return

            # Process fetched prompts
            all_prompts = []
            for server_name, prompts_info in prompt_servers.items():
                if not prompts_info:
                    continue

                # Extract prompts
                prompts: list[Prompt] = []
                if hasattr(prompts_info, "prompts"):
                    prompts = prompts_info.prompts
                elif isinstance(prompts_info, list):
                    prompts = prompts_info

                # Process each prompt
                for prompt in prompts:
                    # Get basic prompt info
                    prompt_name = prompt.name
                    prompt_title = prompt.title or None
                    prompt_description = prompt.description or "No description"

                    # Extract argument information
                    arg_names = []
                    required_args = []
                    optional_args = []
                    arg_descriptions = {}

                    # Get arguments list
                    if prompt.arguments:
                        for arg in prompt.arguments:
                            arg_names.append(arg.name)

                            # Store description if available
                            if arg.description:
                                arg_descriptions[arg.name] = arg.description

                            # Check if required
                            if arg.required:
                                required_args.append(arg.name)
                            else:
                                optional_args.append(arg.name)

                    # Create namespaced version using the consistent separator
                    namespaced_name = f"{server_name}{SEP}{prompt_name}"

                    # Add to collection
                    all_prompts.append(
                        {
                            "server": server_name,
                            "name": prompt_name,
                            "namespaced_name": namespaced_name,
                            "title": prompt_title,
                            "description": prompt_description,
                            "arg_count": len(arg_names),
                            "arg_names": arg_names,
                            "required_args": required_args,
                            "optional_args": optional_args,
                            "arg_descriptions": arg_descriptions,
                        }
                    )

            if not all_prompts:
                rich_print("[yellow]No prompts available for this agent[/yellow]")
                return

            # Sort prompts by server then name
            all_prompts.sort(key=lambda p: (p["server"], p["name"]))

            # Handle specifically requested prompt
            if requested_name:
                matching_prompts = [
                    p
                    for p in all_prompts
                    if p["name"] == requested_name or p["namespaced_name"] == requested_name
                ]

                if not matching_prompts:
                    rich_print(f"[red]Prompt '{requested_name}' not found[/red]")
                    rich_print("[yellow]Available prompts:[/yellow]")
                    for p in all_prompts:
                        rich_print(f"  {p['namespaced_name']}")
                    return

                # If exactly one match, use it
                if len(matching_prompts) == 1:
                    selected_prompt = matching_prompts[0]
                else:
                    # Handle multiple matches
                    rich_print(f"[yellow]Multiple prompts match '{requested_name}':[/yellow]")
                    for i, p in enumerate(matching_prompts):
                        rich_print(f"  {i + 1}. {p['namespaced_name']} - {p['description']}")

                    # Get user selection
                    selection = (
                        await get_selection_input("Enter prompt number to select: ", default="1")
                        or ""
                    )

                    try:
                        idx = int(selection) - 1
                        if 0 <= idx < len(matching_prompts):
                            selected_prompt = matching_prompts[idx]
                        else:
                            rich_print("[red]Invalid selection[/red]")
                            return
                    except ValueError:
                        rich_print("[red]Invalid input, please enter a number[/red]")
                        return
            else:
                # Show prompt selection UI using clean compact format
                rich_print(f"\n[bold]Select a prompt for agent [cyan]{agent_name}[/cyan]:[/bold]")
                rich_print()

                # Display prompts using the same format as _list_prompts
                for i, prompt in enumerate(all_prompts, 1):
                    # Main line: [ 1] server•prompt_name Title
                    from rich.text import Text

                    prompt_line = Text()
                    prompt_line.append(f"[{i:2}] ", style="dim cyan")
                    prompt_line.append(f"{prompt['server']}•", style="dim green")
                    prompt_line.append(prompt["name"], style="bright_blue bold")

                    # Add title if available
                    if prompt["title"] and prompt["title"].strip():
                        prompt_line.append(f" {prompt['title']}", style="default")

                    rich_print(prompt_line)

                    # Description lines - show 2-3 rows if needed
                    if prompt["description"] and prompt["description"].strip():
                        description = prompt["description"].strip()
                        # Calculate rough character limit for 2-3 lines (assuming ~80 chars per line with indent)
                        char_limit = 240  # About 3 lines worth

                        if len(description) > char_limit:
                            # Find a good break point near the limit (prefer sentence/word boundaries)
                            truncate_pos = char_limit
                            # Look back for sentence end
                            sentence_break = description.rfind(". ", 0, char_limit + 20)
                            if (
                                sentence_break > char_limit - 50
                            ):  # If we found a nearby sentence break
                                truncate_pos = sentence_break + 1
                            else:
                                # Look for word boundary
                                word_break = description.rfind(" ", 0, char_limit + 10)
                                if word_break > char_limit - 30:  # If we found a nearby word break
                                    truncate_pos = word_break

                            description = description[:truncate_pos].rstrip() + "..."

                        # Split into lines and wrap
                        import textwrap

                        wrapped_lines = textwrap.wrap(
                            description, width=72, subsequent_indent="     "
                        )
                        for line in wrapped_lines:
                            if line.startswith("     "):  # Already indented continuation line
                                rich_print(f"     [white]{line[5:]}[/white]")
                            else:  # First line needs indent
                                rich_print(f"     [white]{line}[/white]")

                    # Arguments line - show argument names if available
                    if prompt["arg_count"] > 0:
                        arg_names = prompt.get("arg_names", [])
                        required_args = prompt.get("required_args", [])

                        if arg_names:
                            arg_list = []
                            for arg_name in arg_names:
                                if arg_name in required_args:
                                    arg_list.append(f"{arg_name}*")
                                else:
                                    arg_list.append(arg_name)

                            args_text = ", ".join(arg_list)
                            if len(args_text) > 80:
                                args_text = args_text[:77] + "..."
                            rich_print(f"     [dim magenta]args: {args_text}[/dim magenta]")
                        else:
                            rich_print(
                                f"     [dim magenta]args: {prompt['arg_count']} parameter{'s' if prompt['arg_count'] != 1 else ''}[/dim magenta]"
                            )

                    rich_print()  # Space between prompts

                prompt_names = [str(i) for i, _ in enumerate(all_prompts, 1)]

                # Get user selection
                selection = await get_selection_input(
                    "Enter prompt number to select (or press Enter to cancel): ",
                    options=prompt_names,
                    allow_cancel=True,
                )

                # Handle cancellation
                if not selection or selection.strip() == "":
                    rich_print("[yellow]Prompt selection cancelled[/yellow]")
                    return

                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(all_prompts):
                        selected_prompt = all_prompts[idx]
                    else:
                        rich_print("[red]Invalid selection[/red]")
                        return
                except ValueError:
                    rich_print("[red]Invalid input, please enter a number[/red]")
                    return

            # Get prompt arguments
            required_args = selected_prompt["required_args"]
            optional_args = selected_prompt["optional_args"]
            arg_descriptions = selected_prompt.get("arg_descriptions", {})
            arg_values = {}

            # Show argument info if any
            if required_args or optional_args:
                if required_args and optional_args:
                    rich_print(
                        f"\n[bold]Prompt [cyan]{selected_prompt['name']}[/cyan] requires {len(required_args)} arguments and has {len(optional_args)} optional arguments:[/bold]"
                    )
                elif required_args:
                    rich_print(
                        f"\n[bold]Prompt [cyan]{selected_prompt['name']}[/cyan] requires {len(required_args)} arguments:[/bold]"
                    )
                elif optional_args:
                    rich_print(
                        f"\n[bold]Prompt [cyan]{selected_prompt['name']}[/cyan] has {len(optional_args)} optional arguments:[/bold]"
                    )

                # Collect required arguments
                for arg_name in required_args:
                    description = arg_descriptions.get(arg_name, "")
                    arg_value = await get_argument_input(
                        arg_name=arg_name,
                        description=description,
                        required=True,
                    )
                    if arg_value is not None:
                        arg_values[arg_name] = arg_value

                # Collect optional arguments
                if optional_args:
                    for arg_name in optional_args:
                        description = arg_descriptions.get(arg_name, "")
                        arg_value = await get_argument_input(
                            arg_name=arg_name,
                            description=description,
                            required=False,
                        )
                        if arg_value:
                            arg_values[arg_name] = arg_value

            # Apply the prompt using generate() for proper progress display
            namespaced_name = selected_prompt["namespaced_name"]
            rich_print(f"\n[bold]Applying prompt [cyan]{namespaced_name}[/cyan]...[/bold]")

            # Get the agent directly for generate() call
            assert hasattr(prompt_provider, "_agent"), (
                "Interactive prompt expects an AgentApp with _agent()"
            )
            agent = prompt_provider._agent(agent_name)

            try:
                # Use agent.apply_prompt() which handles everything properly:
                # - get_prompt() to fetch template
                # - convert to multipart
                # - call generate() for progress display
                # - return response text
                # Response display is handled by the agent's show_ methods, don't print it here

                # Fetch the prompt first (without progress display)
                prompt_result = await agent.get_prompt(namespaced_name, arg_values)

                if not prompt_result or not prompt_result.messages:
                    rich_print(
                        f"[red]Prompt '{namespaced_name}' could not be found or contains no messages[/red]"
                    )
                    return

                # Convert to multipart format
                from fast_agent.types import PromptMessageExtended

                multipart_messages = PromptMessageExtended.from_get_prompt_result(prompt_result)

                # Now start progress display for the actual generation
                progress_display.resume()
                try:
                    await agent.generate(multipart_messages, None)
                finally:
                    # Pause again for the next UI interaction
                    progress_display.pause()

                # Show usage info after the turn (same as send_wrapper does)
                prompt_provider._show_turn_usage(agent_name)

            except Exception as e:
                rich_print(f"[red]Error applying prompt: {e}[/red]")

        except Exception as e:
            import traceback

            rich_print(f"[red]Error selecting or applying prompt: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

    async def _list_tools(self, prompt_provider: "AgentApp", agent_name: str) -> None:
        """
        List available tools for an agent.

        Args:
            prompt_provider: Provider that implements list_tools
            agent_name: Name of the agent
        """
        try:
            # Get agent to list tools from
            assert hasattr(prompt_provider, "_agent"), (
                "Interactive prompt expects an AgentApp with _agent()"
            )
            agent = prompt_provider._agent(agent_name)

            rich_print(f"\n[bold]Tools for agent [cyan]{agent_name}[/cyan]:[/bold]")

            # Get tools using list_tools
            tools_result = await agent.list_tools()

            if not tools_result or not hasattr(tools_result, "tools") or not tools_result.tools:
                rich_print("[yellow]No tools available for this agent[/yellow]")
                return

            rich_print()

            # Display tools using clean compact format
            index = 1
            for tool in tools_result.tools:
                # Main line: [ 1] tool_name Title
                from rich.text import Text

                meta = getattr(tool, "meta", {}) or {}

                tool_line = Text()
                tool_line.append(f"[{index:2}] ", style="dim cyan")
                tool_line.append(tool.name, style="bright_blue bold")

                # Add title if available
                if tool.title and tool.title.strip():
                    tool_line.append(f" {tool.title}", style="default")

                if meta.get("openai/skybridgeEnabled"):
                    tool_line.append(" (skybridge)", style="cyan")

                rich_print(tool_line)

                # Description lines - show 2-3 rows if needed
                if tool.description and tool.description.strip():
                    description = tool.description.strip()
                    # Calculate rough character limit for 2-3 lines (assuming ~80 chars per line with indent)
                    char_limit = 240  # About 3 lines worth

                    if len(description) > char_limit:
                        # Find a good break point near the limit (prefer sentence/word boundaries)
                        truncate_pos = char_limit
                        # Look back for sentence end
                        sentence_break = description.rfind(". ", 0, char_limit + 20)
                        if sentence_break > char_limit - 50:  # If we found a nearby sentence break
                            truncate_pos = sentence_break + 1
                        else:
                            # Look for word boundary
                            word_break = description.rfind(" ", 0, char_limit + 10)
                            if word_break > char_limit - 30:  # If we found a nearby word break
                                truncate_pos = word_break

                        description = description[:truncate_pos].rstrip() + "..."

                    # Split into lines and wrap
                    import textwrap

                    wrapped_lines = textwrap.wrap(description, width=72, subsequent_indent="     ")
                    for line in wrapped_lines:
                        if line.startswith("     "):  # Already indented continuation line
                            rich_print(f"     [white]{line[5:]}[/white]")
                        else:  # First line needs indent
                            rich_print(f"     [white]{line}[/white]")

                # Arguments line - show schema info if available
                if hasattr(tool, "inputSchema") and tool.inputSchema:
                    schema = tool.inputSchema
                    if "properties" in schema:
                        properties = schema["properties"]
                        required = schema.get("required", [])

                        arg_list = []
                        for prop_name, prop_info in properties.items():
                            if prop_name in required:
                                arg_list.append(f"{prop_name}*")
                            else:
                                arg_list.append(prop_name)

                        if arg_list:
                            args_text = ", ".join(arg_list)
                            if len(args_text) > 80:
                                args_text = args_text[:77] + "..."
                            rich_print(f"     [dim magenta]args: {args_text}[/dim magenta]")

                if meta.get("openai/skybridgeEnabled"):
                    template = meta.get("openai/skybridgeTemplate")
                    if template:
                        rich_print(f"     [dim magenta]template:[/dim magenta] {template}")

                rich_print()  # Space between tools
                index += 1

            if index == 1:
                rich_print("[yellow]No MCP tools available for this agent[/yellow]")
        except Exception as e:
            import traceback

            rich_print(f"[red]Error listing tools: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

    async def _list_skills(self, prompt_provider: "AgentApp", agent_name: str) -> None:
        """List available local skills for an agent."""

        try:
            directories = resolve_skill_directories()
            all_manifests: dict[Path, list[SkillManifest]] = {}
            for directory in directories:
                all_manifests[directory] = list_local_skills(directory) if directory.exists() else []
            self._render_local_skills_by_directory(all_manifests)

        except Exception as exc:  # noqa: BLE001
            import traceback

            rich_print(f"[red]Error listing skills: {exc}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

    async def _handle_skills_command(
        self, prompt_provider: "AgentApp", agent_name: str, payload: dict[str, Any]
    ) -> None:
        action = str(payload.get("action") or "list").lower()
        argument = payload.get("argument")

        if action in {"list", ""}:
            await self._list_skills(prompt_provider, agent_name)
            return
        if action in {"add", "install"}:
            await self._add_skill(prompt_provider, agent_name, argument)
            return
        if action in {"registry", "marketplace", "source"}:
            await self._set_skills_registry(argument)
            return
        if action in {"remove", "rm", "delete", "uninstall"}:
            await self._remove_skill(prompt_provider, agent_name, argument)
            return

        rich_print(f"[yellow]Unknown /skills action: {action}[/yellow]")

    async def _set_skills_registry(self, argument: str | None) -> None:
        settings = get_settings()
        configured_urls = (
            settings.skills.marketplace_urls if settings.skills else None
        ) or list(DEFAULT_SKILL_REGISTRIES)

        if not argument:
            current = get_marketplace_url(settings)
            rich_print(f"[dim]Current registry:[/dim] {format_marketplace_display_url(current)}")

            # Show numbered list of configured registries
            if configured_urls:
                rich_print("\n[dim]Available registries:[/dim]")
                for i, reg_url in enumerate(configured_urls, 1):
                    display = format_marketplace_display_url(reg_url)
                    rich_print(f"  [cyan][{i}][/cyan] {display}")

            rich_print("\n[dim]Usage: /skills registry <number|url|path>[/dim]")
            return

        arg = str(argument).strip()

        # Check if argument is a number (select from configured registries)
        if arg.isdigit():
            index = int(arg)
            if not configured_urls:
                rich_print("[yellow]No registries configured.[/yellow]")
                return
            if 1 <= index <= len(configured_urls):
                url = configured_urls[index - 1]
            else:
                rich_print(
                    f"[yellow]Invalid registry number. Use 1-{len(configured_urls)}.[/yellow]"
                )
                return
        else:
            url = arg

        try:
            marketplace, resolved_url = await fetch_marketplace_skills_with_source(url)
        except Exception as exc:  # noqa: BLE001
            rich_print(f"[red]Failed to load registry: {exc}[/red]")
            return

        # Update only the active registry, preserve the configured list
        skills_settings = getattr(settings, "skills", None)
        if skills_settings is not None:
            skills_settings.marketplace_url = resolved_url

        if resolved_url != url:
            rich_print(f"[dim]Resolved from:[/dim] {url}")
        rich_print(
            f"[green]Registry set to:[/green] {format_marketplace_display_url(resolved_url)}"
        )
        rich_print(f"[dim]Skills discovered:[/dim] {len(marketplace)}")

    async def _add_skill(
        self, prompt_provider: "AgentApp", agent_name: str, argument: str | None
    ) -> None:
        manager_dir = get_manager_directory()
        marketplace_url = get_marketplace_url()
        try:
            marketplace = await fetch_marketplace_skills(marketplace_url)
        except Exception as exc:  # noqa: BLE001
            rich_print(f"[red]Failed to load marketplace: {exc}[/red]")
            return

        if not marketplace:
            rich_print("[yellow]No skills found in the marketplace.[/yellow]")
            return

        selection = argument
        if not selection:
            rich_print("\n[bold]Marketplace skills:[/bold]\n")
            repo_hint = None
            if marketplace:
                repo_url = getattr(marketplace[0], "repo_url", None)
                if repo_url:
                    repo_ref = getattr(marketplace[0], "repo_ref", None)
                    repo_hint = f"{repo_url}@{repo_ref}" if repo_ref else repo_url
            if repo_hint:
                rich_print(f"[dim]Repository: {format_marketplace_display_url(repo_hint)}[/dim]")
            self._render_marketplace_skills(marketplace)
            selection = await get_selection_input(
                "Install skill by number or name (empty to cancel): ",
                options=[entry.name for entry in marketplace],
                allow_cancel=True,
            )
            if selection is None:
                return

        skill = select_skill_by_name_or_index(marketplace, selection)
        if not skill:
            rich_print(f"[red]Skill not found: {selection}[/red]")
            return

        try:
            install_path = await install_marketplace_skill(skill, destination_root=manager_dir)
        except Exception as exc:  # noqa: BLE001
            rich_print(f"[red]Failed to install skill: {exc}[/red]")
            return

        self._render_install_result(skill, install_path)
        await self._refresh_agent_skills(prompt_provider, agent_name)

    async def _remove_skill(
        self, prompt_provider: "AgentApp", agent_name: str, argument: str | None
    ) -> None:
        manager_dir = get_manager_directory()
        manifests = list_local_skills(manager_dir)
        if not manifests:
            rich_print("[yellow]No local skills to remove.[/yellow]")
            return

        selection = argument
        if not selection:
            self._render_local_skills(manifests, manager_dir)
            selection = await get_selection_input(
                "Remove skill by number or name (empty to cancel): ",
                options=[manifest.name for manifest in manifests],
                allow_cancel=True,
            )
            if selection is None:
                return

        manifest = select_manifest_by_name_or_index(manifests, selection)
        if not manifest:
            rich_print(f"[red]Skill not found: {selection}[/red]")
            return

        try:
            skill_dir = Path(manifest.path).parent
            remove_local_skill(skill_dir, destination_root=manager_dir)
        except Exception as exc:  # noqa: BLE001
            rich_print(f"[red]Failed to remove skill: {exc}[/red]")
            return

        rich_print(f"[green]Removed skill:[/green] {manifest.name}")
        await self._refresh_agent_skills(prompt_provider, agent_name)

    async def _refresh_agent_skills(self, prompt_provider: "AgentApp", agent_name: str) -> None:
        assert hasattr(prompt_provider, "_agent"), (
            "Interactive prompt expects an AgentApp with _agent()"
        )
        agent = prompt_provider._agent(agent_name)
        override_dirs = resolve_skill_directories(get_settings())
        registry, manifests = reload_skill_manifests(
            base_dir=Path.cwd(), override_directories=override_dirs
        )
        instruction_context = None
        try:
            skills_text = format_skills_for_prompt(manifests, read_tool_name="read_skill")
            instruction_context = {"agentSkills": skills_text}
        except Exception:
            instruction_context = None

        await rebuild_agent_instruction(
            agent,
            skill_manifests=manifests,
            context=instruction_context,
            skill_registry=registry,
        )

    def _render_marketplace_skills(self, marketplace: list[Any]) -> None:
        current_bundle = None
        for index, entry in enumerate(marketplace, 1):
            from rich.text import Text

            bundle_name = getattr(entry, "bundle_name", None)
            bundle_description = getattr(entry, "bundle_description", None)
            if bundle_name and bundle_name != current_bundle:
                current_bundle = bundle_name
                rich_print("")
                rich_print(f"[bold]{bundle_name}[/bold]")
                if bundle_description:
                    wrapped_lines = textwrap.wrap(bundle_description.strip(), width=72)
                    for line in wrapped_lines:
                        rich_print(f"[white]{line.strip()}[/white]")
                rich_print("")

            tool_line = Text()
            tool_line.append(f"[{index:2}] ", style="dim cyan")
            tool_line.append(entry.name, style="bright_blue bold")
            rich_print(tool_line)

            if entry.description:
                wrapped_lines = textwrap.wrap(
                    entry.description.strip(), width=72, subsequent_indent="     "
                )
                for line in wrapped_lines:
                    if line.startswith("     "):
                        rich_print(f"     [white]{line[5:]}[/white]")
                    else:
                        rich_print(f"     [white]{line}[/white]")
            if entry.source_url:
                rich_print(f"     [dim green]source:[/dim green] {entry.source_url}")
            rich_print()

    def _render_local_skills(self, manifests: list[Any], manager_dir: Path) -> None:
        rich_print(f"\n[bold]Skills in [cyan]{manager_dir}[/cyan]:[/bold]\n")
        if not manifests:
            rich_print("[yellow]No skills available in the manager directory[/yellow]")
            rich_print("[dim]Use /skills add to install a skill[/dim]")
            return
        for index, manifest in enumerate(manifests, 1):
            from rich.text import Text

            name = getattr(manifest, "name", "")
            description = getattr(manifest, "description", "")
            path = Path(getattr(manifest, "path", Path()))

            tool_line = Text()
            tool_line.append(f"[{index:2}] ", style="dim cyan")
            tool_line.append(name, style="bright_blue bold")
            rich_print(tool_line)

            if description:
                wrapped_lines = textwrap.wrap(
                    description.strip(), width=72, subsequent_indent="     "
                )
                for line in wrapped_lines:
                    if line.startswith("     "):
                        rich_print(f"     [white]{line[5:]}[/white]")
                    else:
                        rich_print(f"     [white]{line}[/white]")

            source_path = path if path else Path(".")
            if source_path.is_file():
                source_path = source_path.parent
            try:
                display_path = source_path.relative_to(Path.cwd())
            except ValueError:
                display_path = source_path

        rich_print(f"     [dim green]source:[/dim green] {display_path}")
        rich_print()
        rich_print("[dim]Use /skills add to install a skill[/dim]")
        rich_print("[dim]Remove a skill with /skills remove <number|name>[/dim]")

    def _render_local_skills_by_directory(self, manifests_by_dir: dict[Path, list[SkillManifest]]) -> None:
        from rich.text import Text

        total_skills = sum(len(m) for m in manifests_by_dir.values())
        skill_index = 0

        for directory, manifests in manifests_by_dir.items():
            try:
                display_dir = directory.relative_to(Path.cwd())
            except ValueError:
                display_dir = directory

            rich_print(f"\n[bold]Skills in [cyan]{display_dir}[/cyan]:[/bold]\n")

            if not manifests:
                rich_print("[yellow]No skills in this directory[/yellow]")
            else:
                for manifest in manifests:
                    skill_index += 1

                    tool_line = Text()
                    tool_line.append(f"[{skill_index:2}] ", style="dim cyan")
                    tool_line.append(manifest.name, style="bright_blue bold")
                    rich_print(tool_line)

                    if manifest.description:
                        wrapped_lines = textwrap.wrap(
                            manifest.description.strip(), width=72, subsequent_indent="     "
                        )
                        for line in wrapped_lines:
                            if line.startswith("     "):
                                rich_print(f"     [white]{line[5:]}[/white]")
                            else:
                                rich_print(f"     [white]{line}[/white]")

                    source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
                    try:
                        source_display = source_path.relative_to(Path.cwd())
                    except ValueError:
                        source_display = source_path
                    rich_print(f"     [dim green]source:[/dim green] {source_display}")
                    rich_print()

        if total_skills == 0:
            rich_print("[dim]Use /skills add to install a skill[/dim]")
        else:
            rich_print("[dim]Use /skills add to install a skill[/dim]")
            rich_print("[dim]Remove a skill with /skills remove <number|name>[/dim]")

    def _render_install_result(self, skill: Any, install_path: Path) -> None:
        try:
            display_path = install_path.relative_to(Path.cwd())
        except ValueError:
            display_path = install_path
        rich_print(f"[green]Installed skill:[/green] {skill.name}")
        rich_print(f"[dim green]location:[/dim green] {display_path}")

    async def _show_usage(self, prompt_provider: "AgentApp", agent_name: str) -> None:
        """
        Show usage statistics for the current agent(s) in a colorful table format.

        Args:
            prompt_provider: Provider that has access to agents
            agent_name: Name of the current agent
        """
        try:
            # Collect all agents from the prompt provider
            agents_to_show = collect_agents_from_provider(prompt_provider, agent_name)

            if not agents_to_show:
                rich_print("[yellow]No usage data available[/yellow]")
                return

            # Use the shared display utility
            display_usage_report(agents_to_show, show_if_progress_disabled=True)

        except Exception as e:
            rich_print(f"[red]Error showing usage: {e}[/red]")

    async def _show_system(self, prompt_provider: "AgentApp", agent_name: str) -> None:
        """
        Show the current system prompt for the agent.

        Args:
            prompt_provider: Provider that has access to agents
            agent_name: Name of the current agent
        """
        try:
            # Get agent to display from
            assert hasattr(prompt_provider, "_agent"), (
                "Interactive prompt expects an AgentApp with _agent()"
            )
            agent = prompt_provider._agent(agent_name)

            # Get the system prompt
            system_prompt = getattr(agent, "instruction", None)
            if not system_prompt:
                rich_print("[yellow]No system prompt available[/yellow]")
                return

            # Get server count for display
            server_count = 0
            if isinstance(agent, McpAgentProtocol):
                server_names = agent.aggregator.server_names
                server_count = len(server_names) if server_names else 0

            # Use the display utility to show the system prompt
            agent_display = getattr(agent, "display", None)
            if agent_display:
                agent_display.show_system_message(
                    system_prompt=system_prompt, agent_name=agent_name, server_count=server_count
                )
            else:
                # Fallback to basic display
                from fast_agent.ui.console_display import ConsoleDisplay

                agent_context = getattr(agent, "context", None)
                display = ConsoleDisplay(
                    config=agent_context.config if hasattr(agent_context, "config") else None
                )
                display.show_system_message(
                    system_prompt=system_prompt, agent_name=agent_name, server_count=server_count
                )

        except Exception as e:
            import traceback

            rich_print(f"[red]Error showing system prompt: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

    async def _show_markdown(self, prompt_provider: "AgentApp", agent_name: str) -> None:
        """
        Show the last assistant message without markdown formatting.

        Args:
            prompt_provider: Provider that has access to agents
            agent_name: Name of the current agent
        """
        try:
            # Get agent to display from
            assert hasattr(prompt_provider, "_agent"), (
                "Interactive prompt expects an AgentApp with _agent()"
            )
            agent = prompt_provider._agent(agent_name)

            # Check if agent has message history
            if not agent.llm:
                rich_print("[yellow]No message history available[/yellow]")
                return

            message_history = agent.message_history
            if not message_history:
                rich_print("[yellow]No messages in history[/yellow]")
                return

            # Find the last assistant message
            last_assistant_msg = None
            for msg in reversed(message_history):
                if msg.role == "assistant":
                    last_assistant_msg = msg
                    break

            if not last_assistant_msg:
                rich_print("[yellow]No assistant messages found[/yellow]")
                return

            # Get the text content and display without markdown
            content = last_assistant_msg.last_text()

            # Display with a simple header
            rich_print("\n[bold blue]Last Assistant Response (Plain Text):[/bold blue]")
            rich_print("─" * 60)
            # Use console.print with markup=False to display raw text
            from fast_agent.ui import console

            console.console.print(content, markup=False)
            rich_print("─" * 60)
            rich_print()

        except Exception as e:
            rich_print(f"[red]Error showing markdown: {e}[/red]")
