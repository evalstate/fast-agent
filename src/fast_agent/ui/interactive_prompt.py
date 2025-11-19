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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Union, cast

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp

from mcp.types import Prompt, PromptMessage
from rich import print as rich_print

from fast_agent.agents.agent_types import AgentType
from fast_agent.history.history_exporter import HistoryExporter
from fast_agent.mcp.mcp_aggregator import SEP
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.enhanced_prompt import (
    _display_agent_info_helper,
    get_argument_input,
    get_enhanced_input,
    get_selection_input,
    handle_special_commands,
    show_mcp_status,
)
from fast_agent.ui.history_display import display_history_overview
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.usage_display import collect_agents_from_provider, display_usage_report

# Type alias for the send function
SendFunc = Callable[[Union[str, PromptMessage, PromptMessageExtended], str], Awaitable[str]]

# Type alias for the agent getter function
AgentGetter = Callable[[str], Optional[object]]


class InteractivePrompt:
    """
    Provides interactive prompt functionality that works with any agent implementation.
    This is extracted from the original AgentApp implementation to support DirectAgentApp.
    """

    def __init__(self, agent_types: Optional[Dict[str, AgentType]] = None) -> None:
        """
        Initialize the interactive prompt.

        Args:
            agent_types: Dictionary mapping agent names to their types for display
        """
        self.agent_types: Dict[str, AgentType] = agent_types or {}

    async def prompt_loop(
        self,
        send_func: SendFunc,
        default_agent: str,
        available_agents: List[str],
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
        while True:
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
                )

                # Handle special commands - pass "True" to enable agent switching
                command_result = await handle_special_commands(user_input, True)

                # Check if we should switch agents
                if isinstance(command_result, dict):
                    command_dict: Dict[str, Any] = command_result
                    if "switch_agent" in command_dict:
                        new_agent = command_dict["switch_agent"]
                        if new_agent in available_agents_set:
                            agent = new_agent
                            # Display new agent info immediately when switching
                            rich_print()  # Add spacing
                            await _display_agent_info_helper(agent, prompt_provider)
                            continue
                        else:
                            rich_print(f"[red]Agent '{new_agent}' not found[/red]")
                            continue
                    # Keep the existing list_prompts handler for backward compatibility
                    elif "list_prompts" in command_dict:
                        # Use the prompt_provider directly
                        await self._list_prompts(prompt_provider, agent)
                        continue
                    elif "select_prompt" in command_dict:
                        # Handle prompt selection, using both list_prompts and apply_prompt
                        prompt_name = command_dict.get("prompt_name")
                        prompt_index = command_dict.get("prompt_index")

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
                    elif "list_tools" in command_dict:
                        # Handle tools list display
                        await self._list_tools(prompt_provider, agent)
                        continue
                    elif "list_skills" in command_dict:
                        await self._list_skills(prompt_provider, agent)
                        continue
                    elif "show_usage" in command_dict:
                        # Handle usage display
                        await self._show_usage(prompt_provider, agent)
                        continue
                    elif "show_history" in command_dict:
                        history_info = command_dict.get("show_history")
                        history_agent = (
                            history_info.get("agent") if isinstance(history_info, dict) else None
                        )
                        target_agent = history_agent or agent
                        try:
                            agent_obj = prompt_provider._agent(target_agent)
                        except Exception:
                            rich_print(f"[red]Unable to load agent '{target_agent}'[/red]")
                            continue

                        history = getattr(agent_obj, "message_history", [])
                        usage = getattr(agent_obj, "usage_accumulator", None)
                        display_history_overview(target_agent, history, usage)
                        continue
                    elif "clear_last" in command_dict:
                        clear_info = command_dict.get("clear_last")
                        clear_agent = (
                            clear_info.get("agent") if isinstance(clear_info, dict) else None
                        )
                        target_agent = clear_agent or agent
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
                            role_display = role.capitalize() if isinstance(role, str) else "Message"
                            rich_print(
                                f"[green]Removed last {role_display} for agent '{target_agent}'.[/green]"
                            )
                        else:
                            rich_print(
                                f"[yellow]No messages to remove for agent '{target_agent}'.[/yellow]"
                            )
                        continue
                    elif "clear_history" in command_dict:
                        clear_info = command_dict.get("clear_history")
                        clear_agent = (
                            clear_info.get("agent") if isinstance(clear_info, dict) else None
                        )
                        target_agent = clear_agent or agent
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
                    elif "show_system" in command_dict:
                        # Handle system prompt display
                        await self._show_system(prompt_provider, agent)
                        continue
                    elif "show_markdown" in command_dict:
                        # Handle markdown display
                        await self._show_markdown(prompt_provider, agent)
                        continue
                    elif "show_mcp_status" in command_dict:
                        rich_print()
                        await show_mcp_status(agent, prompt_provider)
                        continue
                    elif "save_history" in command_dict:
                        # Save history for the current agent
                        filename = command_dict.get("filename")
                        try:
                            agent_obj = prompt_provider._agent(agent)

                            # Prefer type-safe exporter over magic string
                            saved_path = await HistoryExporter.save(agent_obj, filename)
                            rich_print(f"[green]History saved to {saved_path}[/green]")
                        except Exception:
                            # Fallback to magic string path for maximum compatibility
                            control = "***SAVE_HISTORY" + (f" {filename}" if filename else "")
                            result = await send_func(control, agent)
                            if result:
                                rich_print(f"[green]{result}[/green]")
                        continue
                    elif "load_history" in command_dict:
                        # Load history for the current agent
                        if command_dict.get("error"):
                            rich_print(f"[red]{command_dict['error']}[/red]")
                            continue

                        filename = command_dict.get("filename")
                        try:
                            from fast_agent.mcp.prompts.prompt_load import load_prompt

                            # Load the messages from the file
                            messages = load_prompt(Path(filename))

                            # Get the agent object
                            agent_obj = prompt_provider._agent(agent)

                            # Clear the agent's history first
                            agent_obj.clear()

                            # Load the messages into the agent's history
                            # We use generate() to properly process the loaded history
                            await agent_obj.generate(messages)

                            rich_print(f"[green]History loaded from {filename}[/green]")
                        except FileNotFoundError:
                            rich_print(f"[red]File not found: {filename}[/red]")
                        except Exception as e:
                            rich_print(f"[red]Error loading history: {e}[/red]")
                        continue

                # Skip further processing if:
                # 1. The command was handled (command_result is truthy)
                # 2. The original input was a dictionary (special command like /prompt)
                # 3. The command result itself is a dictionary (special command handling result)
                # This fixes the issue where /prompt without arguments gets sent to the LLM
                if (
                    command_result
                    or isinstance(user_input, dict)
                    or isinstance(command_result, dict)
                ):
                    continue

                if user_input.upper() == "STOP":
                    return result
                if user_input == "":
                    continue

            # Send the message to the agent
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

    async def _get_all_prompts(self, prompt_provider: "AgentApp", agent_name: Optional[str] = None):
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
        requested_name: Optional[str] = None,
        send_func: Optional[SendFunc] = None,
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
                prompts: List[Prompt] = []
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

                prompt_names = [str(i + 1) for i in range(len(all_prompts))]

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
            assert hasattr(prompt_provider, "_agent"), (
                "Interactive prompt expects an AgentApp with _agent()"
            )
            agent = prompt_provider._agent(agent_name)

            rich_print(f"\n[bold]Skills for agent [cyan]{agent_name}[/cyan]:[/bold]")

            skill_manifests = getattr(agent, "_skill_manifests", None)
            manifests = list(skill_manifests) if skill_manifests else []

            if not manifests:
                rich_print("[yellow]No skills available for this agent[/yellow]")
                return

            rich_print()

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
                    import textwrap

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

        except Exception as exc:  # noqa: BLE001
            import traceback

            rich_print(f"[red]Error listing skills: {exc}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

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

            message_history = agent.llm.message_history
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
