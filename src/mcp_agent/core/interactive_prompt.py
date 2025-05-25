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
        apply_prompt_func=agent_app.apply_prompt
    )
"""

import asyncio
import json
from typing import Dict, List, Optional, Union, Any

from rich import print as rich_print
from rich.console import Console
from rich.table import Table

from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.enhanced_prompt import (
    get_argument_input,
    get_enhanced_input,
    get_selection_input,
    handle_special_commands,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.mcp_aggregator import SEP  # Import SEP once at the top
from mcp_agent.progress_display import progress_display

logger = get_logger(__name__)


class RedisPubSubInputHandler:
    """
    Handler for Redis PubSub input in interactive mode.
    
    This class enables an agent to receive messages from a Redis PubSub channel
    in addition to standard input, allowing for remote interaction with the agent.
    """
    
    def __init__(self, redis_client, channel_name: str) -> None:
        """
        Initialize the Redis PubSub input handler.
        
        Args:
            redis_client: Redis client instance
            channel_name: Name of the channel to subscribe to
        """
        self.redis_client = redis_client
        self.channel_name = channel_name
        self.pubsub = redis_client.pubsub()
        self.message_queue = asyncio.Queue()
        self._listener_task = None
        self._stop_event = asyncio.Event()
        self.last_message_info = {}  # Store info about the last message for responses
        
        # Extract the agent ID from the channel name (everything after the last ':')
        if ':' in channel_name:
            self.agent_id = channel_name.split(':')[-1]
        else:
            self.agent_id = channel_name
        
        # Define the response channel name using the same prefix but with _response suffix
        if ':' in channel_name:
            prefix = channel_name.rsplit(':', 1)[0]
            self.response_channel = f"{prefix}:{self.agent_id}_response"
        else:
            self.response_channel = f"{channel_name}_response"
            
        logger.info(f"Initialized Redis PubSub input handler for channel: {channel_name}")
        logger.info(f"Response channel for {channel_name} will be {self.response_channel}")
    
    async def start_listening(self) -> None:
        """
        Start listening for messages on the Redis channel.
        
        This method subscribes to the Redis channel and starts a background task
        to process incoming messages.
        """
        logger.info(f"Subscribing to Redis channel: {self.channel_name}")
        await self.pubsub.subscribe(self.channel_name)
        self._listener_task = asyncio.create_task(self._listen_for_messages())
        logger.info(f"Started Redis PubSub listener for channel: {self.channel_name}")
    
    async def stop_listening(self) -> None:
        """
        Stop listening for messages and clean up resources.
        """
        if self._listener_task:
            self._stop_event.set()
            await self.pubsub.unsubscribe(self.channel_name)
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped Redis PubSub listener for channel: {self.channel_name}")
    
    async def _listen_for_messages(self) -> None:
        """
        Background task that listens for Redis messages and puts them in the queue.
        """
        logger.info(f"Redis listener loop started for channel: {self.channel_name}")
        try:
            while not self._stop_event.is_set():
                message = await self.pubsub.get_message(ignore_subscribe_messages=True)
                
                if message and message.get('type') == 'message':
                    try:
                        # Process the message data
                        data = message.get('data')
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        
                        # Try to parse JSON
                        try:
                            data = json.loads(data)
                            logger.debug(f"Received JSON message from Redis: {data}")
                            
                            # If this is a user message, extract content and add to queue
                            if data.get('type') == 'user' and 'content' in data:
                                # Get message content and metadata
                                user_input = data['content']
                                metadata = data.get('metadata', {})
                                channel_id = data.get('channel_id', self.agent_id)
                                
                                # Store the original message info for potential response
                                self.last_message_info = {
                                    'channel_id': channel_id,
                                    'metadata': metadata
                                }
                                
                                logger.info(f"Adding user input from Redis to queue: {user_input}")
                                await self.message_queue.put(user_input)
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON message from Redis: {data}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Small delay to prevent CPU spike
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Redis listener task was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Redis listener: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def publish_response(self, response: str, agent_name: str) -> bool:
        """
        Publish a response to the response channel.
        
        Args:
            response: The response content
            agent_name: The name of the agent sending the response
            
        Returns:
            bool: True if publishing was successful, False otherwise
        """
        try:
            import json
            from datetime import datetime
            
            # Get metadata from the last message if available
            metadata = getattr(self, 'last_message_info', {}).get('metadata', {})
            channel_id = getattr(self, 'last_message_info', {}).get('channel_id', self.agent_id)
            
            # Create response message
            # Handle metadata properly
            meta_dict = {}
            if isinstance(metadata, dict):
                meta_dict.update(metadata)
            
            # Add our own metadata
            meta_dict["source"] = agent_name
            meta_dict["timestamp"] = datetime.now().isoformat()
            
            response_data = {
                "type": "assistant",
                "content": response,
                "channel_id": channel_id,
                "metadata": meta_dict
            }
            
            # Publish to response channel
            await self.redis_client.publish(
                self.response_channel,
                json.dumps(response_data)
            )
            
            logger.info(f"Published response to Redis channel: {self.response_channel}")
            return True
        except Exception as e:
            logger.error(f"Error publishing response to Redis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def get_message(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Get the next message from the queue, with optional timeout.
        
        Args:
            timeout: Timeout in seconds to wait for a message
            
        Returns:
            The next message, or None if timeout is reached
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
            else:
                # No timeout - this would block indefinitely if no message comes
                return await self.message_queue.get()
        except asyncio.TimeoutError:
            return None


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
        self._redis_input_handler: Optional[RedisPubSubInputHandler] = None

    async def set_redis_input_handler(self, redis_client, channel_name: str) -> None:
        """
        Set up a Redis PubSub input handler for this prompt.
        
        Args:
            redis_client: Redis client instance
            channel_name: Name of the channel to subscribe to
        """
        # Clean up any existing handler
        if self._redis_input_handler:
            await self._redis_input_handler.stop_listening()
        
        # Create and start a new handler
        self._redis_input_handler = RedisPubSubInputHandler(redis_client, channel_name)
        await self._redis_input_handler.start_listening()
        logger.info(f"Redis input handler set up for channel: {channel_name}")

    async def prompt_loop(
        self,
        send_func,
        default_agent: str,
        available_agents: List[str],
        apply_prompt_func=None,
        list_prompts_func=None,
        default: str = "",
        redis_poll_interval: float = 0.1,
        use_terminal_input: bool = False  # New parameter to control terminal input
    ) -> str:
        """
        Start an interactive prompt session.

        Args:
            send_func: Function to send messages to agents (signature: async (message, agent_name))
            default_agent: Name of the default agent to use
            available_agents: List of available agent names
            apply_prompt_func: Optional function to apply prompts (signature: async (name, args, agent))
            list_prompts_func: Optional function to list available prompts (signature: async (agent_name))
            default: Default message to use when user presses enter
            redis_poll_interval: How often to check for Redis messages (seconds)
            use_terminal_input: Whether to use terminal input or only Redis (defaults to False)

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
        
        # If we're only using Redis and there's no Redis handler, we can't proceed
        if not use_terminal_input and not self._redis_input_handler:
            logger.error("Redis input handler not configured but terminal input is disabled")
            return "Error: Redis input handler not configured but terminal input is disabled"
        
        # Log the mode we're operating in
        if use_terminal_input:
            logger.info("Interactive prompt running with terminal input enabled")
        else:
            logger.info("Interactive prompt running in Redis-only mode (terminal input disabled)")
            # Print a message to the console so the user knows what's happening
            rich_print("[bold green]Agent running in Redis-only mode. Waiting for messages on Redis channel...[/bold green]")
            rich_print("[bold]Send CTRL+C to exit[/bold]")
            
        # Main interactive loop
        try:
            while True:
                with progress_display.paused():
                    user_input = None
                    
                    if use_terminal_input:
                        # Create an event to signal when stdin input is ready
                        stdin_ready = asyncio.Event()
                        stdin_buffer = []
                        
                        # This will run the stdin reader concurrently with Redis polling
                        async def get_stdin_input():
                            nonlocal stdin_buffer
                            try:
                                # Use the enhanced input method with advanced features
                                user_input = await get_enhanced_input(
                                    agent_name=agent,
                                    default=default,
                                    show_default=(default != ""),
                                    show_stop_hint=True,
                                    multiline=False,  # Default to single-line mode
                                    available_agent_names=available_agents,
                                    agent_types=self.agent_types,  # Pass agent types for display
                                )
                                stdin_buffer.append(user_input)
                                stdin_ready.set()
                            except Exception as e:
                                logger.error(f"Error getting stdin input: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                stdin_buffer.append(None)  # Mark error
                                stdin_ready.set()
                        
                        # Start stdin reader task if terminal input is enabled
                        stdin_task = asyncio.create_task(get_stdin_input())
                    
                    # Wait for either stdin (if enabled) or Redis input
                    while True:
                        # Check if we have Redis input available (if handler is configured)
                        redis_message = None
                        if self._redis_input_handler:
                            redis_message = await self._redis_input_handler.get_message(timeout=redis_poll_interval)
                            
                        if redis_message:
                            # Got a message from Redis
                            user_input = redis_message
                            logger.info(f"Processing Redis message: {user_input}")
                            
                            # Cancel stdin task if we're using terminal input
                            if use_terminal_input:
                                stdin_task.cancel()
                                try:
                                    await stdin_task
                                except asyncio.CancelledError:
                                    pass
                            break
                            
                        # Check if stdin input is ready (if terminal input is enabled)
                        if use_terminal_input and stdin_ready.is_set():
                            # Get input from buffer
                            user_input = stdin_buffer[0] if stdin_buffer else None
                            logger.info(f"Processing stdin input: {user_input}")
                            break
                            
                        # If we're only using Redis, we just wait for Redis messages
                        if not use_terminal_input:
                            # Just sleep a little to prevent CPU spike
                            await asyncio.sleep(0.01)
                        else:
                            # If both terminal and Redis are enabled, we wait for either
                            await asyncio.sleep(0.01)
                    
                    # If we got None as input (error case), continue to next iteration
                    if user_input is None:
                        continue
                    
                    # Handle special commands - pass "True" to enable agent switching
                    command_result = await handle_special_commands(user_input, True)

                    # Check if we should switch agents
                    if isinstance(command_result, dict):
                        if "switch_agent" in command_result:
                            new_agent = command_result["switch_agent"]
                            if new_agent in available_agents_set:
                                agent = new_agent
                                continue
                            else:
                                if use_terminal_input:  # Only show in terminal if terminal input is enabled
                                    rich_print(f"[red]Agent '{new_agent}' not found[/red]")
                                logger.warning(f"Agent '{new_agent}' not found")
                                continue
                        # Keep the existing list_prompts handler for backward compatibility
                        elif "list_prompts" in command_result and list_prompts_func:
                            # Use the list_prompts_func directly
                            if use_terminal_input:  # Only show in terminal if terminal input is enabled
                                await self._list_prompts(list_prompts_func, agent)
                            continue
                        elif "select_prompt" in command_result and (
                            list_prompts_func and apply_prompt_func
                        ):
                            # Handle prompt selection, using both list_prompts and apply_prompt
                            prompt_name = command_result.get("prompt_name")
                            prompt_index = command_result.get("prompt_index")

                            # If a specific index was provided (from /prompt <number>)
                            if prompt_index is not None:
                                # First get a list of all prompts to look up the index
                                all_prompts = await self._get_all_prompts(list_prompts_func, agent)
                                if not all_prompts:
                                    if use_terminal_input:  # Only show in terminal if terminal input is enabled
                                        rich_print("[yellow]No prompts available[/yellow]")
                                    logger.warning("No prompts available")
                                    continue

                                # Check if the index is valid
                                if 1 <= prompt_index <= len(all_prompts):
                                    # Get the prompt at the specified index (1-based to 0-based)
                                    selected_prompt = all_prompts[prompt_index - 1]
                                    # Use the already created namespaced_name to ensure consistency
                                    await self._select_prompt(
                                        list_prompts_func,
                                        apply_prompt_func,
                                        agent,
                                        selected_prompt["namespaced_name"],
                                    )
                                else:
                                    if use_terminal_input:  # Only show in terminal if terminal input is enabled
                                        rich_print(
                                            f"[red]Invalid prompt number: {prompt_index}. Valid range is 1-{len(all_prompts)}[/red]"
                                        )
                                        # Show the prompt list for convenience
                                        await self._list_prompts(list_prompts_func, agent)
                                    logger.warning(f"Invalid prompt number: {prompt_index}")
                                    continue
                            else:
                                # Use the name-based selection
                                await self._select_prompt(
                                    list_prompts_func, apply_prompt_func, agent, prompt_name
                                )
                            continue

                    # Skip further processing if:
                    # 1. The command was handled (command_result is truthy)
                    # 2. The original input was a dictionary (special command like /prompt)
                    # 3. The command result itself is a dictionary (special command handling result)
                    # This fixes the issue where /prompt without arguments gets sent to the LLM
                    if command_result or isinstance(user_input, dict) or isinstance(command_result, dict):
                        continue

                    if user_input.upper() == "STOP":
                        break
                    if user_input == "":
                        continue

                # Send the message to the agent
                logger.info(f"Sending message to agent {agent}: {user_input}")
                result = await send_func(user_input, agent)
                logger.info(f"Got response from agent {agent}: {result}")
                
                # If we're using Redis-only mode, send the response back to a response channel
                if not use_terminal_input and self._redis_input_handler:
                    # Use the publish_response method of the Redis handler
                    asyncio.create_task(self._redis_input_handler.publish_response(result, agent))
                        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, exiting prompt loop")
        except asyncio.CancelledError:
            logger.info("Prompt loop task cancelled")
        except Exception as e:
            logger.error(f"Error in prompt loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Clean up resources
            if self._redis_input_handler:
                await self._redis_input_handler.stop_listening()
            
        return result

    async def _get_all_prompts(self, list_prompts_func, agent_name):
        """
        Get a list of all available prompts.

        Args:
            list_prompts_func: Function to get available prompts
            agent_name: Name of the agent

        Returns:
            List of prompt info dictionaries, sorted by server and name
        """
        try:
            # Pass None instead of agent_name to get prompts from all servers
            # the agent_name parameter should never be used as a server name
            prompt_servers = await list_prompts_func(None)
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
                                    "description": getattr(prompt, "description", "No description"),
                                    "arg_count": len(getattr(prompt, "arguments", [])),
                                    "arguments": getattr(prompt, "arguments", []),
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
                                        "description": prompt.get("description", "No description"),
                                        "arg_count": len(prompt.get("arguments", [])),
                                        "arguments": prompt.get("arguments", []),
                                    }
                                )
                            else:
                                all_prompts.append(
                                    {
                                        "server": server_name,
                                        "name": str(prompt),
                                        "namespaced_name": f"{server_name}{SEP}{str(prompt)}",
                                        "description": "No description",
                                        "arg_count": 0,
                                        "arguments": [],
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

    async def _list_prompts(self, list_prompts_func, agent_name) -> None:
        """
        List available prompts for an agent.

        Args:
            list_prompts_func: Function to get available prompts
            agent_name: Name of the agent
        """
        from rich import print as rich_print
        from rich.console import Console
        from rich.table import Table

        console = Console()

        try:
            # Directly call the list_prompts function for this agent
            rich_print(f"\n[bold]Fetching prompts for agent [cyan]{agent_name}[/cyan]...[/bold]")

            # Get all prompts using the helper function - pass None as server name
            # to get prompts from all available servers
            all_prompts = await self._get_all_prompts(list_prompts_func, None)

            if all_prompts:
                # Create a table for better display
                table = Table(title="Available MCP Prompts")
                table.add_column("#", justify="right", style="cyan")
                table.add_column("Server", style="green")
                table.add_column("Prompt Name", style="bright_blue")
                table.add_column("Description")
                table.add_column("Args", justify="center")

                # Add prompts to table
                for i, prompt in enumerate(all_prompts):
                    table.add_row(
                        str(i + 1),
                        prompt["server"],
                        prompt["name"],
                        prompt["description"],
                        str(prompt["arg_count"]),
                    )

                console.print(table)

                # Add usage instructions
                rich_print("\n[bold]Usage:[/bold]")
                rich_print("  • Use [cyan]/prompt <number>[/cyan] to select a prompt by number")
                rich_print("  • Or use [cyan]/prompts[/cyan] to open the prompt selection UI")
            else:
                rich_print("[yellow]No prompts available[/yellow]")
        except Exception as e:
            import traceback

            rich_print(f"[red]Error listing prompts: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

    async def _select_prompt(
        self, list_prompts_func, apply_prompt_func, agent_name, requested_name=None
    ) -> None:
        """
        Select and apply a prompt.

        Args:
            list_prompts_func: Function to get available prompts
            apply_prompt_func: Function to apply prompts
            agent_name: Name of the agent
            requested_name: Optional name of the prompt to apply
        """
        # We already imported these at the top
        from rich import print as rich_print

        console = Console()

        try:
            # Get all available prompts directly from the list_prompts function
            rich_print(f"\n[bold]Fetching prompts for agent [cyan]{agent_name}[/cyan]...[/bold]")
            # IMPORTANT: list_prompts_func gets MCP server prompts, not agent prompts
            # So we pass None to get prompts from all servers, not using agent_name as server name
            prompt_servers = await list_prompts_func(None)

            if not prompt_servers:
                rich_print("[yellow]No prompts available for this agent[/yellow]")
                return

            # Process fetched prompts
            all_prompts = []
            for server_name, prompts_info in prompt_servers.items():
                if not prompts_info:
                    continue

                # Extract prompts
                prompts = []
                if hasattr(prompts_info, "prompts"):
                    prompts = prompts_info.prompts
                elif isinstance(prompts_info, list):
                    prompts = prompts_info

                # Process each prompt
                for prompt in prompts:
                    # Get basic prompt info
                    prompt_name = getattr(prompt, "name", "Unknown")
                    description = getattr(prompt, "description", "No description")

                    # Extract argument information
                    arg_names = []
                    required_args = []
                    optional_args = []
                    arg_descriptions = {}

                    # Get arguments list
                    arguments = getattr(prompt, "arguments", None)
                    if arguments:
                        for arg in arguments:
                            name = getattr(arg, "name", None)
                            if name:
                                arg_names.append(name)

                                # Store description if available
                                description = getattr(arg, "description", None)
                                if description:
                                    arg_descriptions[name] = description

                                # Check if required
                                if getattr(arg, "required", False):
                                    required_args.append(name)
                                else:
                                    optional_args.append(name)

                    # Create namespaced version using the consistent separator
                    namespaced_name = f"{server_name}{SEP}{prompt_name}"

                    # Add to collection
                    all_prompts.append(
                        {
                            "server": server_name,
                            "name": prompt_name,
                            "namespaced_name": namespaced_name,
                            "description": description,
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
                # Show prompt selection UI
                table = Table(title="Available MCP Prompts")
                table.add_column("#", justify="right", style="cyan")
                table.add_column("Server", style="green")
                table.add_column("Prompt Name", style="bright_blue")
                table.add_column("Description")
                table.add_column("Args", justify="center")

                # Add prompts to table
                for i, prompt in enumerate(all_prompts):
                    required_args = prompt["required_args"]
                    optional_args = prompt["optional_args"]

                    # Format args column
                    if required_args and optional_args:
                        args_display = f"[bold]{len(required_args)}[/bold]+{len(optional_args)}"
                    elif required_args:
                        args_display = f"[bold]{len(required_args)}[/bold]"
                    elif optional_args:
                        args_display = f"{len(optional_args)} opt"
                    else:
                        args_display = "0"

                    table.add_row(
                        str(i + 1),
                        prompt["server"],
                        prompt["name"],
                        prompt["description"] or "No description",
                        args_display,
                    )

                console.print(table)
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

            # Apply the prompt
            namespaced_name = selected_prompt["namespaced_name"]
            rich_print(f"\n[bold]Applying prompt [cyan]{namespaced_name}[/cyan]...[/bold]")

            # Call apply_prompt function with the prompt name and arguments
            await apply_prompt_func(namespaced_name, arg_values, agent_name)

        except Exception as e:
            import traceback

            rich_print(f"[red]Error selecting or applying prompt: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")
