"""Run an interactive agent directly from the command line."""

import asyncio
import shlex
import sys
from typing import Dict, List, Optional

import typer

from mcp_agent.cli.commands.server_helpers import add_servers_to_config, generate_server_name
from mcp_agent.cli.commands.url_parser import generate_server_configs, parse_server_urls
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.ui.console_display import ConsoleDisplay

app = typer.Typer(
    help="Run an interactive agent directly from the command line without creating an agent.py file",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


async def _run_agent(
    name: str = "fast-agent cli",
    instruction: str = "You are a helpful AI Agent.",
    config_path: Optional[str] = None,
    server_list: Optional[List[str]] = None,
    model: Optional[str] = None,
    message: Optional[str] = None,
    prompt_file: Optional[str] = None,
    url_servers: Optional[Dict[str, Dict[str, str]]] = None,
    stdio_servers: Optional[Dict[str, Dict[str, str]]] = None,
    agent_name: Optional[str] = "agent",
) -> None:
    """Async implementation to run an interactive agent."""
    from pathlib import Path

    from mcp_agent.mcp.prompts.prompt_load import load_prompt_multipart

    # Create the FastAgent instance

    fast_kwargs = {
        "name": name,
        "config_path": config_path,
        "ignore_unknown_args": True,
        "parse_cli_args": False,  # Don't parse CLI args, we're handling it ourselves
    }

    fast = FastAgent(**fast_kwargs)

    # Add all dynamic servers to the configuration
    await add_servers_to_config(fast, url_servers)
    await add_servers_to_config(fast, stdio_servers)

    # Check if we have multiple models (comma-delimited)
    if model and "," in model:
        # Parse multiple models
        models = [m.strip() for m in model.split(",") if m.strip()]

        # Create an agent for each model
        fan_out_agents = []
        for i, model_name in enumerate(models):
            agent_name = f"{model_name}"

            # Define the agent with specified parameters
            agent_kwargs = {"instruction": instruction, "name": agent_name}
            if server_list:
                agent_kwargs["servers"] = server_list
            agent_kwargs["model"] = model_name

            @fast.agent(**agent_kwargs)
            async def model_agent():
                pass

            fan_out_agents.append(agent_name)

        # Create a silent fan-in agent for cleaner output
        @fast.agent(
            name="aggregate",
            model="silent",
            instruction="You are a silent agent that combines outputs from parallel agents.",
        )
        async def fan_in_agent():
            pass

        # Create a parallel agent with silent fan_in
        @fast.parallel(
            name="parallel",
            fan_out=fan_out_agents,
            fan_in="aggregate",
            include_request=True,
        )
        async def cli_agent():
            async with fast.run() as agent:
                if message:
                    await agent.parallel.send(message)
                    display = ConsoleDisplay(config=None)
                    display.show_parallel_results(agent.parallel)
                elif prompt_file:
                    prompt = load_prompt_multipart(Path(prompt_file))
                    await agent.parallel.generate(prompt)
                    display = ConsoleDisplay(config=None)
                    display.show_parallel_results(agent.parallel)
                else:
                    await agent.interactive(agent_name="parallel", pretty_print_parallel=True)
    else:
        # Single model - use original behavior
        # Define the agent with specified parameters
        agent_kwargs = {"instruction": instruction}
        if agent_name:
            agent_kwargs["name"] = agent_name
        if server_list:
            agent_kwargs["servers"] = server_list
        if model:
            agent_kwargs["model"] = model

        @fast.agent(**agent_kwargs)
        async def cli_agent():
            async with fast.run() as agent:
                if message:
                    response = await agent.send(message)
                    # Print the response and exit
                    print(response)
                elif prompt_file:
                    prompt = load_prompt_multipart(Path(prompt_file))
                    response = await agent.generate(prompt)
                    # Print the response text and exit
                    print(response.last_text())
                else:
                    await agent.interactive()

    # Run the agent
    await cli_agent()


def run_async_agent(
    name: str,
    instruction: str,
    config_path: Optional[str] = None,
    servers: Optional[str] = None,
    urls: Optional[str] = None,
    auth: Optional[str] = None,
    model: Optional[str] = None,
    message: Optional[str] = None,
    prompt_file: Optional[str] = None,
    stdio_commands: Optional[List[str]] = None,
    agent_name: Optional[str] = None,
):
    """Run the async agent function with proper loop handling."""
    server_list = servers.split(",") if servers else None

    # Parse URLs and generate server configurations if provided
    url_servers = None
    if urls:
        try:
            parsed_urls = parse_server_urls(urls, auth)
            url_servers = generate_server_configs(parsed_urls)
            # If we have servers from URLs, add their names to the server_list
            if url_servers and not server_list:
                server_list = list(url_servers.keys())
            elif url_servers and server_list:
                # Merge both lists
                server_list.extend(list(url_servers.keys()))
        except ValueError as e:
            print(f"Error parsing URLs: {e}")
            return

    # Generate STDIO server configurations if provided
    stdio_servers = None

    if stdio_commands:
        stdio_servers = {}
        for i, stdio_cmd in enumerate(stdio_commands):
            # Parse the stdio command string
            try:
                parsed_command = shlex.split(stdio_cmd)
                if not parsed_command:
                    print(f"Error: Empty stdio command: {stdio_cmd}")
                    continue

                command = parsed_command[0]
                initial_args = parsed_command[1:] if len(parsed_command) > 1 else []

                # Generate a server name from the command
                if initial_args:
                    # Try to extract a meaningful name from the args
                    for arg in initial_args:
                        if arg.endswith(".py") or arg.endswith(".js") or arg.endswith(".ts"):
                            base_name = generate_server_name(arg)
                            break
                    else:
                        # Fallback to command name
                        base_name = generate_server_name(command)
                else:
                    base_name = generate_server_name(command)

                # Ensure unique server names when multiple servers
                server_name = base_name
                if len(stdio_commands) > 1:
                    server_name = f"{base_name}_{i + 1}"

                # Build the complete args list
                stdio_command_args = initial_args.copy()

                # Add this server to the configuration
                stdio_servers[server_name] = {
                    "transport": "stdio",
                    "command": command,
                    "args": stdio_command_args,
                }

                # Add STDIO server to the server list
                if not server_list:
                    server_list = [server_name]
                else:
                    server_list.append(server_name)

            except ValueError as e:
                print(f"Error parsing stdio command '{stdio_cmd}': {e}")
                continue

    # Check if we're already in an event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside a running event loop, so we can't use asyncio.run
            # Instead, create a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # No event loop exists, so we'll create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _run_agent(
                name=name,
                instruction=instruction,
                config_path=config_path,
                server_list=server_list,
                model=model,
                message=message,
                prompt_file=prompt_file,
                url_servers=url_servers,
                stdio_servers=stdio_servers,
                agent_name=agent_name,
            )
        )
    finally:
        try:
            # Clean up the loop
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            # Run the event loop until all tasks are done
            if sys.version_info >= (3, 7):
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception:
            pass


@app.callback(invoke_without_command=True, no_args_is_help=False)
def go(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent", "--name", help="Name for the agent"),
    instruction: Optional[str] = typer.Option(
        None, "--instruction", "-i", help="Path to file or URL containing instruction for the agent"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config-path", "-c", help="Path to config file"
    ),
    servers: Optional[str] = typer.Option(
        None, "--servers", help="Comma-separated list of server names to enable from config"
    ),
    urls: Optional[str] = typer.Option(
        None, "--url", help="Comma-separated list of HTTP/SSE URLs to connect to"
    ),
    auth: Optional[str] = typer.Option(
        None, "--auth", help="Bearer token for authorization with URL-based servers"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "--models", help="Override the default model (e.g., haiku, sonnet, gpt-4)"
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m", help="Message to send to the agent (skips interactive mode)"
    ),
    prompt_file: Optional[str] = typer.Option(
        None, "--prompt-file", "-p", help="Path to a prompt file to use (either text or JSON)"
    ),
    npx: Optional[str] = typer.Option(
        None, "--npx", help="NPX package and args to run as MCP server (quoted)"
    ),
    uvx: Optional[str] = typer.Option(
        None, "--uvx", help="UVX package and args to run as MCP server (quoted)"
    ),
    stdio: Optional[str] = typer.Option(
        None, "--stdio", help="Command to run as STDIO MCP server (quoted)"
    ),
) -> None:
    """
    Run an interactive agent directly from the command line.

    Examples:
        fast-agent go --model=haiku --instruction=./instruction.md --servers=fetch,filesystem
        fast-agent go --instruction=https://raw.githubusercontent.com/user/repo/prompt.md
        fast-agent go --message="What is the weather today?" --model=haiku
        fast-agent go --prompt-file=my-prompt.txt --model=haiku
        fast-agent go --url=http://localhost:8001/mcp,http://api.example.com/sse
        fast-agent go --url=https://api.example.com/mcp --auth=YOUR_API_TOKEN
        fast-agent go --npx "@modelcontextprotocol/server-filesystem /path/to/data"
        fast-agent go --uvx "mcp-server-fetch --verbose"
        fast-agent go --stdio "python my_server.py --debug"
        fast-agent go --stdio "uv run server.py --config=settings.json"

    This will start an interactive session with the agent, using the specified model
    and instruction. It will use the default configuration from fastagent.config.yaml
    unless --config-path is specified.

    Common options:
        --model               Override the default model (e.g., --model=haiku)
        --quiet               Disable progress display and logging
        --servers             Comma-separated list of server names to enable from config
        --url                 Comma-separated list of HTTP/SSE URLs to connect to
        --auth                Bearer token for authorization with URL-based servers
        --message, -m         Send a single message and exit
        --prompt-file, -p     Use a prompt file instead of interactive mode
        --npx                 NPX package and args to run as MCP server (quoted)
        --uvx                 UVX package and args to run as MCP server (quoted)
        --stdio               Command to run as STDIO MCP server (quoted)
    """
    # Collect all stdio commands from convenience options
    stdio_commands = []

    if npx:
        stdio_commands.append(f"npx {npx}")

    if uvx:
        stdio_commands.append(f"uvx {uvx}")

    if stdio:
        stdio_commands.append(stdio)

    # Resolve instruction from file/URL or use default
    resolved_instruction = "You are a helpful AI Agent."  # Default
    agent_name = "agent"

    if instruction:
        try:
            from pathlib import Path

            from pydantic import AnyUrl

            from mcp_agent.core.direct_decorators import _resolve_instruction

            # Check if it's a URL
            if instruction.startswith(("http://", "https://")):
                resolved_instruction = _resolve_instruction(AnyUrl(instruction))
            else:
                # Treat as file path
                resolved_instruction = _resolve_instruction(Path(instruction))
                # Extract filename without extension to use as agent name
                instruction_path = Path(instruction)
                if instruction_path.exists() and instruction_path.is_file():
                    # Get filename without extension
                    agent_name = instruction_path.stem
        except Exception as e:
            typer.echo(f"Error loading instruction from {instruction}: {e}", err=True)
            raise typer.Exit(1)

    run_async_agent(
        name=name,
        instruction=resolved_instruction,
        config_path=config_path,
        servers=servers,
        urls=urls,
        auth=auth,
        model=model,
        message=message,
        prompt_file=prompt_file,
        stdio_commands=stdio_commands,
        agent_name=agent_name,
    )
