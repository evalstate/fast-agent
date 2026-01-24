"""Run an interactive agent directly from the command line."""

import asyncio
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Literal, cast

import typer

from fast_agent.cli.asyncio_utils import set_asyncio_exception_handler
from fast_agent.cli.commands.server_helpers import add_servers_to_config, generate_server_name
from fast_agent.cli.commands.url_parser import generate_server_configs, parse_server_urls
from fast_agent.cli.constants import RESUME_LATEST_SENTINEL
from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION, FAST_AGENT_SHELL_CHILD_ENV
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.paths import resolve_environment_paths
from fast_agent.utils.async_utils import configure_uvloop, create_event_loop, ensure_event_loop

app = typer.Typer(
    help="Run an interactive agent directly from the command line without creating an agent.py file",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

default_instruction = DEFAULT_AGENT_INSTRUCTION

CARD_EXTENSIONS = {".md", ".markdown", ".yaml", ".yml"}

DEFAULT_ENV_PATHS = resolve_environment_paths()
DEFAULT_AGENT_CARDS_DIR = DEFAULT_ENV_PATHS.agent_cards
DEFAULT_TOOL_CARDS_DIR = DEFAULT_ENV_PATHS.tool_cards


def _merge_card_sources(
    sources: list[str] | None,
    default_dir: Path,
) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()
    if sources:
        for entry in sources:
            if entry not in seen:
                merged.append(entry)
                seen.add(entry)
        return merged
    if default_dir.is_dir():
        has_cards = any(
            entry.is_file() and entry.suffix.lower() in CARD_EXTENSIONS
            for entry in default_dir.iterdir()
        )
        if has_cards:
            merged.append(str(default_dir))
    return merged or None


def resolve_instruction_option(instruction: str | None) -> tuple[str, str]:
    """
    Resolve the instruction option (file or URL) to the instruction string and agent name.
    Returns (resolved_instruction, agent_name).
    """
    resolved_instruction = default_instruction
    agent_name = "agent"

    if instruction:
        try:
            from pathlib import Path

            from pydantic import AnyUrl

            from fast_agent.core.direct_decorators import _resolve_instruction

            if instruction.startswith(("http://", "https://")):
                resolved_instruction = _resolve_instruction(AnyUrl(instruction))
            else:
                resolved_instruction = _resolve_instruction(Path(instruction))
                instruction_path = Path(instruction)
                if instruction_path.exists() and instruction_path.is_file():
                    agent_name = instruction_path.stem
        except Exception as e:
            typer.echo(f"Error loading instruction from {instruction}: {e}", err=True)
            raise typer.Exit(1)

    return resolved_instruction, agent_name


def collect_stdio_commands(npx: str | None, uvx: str | None, stdio: str | None) -> list[str]:
    """Collect STDIO command definitions from convenience options."""
    stdio_commands: list[str] = []

    if npx:
        stdio_commands.append(f"npx {npx}")
    if uvx:
        stdio_commands.append(f"uvx {uvx}")
    if stdio:
        stdio_commands.append(stdio)

    return stdio_commands


async def _run_agent(
    name: str = "fast-agent cli",
    instruction: str = default_instruction,
    config_path: str | None = None,
    server_list: list[str] | None = None,
    agent_cards: list[str] | None = None,
    card_tools: list[str] | None = None,
    model: str | None = None,
    message: str | None = None,
    prompt_file: str | None = None,
    resume: str | None = None,
    url_servers: dict[str, dict[str, Any]] | None = None,
    stdio_servers: dict[str, dict[str, Any]] | None = None,
    agent_name: str | None = "agent",
    skills_directory: Path | None = None,
    environment_dir: Path | None = None,
    shell_runtime: bool = False,
    mode: Literal["interactive", "serve"] = "interactive",
    transport: str = "http",
    host: str = "0.0.0.0",
    port: int = 8000,
    tool_description: str | None = None,
    instance_scope: str = "shared",
    permissions_enabled: bool = True,
    reload: bool = False,
    watch: bool = False,
) -> None:
    """Async implementation to run an interactive agent."""
    if mode == "serve" and transport in ["stdio", "acp"]:
        from fast_agent.ui.console import configure_console_stream

        configure_console_stream("stderr")

    from fast_agent import FastAgent
    from fast_agent.agents.llm_agent import LlmAgent
    from fast_agent.mcp.prompts.prompt_load import load_prompt
    from fast_agent.ui.console_display import ConsoleDisplay

    def _resume_session_if_requested(agent_app) -> None:
        if not resume:
            return
        from fast_agent.session import get_session_manager
        from fast_agent.ui.enhanced_prompt import queue_startup_notice

        manager = get_session_manager()
        session_id = None if resume in ("", RESUME_LATEST_SENTINEL) else resume
        default_agent = agent_app._agent(None)
        result = manager.resume_session_agents(
            agent_app._agents,
            session_id,
            default_agent_name=getattr(default_agent, "name", None),
        )
        interactive_notice = message is None and prompt_file is None
        if not result:
            if session_id:
                notice = f"[yellow]Session not found:[/yellow] {session_id}"
            else:
                notice = "[yellow]No sessions found to resume.[/yellow]"
            if interactive_notice:
                queue_startup_notice(notice)
            else:
                typer.echo(
                    notice.replace("[yellow]", "").replace("[/yellow]", ""),
                    err=True,
                )
            return

        session, loaded, missing_agents = result
        session_time = session.info.last_activity.strftime("%y-%m-%d %H:%M")
        resume_notice = (
            f"[dim]Resumed session[/dim] [cyan]{session.info.name}[/cyan] [dim]({session_time})[/dim]"
        )
        if interactive_notice:
            queue_startup_notice(resume_notice)
        else:
            typer.echo(
                f"Resumed session {session.info.name} ({session_time})", err=True
            )
        if missing_agents:
            missing_list = ", ".join(sorted(missing_agents))
            missing_notice = f"[yellow]Missing agents from session:[/yellow] {missing_list}"
            if interactive_notice:
                queue_startup_notice(missing_notice)
            else:
                typer.echo(
                    f"Missing agents from session: {missing_list}", err=True
                )
        if missing_agents or not loaded:
            from fast_agent.session import format_history_summary, summarize_session_histories

            summary = summarize_session_histories(session)
            summary_text = format_history_summary(summary)
            if summary_text:
                summary_notice = (
                    f"[dim]Available histories:[/dim] {summary_text}"
                )
                if interactive_notice:
                    queue_startup_notice(summary_notice)
                else:
                    typer.echo(
                        f"Available histories: {summary_text}", err=True
                    )

    # Create the FastAgent instance

    fast = FastAgent(
        name=name,
        config_path=config_path,
        ignore_unknown_args=True,
        parse_cli_args=False,  # Don't parse CLI args, we're handling it ourselves
        quiet=mode == "serve",
        skills_directory=skills_directory,
        environment_dir=environment_dir,
    )

    # Set model on args so model source detection works correctly
    if model:
        fast.args.model = model
    fast.args.reload = reload
    fast.args.watch = watch
    fast.args.agent = agent_name or "agent"

    if shell_runtime:
        await fast.app.initialize()
        setattr(fast.app.context, "shell_runtime", True)

    # Add all dynamic servers to the configuration
    if url_servers:
        await add_servers_to_config(fast, cast("dict[str, dict[str, Any]]", url_servers))
    if stdio_servers:
        await add_servers_to_config(fast, cast("dict[str, dict[str, Any]]", stdio_servers))

    if agent_cards or card_tools:
        try:
            if agent_cards:
                for card_source in agent_cards:
                    if card_source.startswith(("http://", "https://")):
                        fast.load_agents_from_url(card_source)
                    else:
                        fast.load_agents(card_source)

            # Check if any loaded agent card has default: true
            has_explicit_default = False
            for agent_data in fast.agents.values():
                config = agent_data.get("config")
                if config and getattr(config, "default", False):
                    has_explicit_default = True
                    break

            # If no explicit default, create a fallback "agent" as the default
            # This must happen BEFORE loading card_tools so "agent" exists for attachment
            if not has_explicit_default:

                @fast.agent(
                    name="agent",
                    instruction=instruction,
                    servers=server_list or [],
                    model=model,
                    default=True,
                )
                async def default_fallback_agent():
                    pass

            tool_loaded_names: list[str] = []
            if card_tools:
                for card_source in card_tools:
                    if card_source.startswith(("http://", "https://")):
                        tool_loaded_names.extend(fast.load_agents_from_url(card_source))
                    else:
                        tool_loaded_names.extend(fast.load_agents(card_source))

            if tool_loaded_names:
                # Use explicit agent_name if provided and exists, otherwise find the default
                target_name = agent_name if agent_name and agent_name in fast.agents else None
                if not target_name:
                    target_name = fast.get_default_agent_name()
                if target_name:
                    fast.attach_agent_tools(target_name, tool_loaded_names)
        except AgentConfigError as exc:
            fast._handle_error(exc)
            raise typer.Exit(1) from exc

        # Add CLI servers (--url, --servers, etc.) to the default agent
        if server_list:
            default_agent_data = None
            for agent_data in fast.agents.values():
                config = agent_data.get("config")
                if config and getattr(config, "default", False):
                    default_agent_data = agent_data
                    break
            # If no explicit default, use the first agent
            if default_agent_data is None and fast.agents:
                default_agent_data = next(iter(fast.agents.values()))
            if default_agent_data:
                config = default_agent_data.get("config")
                if config:
                    existing = list(config.servers) if config.servers else []
                    config.servers = existing + [s for s in server_list if s not in existing]

        async def cli_agent():
            async with fast.run() as agent:
                _resume_session_if_requested(agent)
                if message:
                    response = await agent.send(message)
                    print(response)
                elif prompt_file:
                    prompt = load_prompt(Path(prompt_file))
                    agent_obj = agent._agent(None)
                    await agent_obj.generate(prompt)
                    print(f"\nLoaded {len(prompt)} messages from prompt file '{prompt_file}'")
                    await agent.interactive()
                else:
                    await agent.interactive()
    # Check if we have multiple models (comma-delimited)
    elif model and "," in model:
        # Parse multiple models
        models = [m.strip() for m in model.split(",") if m.strip()]

        # Create an agent for each model
        fan_out_agents = []
        for i, model_name in enumerate(models):
            agent_name = f"{model_name}"

            # Define the agent with specified parameters
            @fast.agent(
                name=agent_name,
                instruction=instruction,
                servers=server_list or [],
                model=model_name,
            )
            async def model_agent():
                pass

            fan_out_agents.append(agent_name)

        # Create a silent fan-in agent (suppresses display output)
        class SilentFanInAgent(LlmAgent):
            async def show_assistant_message(self, *args, **kwargs):  # type: ignore[override]
                return None

            def show_user_message(self, *args, **kwargs):  # type: ignore[override]
                return None

        @fast.custom(
            SilentFanInAgent,
            name="aggregate",
            model="passthrough",
            instruction="You aggregate parallel outputs without displaying intermediate messages.",
        )
        async def aggregate():
            pass

        # Create a parallel agent with silent fan_in
        @fast.parallel(
            name="parallel",
            fan_out=fan_out_agents,
            fan_in="aggregate",
            include_request=True,
            default=True,
        )
        async def cli_agent():
            async with fast.run() as agent:
                _resume_session_if_requested(agent)
                if message:
                    await agent.parallel.send(message)
                    display = ConsoleDisplay(config=None)
                    display.show_parallel_results(agent.parallel)
                elif prompt_file:
                    prompt = load_prompt(Path(prompt_file))
                    await agent.parallel.generate(prompt)
                    display = ConsoleDisplay(config=None)
                    display.show_parallel_results(agent.parallel)
                else:
                    await agent.interactive(pretty_print_parallel=True)
    else:
        # Single model - use original behavior
        # Define the agent with specified parameters
        @fast.agent(
            name=agent_name or "agent",
            instruction=instruction,
            servers=server_list or [],
            model=model,
            default=True,
        )
        async def cli_agent():
            async with fast.run() as agent:
                _resume_session_if_requested(agent)
                if message:
                    response = await agent.send(message)
                    # Print the response and exit
                    print(response)
                elif prompt_file:
                    prompt = load_prompt(Path(prompt_file))
                    response = await agent.agent.generate(prompt)
                    print(f"\nLoaded {len(prompt)} messages from prompt file '{prompt_file}'")
                    await agent.interactive()
                else:
                    await agent.interactive()

    # Run the agent
    if mode == "serve":
        await fast.start_server(
            transport=transport,
            host=host,
            port=port,
            tool_description=tool_description,
            instance_scope=instance_scope,
            permissions_enabled=permissions_enabled,
        )
    else:
        await cli_agent()


def run_async_agent(
    name: str,
    instruction: str,
    config_path: str | None = None,
    servers: str | None = None,
    urls: str | None = None,
    auth: str | None = None,
    agent_cards: list[str] | None = None,
    card_tools: list[str] | None = None,
    model: str | None = None,
    message: str | None = None,
    prompt_file: str | None = None,
    resume: str | None = None,
    stdio_commands: list[str] | None = None,
    agent_name: str | None = None,
    skills_directory: Path | None = None,
    environment_dir: Path | None = None,
    shell_enabled: bool = False,
    mode: Literal["interactive", "serve"] = "interactive",
    transport: str = "http",
    host: str = "0.0.0.0",
    port: int = 8000,
    tool_description: str | None = None,
    instance_scope: str = "shared",
    permissions_enabled: bool = True,
    reload: bool = False,
    watch: bool = False,
):
    """Run the async agent function with proper loop handling."""
    configure_uvloop()
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
            print(f"Error parsing URLs: {e}", file=sys.stderr)
            sys.exit(1)

    # Generate STDIO server configurations if provided
    stdio_servers = None

    if stdio_commands:
        stdio_servers = {}
        for i, stdio_cmd in enumerate(stdio_commands):
            # Parse the stdio command string
            try:
                parsed_command = shlex.split(stdio_cmd)
                if not parsed_command:
                    print(f"Error: Empty stdio command: {stdio_cmd}", file=sys.stderr)
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
                print(f"Error parsing stdio command '{stdio_cmd}': {e}", file=sys.stderr)
                continue

    if environment_dir:
        env_paths = resolve_environment_paths(override=environment_dir)
        default_agent_cards_dir = env_paths.agent_cards
        default_tool_cards_dir = env_paths.tool_cards
    else:
        default_agent_cards_dir = DEFAULT_AGENT_CARDS_DIR
        default_tool_cards_dir = DEFAULT_TOOL_CARDS_DIR

    agent_cards = _merge_card_sources(agent_cards, default_agent_cards_dir)
    card_tools = _merge_card_sources(card_tools, default_tool_cards_dir)

    # Check if we're already in an event loop
    loop = ensure_event_loop()
    if loop.is_running():
        # We're inside a running event loop, so we can't use asyncio.run
        # Instead, create a new loop
        loop = create_event_loop()
    set_asyncio_exception_handler(loop)

    exit_code: int | None = None
    try:
        loop.run_until_complete(
            _run_agent(
                name=name,
                instruction=instruction,
                config_path=config_path,
                server_list=server_list,
                agent_cards=agent_cards,
                card_tools=card_tools,
                model=model,
                message=message,
                prompt_file=prompt_file,
                resume=resume,
                url_servers=url_servers,
                stdio_servers=stdio_servers,
                agent_name=agent_name,
                skills_directory=skills_directory,
                environment_dir=environment_dir,
                shell_runtime=shell_enabled,
                mode=mode,
                transport=transport,
                host=host,
                port=port,
                tool_description=tool_description,
                instance_scope=instance_scope,
                permissions_enabled=permissions_enabled,
                reload=reload,
                watch=watch,
            )
        )
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else None
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

    if exit_code not in (None, 0):
        raise SystemExit(exit_code)


@app.callback(invoke_without_command=True, no_args_is_help=False)
def go(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent", "--name", help="Name for the agent"),
    instruction: str | None = typer.Option(
        None, "--instruction", "-i", help="Path to file or URL containing instruction for the agent"
    ),
    config_path: str | None = CommonAgentOptions.config_path(),
    servers: str | None = CommonAgentOptions.servers(),
    agent_cards: list[str] | None = CommonAgentOptions.agent_cards(),
    card_tools: list[str] | None = CommonAgentOptions.card_tools(),
    urls: str | None = CommonAgentOptions.urls(),
    auth: str | None = CommonAgentOptions.auth(),
    model: str | None = CommonAgentOptions.model(),
    message: str | None = typer.Option(
        None, "--message", "-m", help="Message to send to the agent (skips interactive mode)"
    ),
    prompt_file: str | None = typer.Option(
        None, "--prompt-file", "-p", help="Path to a prompt file to use (either text or JSON)"
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        flag_value=RESUME_LATEST_SENTINEL,
        help="Resume the last session or the specified session id",
    ),
    env_dir: Path | None = CommonAgentOptions.env_dir(),
    skills_dir: Path | None = CommonAgentOptions.skills_dir(),
    npx: str | None = CommonAgentOptions.npx(),
    uvx: str | None = CommonAgentOptions.uvx(),
    stdio: str | None = CommonAgentOptions.stdio(),
    shell: bool = CommonAgentOptions.shell(),
    reload: bool = typer.Option(False, "--reload", help="Enable manual AgentCard reloads (/reload)"),
    watch: bool = CommonAgentOptions.watch(),
) -> None:
    """
    Run an interactive agent directly from the command line.

    Examples:
        fast-agent go --model=haiku --instruction=./instruction.md --servers=fetch,filesystem
        fast-agent go --instruction=https://raw.githubusercontent.com/user/repo/prompt.md
        fast-agent go --message="What is the weather today?" --model=haiku
        fast-agent go --prompt-file=my-prompt.txt --model=haiku
        fast-agent go --agent-cards ./agents --watch
        fast-agent go --url=http://localhost:8001/mcp,http://api.example.com/sse
        fast-agent go --url=https://api.example.com/mcp --auth=YOUR_API_TOKEN
        fast-agent go --npx "@modelcontextprotocol/server-filesystem /path/to/data"
        fast-agent go --uvx "mcp-server-fetch --verbose"
        fast-agent go --stdio "python my_server.py --debug"
        fast-agent go --stdio "uv run server.py --config=settings.json"
        fast-agent go --skills /path/to/myskills -x

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
        --resume [session_id] Resume the last or specified session
        --agent-cards         Load AgentCards from a file or directory
        --card-tool           Load AgentCards and attach them as tools to the default agent
        --skills              Override the default skills folder
        --env                 Override the base fast-agent environment directory
        --shell, -x           Enable local shell runtime
        --npx                 NPX package and args to run as MCP server (quoted)
        --uvx                 UVX package and args to run as MCP server (quoted)
        --stdio               Command to run as STDIO MCP server (quoted)
        --reload              Enable manual AgentCard reloads (/reload)
        --watch               Watch AgentCard paths and reload
    """
    if os.getenv(FAST_AGENT_SHELL_CHILD_ENV):
        typer.echo(
            "fast-agent is already running inside a fast-agent shell command. "
            "Exit the shell or unset FAST_AGENT_SHELL_CHILD to continue.",
            err=True,
        )
        raise typer.Exit(1)

    # Collect all stdio commands from convenience options
    env_dir = resolve_environment_dir_option(ctx, env_dir)

    stdio_commands = collect_stdio_commands(npx, uvx, stdio)
    shell_enabled = shell

    # When shell is enabled we don't add an MCP stdio server; handled inside the agent

    # Resolve instruction from file/URL or use default
    resolved_instruction, agent_name = resolve_instruction_option(instruction)

    run_async_agent(
        name=name,
        instruction=resolved_instruction,
        config_path=config_path,
        servers=servers,
        agent_cards=agent_cards,
        card_tools=card_tools,
        urls=urls,
        auth=auth,
        model=model,
        message=message,
        prompt_file=prompt_file,
        resume=resume,
        stdio_commands=stdio_commands,
        agent_name=agent_name,
        skills_directory=skills_dir,
        environment_dir=env_dir,
        shell_enabled=shell_enabled,
        instance_scope="shared",
        reload=reload,
        watch=watch,
    )
