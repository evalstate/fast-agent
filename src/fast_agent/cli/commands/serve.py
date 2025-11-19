"""Run FastAgent as an MCP server from the command line."""

from enum import Enum
from pathlib import Path

import typer

from fast_agent.cli.commands.go import (
    collect_stdio_commands,
    resolve_instruction_option,
    run_async_agent,
)


class ServeTransport(str, Enum):
    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"
    ACP = "acp"


class InstanceScope(str, Enum):
    SHARED = "shared"
    CONNECTION = "connection"
    REQUEST = "request"


app = typer.Typer(
    help="Run FastAgent as an MCP server without writing an agent.py file",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@app.callback(invoke_without_command=True, no_args_is_help=False)
def serve(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent", "--name", help="Name for the MCP server"),
    instruction: str | None = typer.Option(
        None, "--instruction", "-i", help="Path to file or URL containing instruction for the agent"
    ),
    config_path: str | None = typer.Option(None, "--config-path", "-c", help="Path to config file"),
    servers: str | None = typer.Option(
        None, "--servers", help="Comma-separated list of server names to enable from config"
    ),
    urls: str | None = typer.Option(
        None, "--url", help="Comma-separated list of HTTP/SSE URLs to connect to"
    ),
    auth: str | None = typer.Option(
        None, "--auth", help="Bearer token for authorization with URL-based servers"
    ),
    model: str | None = typer.Option(
        None, "--model", "--models", help="Override the default model (e.g., haiku, sonnet, gpt-4)"
    ),
    skills_dir: Path | None = typer.Option(
        None,
        "--skills-dir",
        "--skills",
        help="Override the default skills directory",
    ),
    npx: str | None = typer.Option(
        None, "--npx", help="NPX package and args to run as MCP server (quoted)"
    ),
    uvx: str | None = typer.Option(
        None, "--uvx", help="UVX package and args to run as MCP server (quoted)"
    ),
    stdio: str | None = typer.Option(
        None, "--stdio", help="Command to run as STDIO MCP server (quoted)"
    ),
    description: str | None = typer.Option(
        None,
        "--description",
        "-d",
        help="Description used for the exposed send tool (use {agent} to reference the agent name)",
    ),
    transport: ServeTransport = typer.Option(
        ServeTransport.HTTP,
        "--transport",
        help="Transport protocol to expose (http, sse, stdio, acp)",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host address to bind when using HTTP or SSE transport",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to use when running as a server with HTTP or SSE transport",
    ),
    shell: bool = typer.Option(
        False,
        "--shell",
        "-x",
        help="Enable a local shell runtime and expose the execute tool (bash or pwsh).",
    ),
    instance_scope: InstanceScope = typer.Option(
        InstanceScope.SHARED,
        "--instance-scope",
        help="Control how MCP clients receive isolated agent instances (shared, connection, request)",
    ),
) -> None:
    """
    Run FastAgent as an MCP server.

    Examples:
        fast-agent serve --model=haiku --instruction=./instruction.md --transport=http --port=8000
        fast-agent serve --url=http://localhost:8001/mcp --auth=YOUR_API_TOKEN
        fast-agent serve --stdio "python my_server.py --debug"
        fast-agent serve --npx "@modelcontextprotocol/server-filesystem /path/to/data"
        fast-agent serve --description "Interact with the {agent} assistant"
    """
    stdio_commands = collect_stdio_commands(npx, uvx, stdio)
    shell_enabled = shell

    resolved_instruction, agent_name = resolve_instruction_option(instruction)

    run_async_agent(
        name=name,
        instruction=resolved_instruction,
        config_path=config_path,
        servers=servers,
        urls=urls,
        auth=auth,
        model=model,
        message=None,
        prompt_file=None,
        stdio_commands=stdio_commands,
        agent_name=agent_name,
        skills_directory=skills_dir,
        shell_enabled=shell_enabled,
        mode="serve",
        transport=transport.value,
        host=host,
        port=port,
        tool_description=description,
        instance_scope=instance_scope.value,
    )
