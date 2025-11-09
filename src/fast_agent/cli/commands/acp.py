"""Dedicated entry point for running FastAgent in ACP mode."""

import sys
from pathlib import Path

import typer

from fast_agent.cli.commands import serve
from fast_agent.cli.commands.go import (
    collect_stdio_commands,
    resolve_instruction_option,
    run_async_agent,
)

app = typer.Typer(
    help="Run FastAgent as an ACP stdio server without specifying --transport=acp explicitly.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

ROOT_SUBCOMMANDS = {
    "go",
    "serve",
    "setup",
    "check",
    "auth",
    "bootstrap",
    "quickstart",
}


@app.callback(invoke_without_command=True, no_args_is_help=False)
def run_acp(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent-acp", "--name", help="Name for the ACP server"),
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
    instance_scope: serve.InstanceScope = typer.Option(
        serve.InstanceScope.SHARED,
        "--instance-scope",
        help="Control how ACP clients receive isolated agent instances (shared, connection, request)",
    ),
) -> None:
    """
    Run FastAgent with ACP transport defaults.

    This mirrors `fast-agent serve --transport acp` but provides a shorter command and
    a distinct default name so ACP-specific tooling can integrate more easily.
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
        transport=serve.ServeTransport.ACP.value,
        host=host,
        port=port,
        tool_description=description,
        instance_scope=instance_scope.value,
    )


def main() -> None:
    """Console script entrypoint for `fast-agent-acp`."""
    args = sys.argv[1:]
    if args and args[0] in ROOT_SUBCOMMANDS:
        from fast_agent.cli.__main__ import main as root_cli_main

        root_cli_main()
        return
    app()
