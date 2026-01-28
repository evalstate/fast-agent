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
from fast_agent.cli.constants import RESUME_LATEST_SENTINEL
from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.shared_options import CommonAgentOptions

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
    config_path: str | None = CommonAgentOptions.config_path(),
    servers: str | None = CommonAgentOptions.servers(),
    agent_cards: list[str] | None = CommonAgentOptions.agent_cards(),
    card_tools: list[str] | None = CommonAgentOptions.card_tools(),
    urls: str | None = CommonAgentOptions.urls(),
    auth: str | None = CommonAgentOptions.auth(),
    model: str | None = CommonAgentOptions.model(),
    env_dir: Path | None = CommonAgentOptions.env_dir(),
    skills_dir: Path | None = CommonAgentOptions.skills_dir(),
    npx: str | None = CommonAgentOptions.npx(),
    uvx: str | None = CommonAgentOptions.uvx(),
    stdio: str | None = CommonAgentOptions.stdio(),
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
    no_permissions: bool = typer.Option(
        False,
        "--no-permissions",
        help="Disable tool permission requests (allow all tool executions without asking)",
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        flag_value=RESUME_LATEST_SENTINEL,
        help="Resume the last session or the specified session id",
    ),
    reload: bool = CommonAgentOptions.reload(),
    watch: bool = CommonAgentOptions.watch(),
) -> None:
    """
    Run FastAgent with ACP transport defaults.

    This mirrors `fast-agent serve --transport acp` but provides a shorter command and
    a distinct default name so ACP-specific tooling can integrate more easily.
    """
    env_dir = resolve_environment_dir_option(ctx, env_dir)

    stdio_commands = collect_stdio_commands(npx, uvx, stdio)
    shell_enabled = shell

    resolved_instruction, agent_name = resolve_instruction_option(instruction, model)

    run_async_agent(
        name=name,
        instruction=resolved_instruction,
        config_path=config_path,
        servers=servers,
        resume=resume,
        agent_cards=agent_cards,
        card_tools=card_tools,
        urls=urls,
        auth=auth,
        model=model,
        message=None,
        prompt_file=None,
        stdio_commands=stdio_commands,
        agent_name=agent_name,
        skills_directory=skills_dir,
        environment_dir=env_dir,
        shell_enabled=shell_enabled,
        mode="serve",
        transport=serve.ServeTransport.ACP.value,
        host=host,
        port=port,
        tool_description=description,
        instance_scope=instance_scope.value,
        permissions_enabled=not no_permissions,
        reload=reload,
        watch=watch,
    )


def main() -> None:
    """Console script entrypoint for `fast-agent-acp`."""
    # Override Click's UsageError exit code from 2 to 1 for consistency
    import click

    click.exceptions.UsageError.exit_code = 1

    args = sys.argv[1:]
    if args and args[0] in ROOT_SUBCOMMANDS:
        from fast_agent.cli.__main__ import main as root_cli_main

        root_cli_main()
        return
    try:
        # Run the Typer app without triggering automatic sys.exit so we can
        # guarantee error output goes to stderr with a non-zero exit code.
        app(standalone_mode=False)
    except click.ClickException as exc:
        # Preserve Typer's rich formatting when available, otherwise fall back to plain text.
        try:
            import typer.rich_utils as rich_utils

            rich_utils.rich_format_error(exc)
        except Exception:
            exc.show(file=sys.stderr)
        sys.exit(getattr(exc, "exit_code", 1))
