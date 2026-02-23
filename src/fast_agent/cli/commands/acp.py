"""Dedicated entry point for running FastAgent in ACP mode."""

from __future__ import annotations

import sys
from pathlib import Path  # noqa: TC003 - typer resolves Path annotations at runtime
from typing import TYPE_CHECKING

import typer

from fast_agent.cli.commands import serve
from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.runtime.request_builders import build_command_run_request
from fast_agent.cli.runtime.runner import run_request
from fast_agent.cli.shared_options import CommonAgentOptions

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest

app = typer.Typer(
    help="Run FastAgent as an ACP stdio server without specifying --transport=acp explicitly.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

ROOT_SUBCOMMANDS = {
    "go",
    "serve",
    "scaffold",
    "check",
    "auth",
    "bootstrap",
    "quickstart",
}


def _build_run_request(
    *,
    ctx: typer.Context,
    name: str,
    instruction: str | None,
    config_path: str | None,
    servers: str | None,
    agent_cards: list[str] | None,
    card_tools: list[str] | None,
    urls: str | None,
    auth: str | None,
    client_metadata_url: str | None,
    model: str | None,
    env_dir: Path | None,
    noenv: bool,
    force_smart: bool,
    skills_dir: Path | None,
    npx: str | None,
    uvx: str | None,
    stdio: str | None,
    description: str | None,
    host: str,
    port: int,
    shell: bool,
    instance_scope: serve.InstanceScope,
    no_permissions: bool,
    resume: str | None,
    reload: bool,
    watch: bool,
    missing_shell_cwd: serve.MissingShellCwdPolicy | None = None,
) -> AgentRunRequest:
    resolved_env_dir = resolve_environment_dir_option(ctx, env_dir, set_env_var=not noenv)
    return build_command_run_request(
        name=name,
        instruction_option=instruction,
        config_path=config_path,
        servers=servers,
        urls=urls,
        auth=auth,
        client_metadata_url=client_metadata_url,
        agent_cards=agent_cards,
        card_tools=card_tools,
        model=model,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=resume,
        npx=npx,
        uvx=uvx,
        stdio=stdio,
        target_agent_name=None,
        skills_directory=skills_dir,
        environment_dir=resolved_env_dir,
        noenv=noenv,
        force_smart=force_smart,
        shell_enabled=shell,
        mode="serve",
        transport=serve.ServeTransport.ACP.value,
        host=host,
        port=port,
        tool_description=description,
        tool_name_template=None,
        instance_scope=instance_scope.value,
        permissions_enabled=not no_permissions,
        reload=reload,
        watch=watch,
        missing_shell_cwd_policy=missing_shell_cwd.value if missing_shell_cwd else None,
    )


@app.callback(invoke_without_command=True, no_args_is_help=False)
def run_acp(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent-acp", "--name", help="Name for the ACP server"),
    instruction: str | None = CommonAgentOptions.instruction(),
    config_path: str | None = CommonAgentOptions.config_path(),
    servers: str | None = CommonAgentOptions.servers(),
    model: str | None = CommonAgentOptions.model(),
    smart: bool = CommonAgentOptions.smart(),
    agent_cards: list[str] | None = CommonAgentOptions.agent_cards(),
    card_tools: list[str] | None = CommonAgentOptions.card_tools(),
    urls: str | None = CommonAgentOptions.urls(),
    auth: str | None = CommonAgentOptions.auth(),
    client_metadata_url: str | None = CommonAgentOptions.client_metadata_url(),
    env_dir: Path | None = CommonAgentOptions.env_dir(),
    noenv: bool = CommonAgentOptions.noenv(),
    skills_dir: Path | None = CommonAgentOptions.skills_dir(),
    npx: str | None = CommonAgentOptions.npx(),
    uvx: str | None = CommonAgentOptions.uvx(),
    stdio: str | None = CommonAgentOptions.stdio(),
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
    description: str | None = typer.Option(
        None,
        "--description",
        "-d",
        help="Description used for the exposed send tool (use {agent} to reference the agent name)",
    ),
    shell: bool = CommonAgentOptions.shell(),
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
    missing_shell_cwd: serve.MissingShellCwdPolicy | None = typer.Option(
        None,
        "--missing-shell-cwd",
        help="Override shell_execution.missing_cwd_policy (ask, create, warn, error)",
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Resume the last session or the specified session id",
    ),
    reload: bool = CommonAgentOptions.reload(),
    watch: bool = CommonAgentOptions.watch(),
) -> None:
    """Run FastAgent with ACP transport defaults."""
    request = _build_run_request(
        ctx=ctx,
        name=name,
        instruction=instruction,
        config_path=config_path,
        servers=servers,
        agent_cards=agent_cards,
        card_tools=card_tools,
        urls=urls,
        auth=auth,
        client_metadata_url=client_metadata_url,
        model=model,
        env_dir=env_dir,
        noenv=noenv,
        force_smart=smart,
        skills_dir=skills_dir,
        npx=npx,
        uvx=uvx,
        stdio=stdio,
        description=description,
        host=host,
        port=port,
        shell=shell,
        instance_scope=instance_scope,
        no_permissions=no_permissions,
        resume=resume,
        reload=reload,
        watch=watch,
        missing_shell_cwd=missing_shell_cwd,
    )
    run_request(request)


def main() -> None:
    """Console script entrypoint for `fast-agent-acp`."""
    import click

    click.exceptions.UsageError.exit_code = 1

    args = sys.argv[1:]
    if args and args[0] in ROOT_SUBCOMMANDS:
        from fast_agent.cli.__main__ import main as root_cli_main

        root_cli_main()
        return
    try:
        app(standalone_mode=False)
    except click.ClickException as exc:
        try:
            import typer.rich_utils as rich_utils

            rich_utils.rich_format_error(exc)
        except Exception:
            exc.show(file=sys.stderr)
        sys.exit(getattr(exc, "exit_code", 1))
