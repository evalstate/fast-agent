"""Run FastAgent as an MCP server from the command line."""

from __future__ import annotations

from enum import Enum
from pathlib import Path  # noqa: TC003 - typer resolves Path annotations at runtime
from typing import TYPE_CHECKING

import typer

from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.runtime.request_builders import build_command_run_request
from fast_agent.cli.runtime.runner import run_request
from fast_agent.cli.shared_options import CommonAgentOptions

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest


class ServeTransport(str, Enum):
    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"
    ACP = "acp"


class InstanceScope(str, Enum):
    SHARED = "shared"
    CONNECTION = "connection"
    REQUEST = "request"


class MissingShellCwdPolicy(str, Enum):
    ASK = "ask"
    CREATE = "create"
    WARN = "warn"
    ERROR = "error"


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
    skills_dir: Path | None,
    env_dir: Path | None,
    noenv: bool,
    force_smart: bool,
    npx: str | None,
    uvx: str | None,
    stdio: str | None,
    description: str | None,
    tool_name_template: str | None,
    transport: ServeTransport,
    host: str,
    port: int,
    shell: bool,
    instance_scope: InstanceScope,
    no_permissions: bool,
    reload: bool,
    watch: bool,
    missing_shell_cwd: MissingShellCwdPolicy | None = None,
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
        resume=None,
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
        transport=transport.value,
        host=host,
        port=port,
        tool_description=description,
        tool_name_template=tool_name_template,
        instance_scope=instance_scope.value,
        permissions_enabled=not no_permissions,
        reload=reload,
        watch=watch,
        missing_shell_cwd_policy=missing_shell_cwd.value if missing_shell_cwd else None,
    )


app = typer.Typer(
    help="Run FastAgent as an MCP server without writing an agent.py file",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@app.callback(invoke_without_command=True, no_args_is_help=False)
def serve(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent", "--name", help="Name for the MCP server"),
    instruction: str | None = CommonAgentOptions.instruction(),
    config_path: str | None = CommonAgentOptions.config_path(),
    model: str | None = CommonAgentOptions.model(),
    servers: str | None = CommonAgentOptions.servers(),
    agent_cards: list[str] | None = CommonAgentOptions.agent_cards(),
    card_tools: list[str] | None = CommonAgentOptions.card_tools(),
    urls: str | None = CommonAgentOptions.urls(),
    auth: str | None = CommonAgentOptions.auth(),
    client_metadata_url: str | None = CommonAgentOptions.client_metadata_url(),
    env_dir: Path | None = CommonAgentOptions.env_dir(),
    noenv: bool = CommonAgentOptions.noenv(),
    smart: bool = CommonAgentOptions.smart(),
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
    tool_name_template: str | None = typer.Option(
        None,
        "--tool-name-template",
        help="Template for exposed agent tool names (use {agent} to reference the agent name)",
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
    shell: bool = CommonAgentOptions.shell(),
    instance_scope: InstanceScope = typer.Option(
        InstanceScope.SHARED,
        "--instance-scope",
        help="Control how MCP clients receive isolated agent instances (shared, connection, request)",
    ),
    no_permissions: bool = typer.Option(
        False,
        "--no-permissions",
        help="Disable tool permission requests (allow all tool executions without asking) - ACP only",
    ),
    missing_shell_cwd: MissingShellCwdPolicy | None = typer.Option(
        None,
        "--missing-shell-cwd",
        help="Override shell_execution.missing_cwd_policy (ask, create, warn, error)",
    ),
    reload: bool = CommonAgentOptions.reload(),
    watch: bool = CommonAgentOptions.watch(),
) -> None:
    """Run FastAgent as an MCP server."""
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
        skills_dir=skills_dir,
        env_dir=env_dir,
        noenv=noenv,
        force_smart=smart,
        npx=npx,
        uvx=uvx,
        stdio=stdio,
        description=description,
        tool_name_template=tool_name_template,
        transport=transport,
        host=host,
        port=port,
        shell=shell,
        instance_scope=instance_scope,
        no_permissions=no_permissions,
        reload=reload,
        watch=watch,
        missing_shell_cwd=missing_shell_cwd,
    )
    run_request(request)
