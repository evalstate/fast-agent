"""Expose fast-agent over MCP (http/stdio) or ACP from the command line."""

from __future__ import annotations

from enum import Enum
from ipaddress import ip_address
from pathlib import Path  # noqa: TC003 - typer resolves Path annotations at runtime
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from fast_agent.cli.home_helpers import resolve_home_option
from fast_agent.cli.runtime.request_builders import build_command_run_request
from fast_agent.cli.runtime.runner import run_request
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.cli.workspace_helpers import resolve_workspace_option
from fast_agent.constants import DEFAULT_HOME_DIR

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest


class ServeTransport(str, Enum):
    HTTP = "http"
    STDIO = "stdio"
    ACP = "acp"
    A2A = "a2a"


class InstanceScope(str, Enum):
    SHARED = "shared"
    CONNECTION = "connection"
    REQUEST = "request"


class MissingShellCwdPolicy(str, Enum):
    ASK = "ask"
    CREATE = "create"
    WARN = "warn"
    ERROR = "error"


_WARNING_CONSOLE = Console(stderr=True)
DEFAULT_HTTP_HOST = "127.0.0.1"


def _validate_acp_instance_scope(instance_scope: InstanceScope) -> InstanceScope:
    if instance_scope != InstanceScope.CONNECTION:
        raise typer.BadParameter(
            "ACP is always connection-scoped. Remove --instance-scope or set it to connection.",
            param_hint="--instance-scope",
        )
    return InstanceScope.CONNECTION


def _resolve_instance_scope(
    ctx: typer.Context,
    *,
    transport: ServeTransport,
    instance_scope: InstanceScope,
) -> InstanceScope:
    """Apply transport-specific defaults without overriding explicit flags."""
    parameter_source = ctx.get_parameter_source("instance_scope")
    if transport == ServeTransport.ACP and (
        parameter_source is None or parameter_source.name == "DEFAULT"
    ):
        return InstanceScope.CONNECTION
    if transport == ServeTransport.ACP:
        return _validate_acp_instance_scope(instance_scope)
    return instance_scope


def _serves_remote_clients(transport: ServeTransport, host: str) -> bool:
    if transport not in (ServeTransport.HTTP, ServeTransport.A2A):
        return False
    normalized_host = host.strip().lower()
    if normalized_host == "localhost":
        return False
    try:
        return not ip_address(normalized_host).is_loopback
    except ValueError:
        return True


def _serve_security_warning_messages(
    *,
    transport: ServeTransport,
    host: str,
    shell: bool,
) -> list[str]:
    messages: list[str] = []
    serves_remote_clients = _serves_remote_clients(transport, host)
    if serves_remote_clients:
        messages.append(
            "[yellow]Warning:[/yellow] serving on "
            f"[bold]{host}[/bold] exposes fast-agent to remote network clients."
        )
    if shell:
        shell_message = (
            "[bold red]Warning: --shell is enabled; the shell execution tool is available"
        )
        if serves_remote_clients:
            shell_message = f"{shell_message} to remote callers"
        messages.append(f"{shell_message}.[/bold red]")
    return messages


def _emit_serve_security_warnings(
    *,
    transport: ServeTransport,
    host: str,
    shell: bool,
) -> None:
    for message in _serve_security_warning_messages(transport=transport, host=host, shell=shell):
        _WARNING_CONSOLE.print(message)


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
    home: Path | None,
    no_home: bool,
    force_smart: bool,
    npx: str | None,
    uvx: str | None,
    stdio: str | None,
    transport: ServeTransport,
    host: str,
    port: int,
    shell: bool,
    instance_scope: InstanceScope,
    no_permissions: bool,
    reload: bool,
    watch: bool,
    prefer_local_shell: bool = False,
    missing_shell_cwd: MissingShellCwdPolicy | None = None,
    no_shell: bool = False,
    workspace: Path | None = None,
) -> AgentRunRequest:
    if watch and transport in (ServeTransport.HTTP, ServeTransport.STDIO):
        raise typer.BadParameter(
            "--watch is not supported for MCP serving; restart the server after card changes.",
            param_hint="--watch",
        )
    resolved_workspace = resolve_workspace_option(ctx, workspace)
    home_option = home
    if home_option is None and resolved_workspace is not None and not no_home:
        home_option = resolved_workspace / DEFAULT_HOME_DIR
    resolved_home = resolve_home_option(ctx, home_option, set_env_var=not no_home)
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
        home=resolved_home,
        no_home=no_home,
        force_smart=force_smart,
        shell_enabled=shell,
        no_shell=no_shell,
        prefer_local_shell=prefer_local_shell,
        mode="serve",
        transport=transport.value,
        host=host,
        port=port,
        instance_scope=instance_scope.value,
        permissions_enabled=not no_permissions,
        reload=reload,
        watch=watch,
        missing_shell_cwd_policy=missing_shell_cwd.value if missing_shell_cwd else None,
    )


app = typer.Typer(
    help=(
        "Expose fast-agent to clients over MCP (http or stdio), ACP, or A2A, "
        "without writing an agent.py file"
    ),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_completion=False,
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
    workspace: Path | None = CommonAgentOptions.workspace(),
    home: Path | None = CommonAgentOptions.home(),
    no_home: bool = CommonAgentOptions.no_home(),
    smart: bool = CommonAgentOptions.smart(),
    skills_dir: Path | None = CommonAgentOptions.skills_dir(),
    npx: str | None = CommonAgentOptions.npx(),
    uvx: str | None = CommonAgentOptions.uvx(),
    stdio: str | None = CommonAgentOptions.stdio(),
    transport: ServeTransport = typer.Option(
        ServeTransport.HTTP,
        "--transport",
        help="Transport protocol to expose (http, stdio, acp, a2a)",
    ),
    host: str = typer.Option(
        DEFAULT_HTTP_HOST,
        "--host",
        help="Host address to bind when using HTTP transport",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to use when running as a server with HTTP transport",
    ),
    shell: bool = CommonAgentOptions.shell(),
    no_shell: bool = CommonAgentOptions.no_shell(),
    prefer_local_shell: bool = typer.Option(
        False,
        "--prefer-local-shell",
        help=(
            "When serving ACP with shell mode, use fast-agent's local shell runtime "
            "instead of the ACP client's terminal capability"
        ),
    ),
    instance_scope: InstanceScope = typer.Option(
        InstanceScope.SHARED,
        "--instance-scope",
        help="Control how clients receive isolated agent instances. ACP is always connection-scoped.",
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
    """Expose fast-agent to clients over MCP (http/stdio), ACP, or A2A."""
    if ctx.invoked_subcommand is not None:
        return
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
        workspace=workspace,
        home=home,
        no_home=no_home,
        force_smart=smart,
        npx=npx,
        uvx=uvx,
        stdio=stdio,
        transport=transport,
        host=host,
        port=port,
        shell=shell,
        no_shell=no_shell,
        prefer_local_shell=prefer_local_shell,
        instance_scope=_resolve_instance_scope(
            ctx,
            transport=transport,
            instance_scope=instance_scope,
        ),
        no_permissions=no_permissions,
        reload=reload,
        watch=watch,
        missing_shell_cwd=missing_shell_cwd,
    )
    _emit_serve_security_warnings(transport=transport, host=host, shell=shell)
    run_request(request)


@app.command("a2a")
def serve_a2a(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent-a2a", "--name", help="Name for the A2A server"),
    instruction: str | None = CommonAgentOptions.instruction(),
    config_path: str | None = CommonAgentOptions.config_path(),
    model: str | None = CommonAgentOptions.model(),
    servers: str | None = CommonAgentOptions.servers(),
    agent_cards: list[str] | None = CommonAgentOptions.agent_cards(),
    card_tools: list[str] | None = CommonAgentOptions.card_tools(),
    urls: str | None = CommonAgentOptions.urls(),
    auth: str | None = CommonAgentOptions.auth(),
    client_metadata_url: str | None = CommonAgentOptions.client_metadata_url(),
    workspace: Path | None = CommonAgentOptions.workspace(),
    home: Path | None = CommonAgentOptions.home(),
    no_home: bool = CommonAgentOptions.no_home(),
    smart: bool = CommonAgentOptions.smart(),
    skills_dir: Path | None = CommonAgentOptions.skills_dir(),
    npx: str | None = CommonAgentOptions.npx(),
    uvx: str | None = CommonAgentOptions.uvx(),
    host: str = typer.Option(
        DEFAULT_HTTP_HOST,
        "--host",
        help="Host address to bind for the A2A HTTP server",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to use for the A2A HTTP server",
    ),
    shell: bool = CommonAgentOptions.shell(),
    no_shell: bool = CommonAgentOptions.no_shell(),
    instance_scope: InstanceScope = typer.Option(
        InstanceScope.SHARED,
        "--instance-scope",
        help="Control how A2A clients receive isolated agent instances.",
    ),
    reload: bool = CommonAgentOptions.reload(),
    watch: bool = CommonAgentOptions.watch(),
) -> None:
    """Expose fast-agent over A2A HTTP transports."""
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
        workspace=workspace,
        home=home,
        no_home=no_home,
        force_smart=smart,
        npx=npx,
        uvx=uvx,
        stdio=None,
        transport=ServeTransport.A2A,
        host=host,
        port=port,
        shell=shell,
        no_shell=no_shell,
        instance_scope=instance_scope,
        no_permissions=False,
        reload=reload,
        watch=watch,
    )
    _emit_serve_security_warnings(transport=ServeTransport.A2A, host=host, shell=shell)
    run_request(request)
