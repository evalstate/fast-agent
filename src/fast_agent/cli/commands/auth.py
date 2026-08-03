"""Authentication and credential management commands."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlparse

import typer
from rich.table import Table

from fast_agent.cli.command_support import get_settings_or_exit
from fast_agent.cli.display import print_detail_line
from fast_agent.core.keyring_utils import get_keyring_status
from fast_agent.mcp.client_gateway import (
    resolve_effective_mcp_auth_mode,
    resolve_oauth_mode,
)
from fast_agent.mcp.connect_targets import redact_mcp_url
from fast_agent.mcp.oauth_client import (
    _derive_base_server_url,
    add_identity_to_index,
    clear_keyring_token,
    compute_server_identity,
    compute_server_identity_candidates,
    keyring_credential_present,
    keyring_token_present,
    list_keyring_credentials,
    oauth_resource_key_candidates,
)
from fast_agent.ui.console import console
from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.async_utils import run_sync
from fast_agent.utils.text import strip_to_none
from fast_agent.utils.transports import uses_mcp_remote_transport

if TYPE_CHECKING:
    from fast_agent.auth.providers import ProviderAuthStatus
    from fast_agent.config import MCPServerSettings, Settings
    from fast_agent.core.keyring_utils import KeyringStatus
    from fast_agent.mcp.client_gateway import EffectiveMCPAuthMode

app = typer.Typer(
    help="Inspect and manage provider and MCP credentials.",
    add_completion=False,
)
provider_app = typer.Typer(
    help="Manage model-provider credentials.",
    add_completion=False,
)
mcp_app = typer.Typer(
    help="Manage MCP OAuth credentials.",
    add_completion=False,
)
app.add_typer(provider_app, name="provider")
app.add_typer(mcp_app, name="mcp")


type CredentialState = Literal[
    "ready",
    "missing",
    "memory",
    "unavailable",
    "not_applicable",
]


@dataclass(frozen=True, slots=True)
class ProviderAuthView:
    id: str
    name: str
    state: Literal["ready", "expired", "not_configured"]
    source: str | None
    expires_at: str | None


@dataclass(frozen=True, slots=True)
class McpServerAuthView:
    name: str
    management: str
    transport: str
    endpoint: str | None
    auth_mode: EffectiveMCPAuthMode
    persistence: str | None
    credential: CredentialState
    resource: str | None
    shared_with: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class McpCredentialView:
    resource: str
    configured_servers: tuple[str, ...]
    orphaned: bool


@dataclass(frozen=True, slots=True)
class LoginTargetConfig:
    server: MCPServerSettings
    transport: Literal["http", "sse"]
    configured_name: str | None = None


def _emit_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2))


def _confirm_destructive_action(*, prompt: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        typer.echo("Non-interactive credential removal requires --yes.", err=True)
        raise typer.Exit(2)
    return typer.confirm(prompt, default=False)


def _keyring_payload(status: KeyringStatus) -> dict[str, str | bool]:
    return {
        "name": status.name,
        "available": status.available,
        "writable": status.writable,
    }


def _configured_mcp_servers(settings: Settings) -> dict[str, MCPServerSettings]:
    if settings.mcp is None:
        return {}
    return settings.mcp.servers


def _configured_server(
    settings: Settings,
    name: str,
    *,
    url_guidance: str,
) -> MCPServerSettings:
    if "://" in name:
        typer.echo(url_guidance, err=True)
        raise typer.Exit(2)
    server = _configured_mcp_servers(settings).get(name)
    if server is None:
        typer.echo(f"Configured MCP server '{name}' was not found.", err=True)
        raise typer.Exit(1)
    return server


def _resource_for_server(server: MCPServerSettings) -> str | None:
    if not uses_mcp_remote_transport(server.transport) or not server.url:
        return None
    return compute_server_identity(server)


def _credential_consumers_by_resource(settings: Settings) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for name, server in _configured_mcp_servers(settings).items():
        resource = _resource_for_server(server)
        auth_mode = resolve_effective_mcp_auth_mode(server)
        persistence = server.auth.persist if server.auth is not None else "keyring"
        if resource is None or auth_mode not in {"auto", "oauth"} or persistence != "keyring":
            continue
        grouped.setdefault(resource, []).append(name)
    return {resource: tuple(sorted(names)) for resource, names in grouped.items()}


def _backfill_configured_credential_index(settings: Settings) -> None:
    """Index legacy client-registration-only records for configured resources."""
    resources: set[str] = set()
    for server in _configured_mcp_servers(settings).values():
        if _resource_for_server(server) is None:
            continue
        resources.update(compute_server_identity_candidates(server))
    for resource in resources:
        if keyring_credential_present(resource):
            add_identity_to_index("fast-agent-mcp", resource)


def _provider_view(status: ProviderAuthStatus) -> ProviderAuthView:
    state: Literal["ready", "expired", "not_configured"]
    if status.expired:
        state = "expired"
    elif status.present:
        state = "ready"
    else:
        state = "not_configured"
    expires_at = (
        datetime.fromtimestamp(status.expires_at).astimezone().isoformat(timespec="minutes")
        if status.expires_at is not None
        else None
    )
    return ProviderAuthView(
        id=status.provider,
        name=status.display_name,
        state=state,
        source=status.source,
        expires_at=expires_at,
    )


def _provider_views() -> list[ProviderAuthView]:
    from fast_agent.auth.providers import provider_ids
    from fast_agent.auth.providers import provider_status as get_status

    return [_provider_view(get_status(provider)) for provider in provider_ids()]


def _provider_view_for(provider: str) -> ProviderAuthView:
    from fast_agent.auth.providers import provider_status as get_status
    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error

    try:
        return _provider_view(get_status(provider))
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc), err=True)
        raise typer.Exit(1) from exc


def _print_provider_views(views: list[ProviderAuthView]) -> None:
    table = Table(show_header=True, box=None)
    table.add_column("Provider", header_style="bold")
    table.add_column("Status", header_style="bold")
    table.add_column("Source", header_style="bold")
    table.add_column("Expires", header_style="bold")
    styles = {
        "ready": "green",
        "expired": "red",
        "not_configured": "dim",
    }
    labels = {
        "ready": "ready",
        "expired": "expired",
        "not_configured": "not configured",
    }
    for view in views:
        table.add_row(
            view.name,
            f"[{styles[view.state]}]{labels[view.state]}[/{styles[view.state]}]",
            view.source or "-",
            view.expires_at or "-",
        )
    console.print(table)


def _credential_state(
    server: MCPServerSettings,
    *,
    auth_mode: EffectiveMCPAuthMode,
    resource: str | None,
    keyring_status: KeyringStatus,
) -> tuple[str | None, CredentialState]:
    if auth_mode not in {"auto", "oauth"} or resource is None:
        return None, "not_applicable"

    persistence = server.auth.persist if server.auth is not None else "keyring"
    if persistence == "memory":
        return persistence, "memory"
    if not keyring_status.available:
        return persistence, "unavailable"
    return (
        persistence,
        (
            "ready"
            if any(
                keyring_token_present(candidate)
                for candidate in compute_server_identity_candidates(server)
            )
            else "missing"
        ),
    )


def _mcp_server_view(
    name: str,
    server: MCPServerSettings,
    *,
    keyring_status: KeyringStatus,
    servers_by_resource: dict[str, tuple[str, ...]],
) -> McpServerAuthView:
    auth_mode = resolve_effective_mcp_auth_mode(server)
    resource = _resource_for_server(server)
    persistence, credential = _credential_state(
        server,
        auth_mode=auth_mode,
        resource=resource,
        keyring_status=keyring_status,
    )
    aliases = servers_by_resource.get(resource, ()) if resource is not None else ()
    if name not in aliases:
        aliases = ()
    return McpServerAuthView(
        name=name,
        management=server.management,
        transport=server.transport,
        endpoint=redact_mcp_url(server.url) if server.url else None,
        auth_mode=auth_mode,
        persistence=persistence,
        credential=credential,
        resource=redact_mcp_url(resource) if resource else None,
        shared_with=tuple(alias for alias in aliases if alias != name),
    )


def _mcp_server_views(
    settings: Settings,
    *,
    keyring_status: KeyringStatus,
) -> list[McpServerAuthView]:
    _backfill_configured_credential_index(settings)
    by_resource = _credential_consumers_by_resource(settings)
    return [
        _mcp_server_view(
            name,
            server,
            keyring_status=keyring_status,
            servers_by_resource=by_resource,
        )
        for name, server in sorted(_configured_mcp_servers(settings).items())
    ]


def _auth_mode_label(view: McpServerAuthView) -> str:
    if view.auth_mode == "forwarded":
        return "forwarded:huggingface"
    return view.auth_mode.replace("_", "-")


def _credential_label(state: CredentialState) -> str:
    return {
        "ready": "[green]ready[/green]",
        "missing": "[dim]missing[/dim]",
        "memory": "[yellow]memory[/yellow]",
        "unavailable": "[red]unavailable[/red]",
        "not_applicable": "[dim]-[/dim]",
    }[state]


def _print_mcp_server_views(
    views: list[McpServerAuthView],
    *,
    keyring_status: KeyringStatus,
) -> None:
    print_detail_line(
        console,
        "keyring backend",
        keyring_status.name if keyring_status.available else "not available",
        value_style="green" if keyring_status.available else "red",
    )
    table = Table(show_header=True, box=None)
    table.add_column("Server", header_style="bold")
    table.add_column("Transport", header_style="bold")
    table.add_column("Auth", header_style="bold")
    table.add_column("Credential", header_style="bold")
    table.add_column("Endpoint", header_style="bold")
    if not views:
        table.add_row("[dim]None[/dim]", "-", "-", "-", "-")
    for view in views:
        table.add_row(
            view.name,
            view.transport.upper(),
            _auth_mode_label(view),
            _credential_label(view.credential),
            view.endpoint or "-",
        )
    console.print(table)


def _credential_views(settings: Settings) -> list[McpCredentialView]:
    _backfill_configured_credential_index(settings)
    by_resource = _credential_consumers_by_resource(settings)
    return [
        McpCredentialView(
            resource=redact_mcp_url(resource),
            configured_servers=by_resource.get(_derive_base_server_url(resource) or resource, ()),
            orphaned=(_derive_base_server_url(resource) or resource) not in by_resource,
        )
        for resource in list_keyring_credentials()
    ]


def _print_credential_views(
    views: list[McpCredentialView],
    *,
    keyring_status: KeyringStatus,
) -> None:
    print_detail_line(
        console,
        "keyring backend",
        keyring_status.name if keyring_status.available else "not available",
        value_style="green" if keyring_status.available else "red",
    )
    table = Table(show_header=True, box=None)
    table.add_column("OAuth Resource", header_style="bold")
    table.add_column("Configured Servers", header_style="bold")
    table.add_column("Orphaned", header_style="bold")
    if not views:
        table.add_row("[dim]None[/dim]", "-", "-")
    for view in views:
        table.add_row(
            view.resource,
            ", ".join(view.configured_servers) or "[dim]None[/dim]",
            "[yellow]yes[/yellow]" if view.orphaned else "no",
        )
    console.print(table)


def _mcp_payload(
    settings: Settings,
    *,
    keyring_status: KeyringStatus,
) -> dict[str, object]:
    return {
        "keyring": _keyring_payload(keyring_status),
        "servers": [
            asdict(view) for view in _mcp_server_views(settings, keyring_status=keyring_status)
        ],
        "credentials": [asdict(view) for view in _credential_views(settings)],
    }


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
) -> None:
    """Show a combined provider and MCP credential overview."""
    if ctx.invoked_subcommand is not None:
        if json_output or config_path is not None:
            typer.echo(
                "`auth --json` and `auth --config-path` apply only to the combined overview. "
                "Place the option after the selected subcommand instead.",
                err=True,
            )
            raise typer.Exit(2)
        return
    settings = get_settings_or_exit(config_path)
    keyring_status = get_keyring_status()
    providers = _provider_views()
    if json_output:
        _emit_json(
            {
                "providers": [asdict(view) for view in providers],
                "mcp": _mcp_payload(settings, keyring_status=keyring_status),
            }
        )
        return
    console.print("[bold]Provider credentials[/bold]")
    _print_provider_views(providers)
    console.print("\n[bold]MCP authentication[/bold]")
    _print_mcp_server_views(
        _mcp_server_views(settings, keyring_status=keyring_status),
        keyring_status=keyring_status,
    )


@provider_app.command("list")
def provider_list(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List model-provider credential status."""
    views = _provider_views()
    if json_output:
        _emit_json({"providers": [asdict(view) for view in views]})
    else:
        _print_provider_views(views)


@provider_app.command("show")
def provider_show(
    provider: str = typer.Argument(..., help="Provider name: xai or codex"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Show one model-provider credential."""
    view = _provider_view_for(provider)
    if json_output:
        _emit_json({"provider": asdict(view)})
        return
    _print_provider_views([view])


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="Provider name: xai or codex"),
) -> None:
    """Authenticate with a model provider."""
    from fast_agent.auth.providers import get_oauth_provider
    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error

    try:
        handler = get_oauth_provider(provider)
        handler.login()
        typer.echo(f"{handler.display_name} OAuth login complete.")
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc), err=True)
        raise typer.Exit(1) from exc


@provider_app.command("logout")
def provider_logout(
    provider: str = typer.Argument(..., help="Provider name: xai or codex"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt for confirmation"),
) -> None:
    """Remove a stored provider credential."""
    from fast_agent.auth.providers import get_oauth_provider
    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error

    try:
        handler = get_oauth_provider(provider)
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc), err=True)
        raise typer.Exit(1) from exc
    if not _confirm_destructive_action(
        prompt=f"Remove the stored {handler.display_name} OAuth credential?",
        yes=yes,
    ):
        typer.echo("Cancelled; no provider credential was removed.")
        return
    typer.echo(
        f"{handler.display_name} OAuth credential removed."
        if handler.logout()
        else f"No {handler.display_name} OAuth credential found."
    )


@provider_app.command("token")
def provider_token(
    provider: str = typer.Argument(..., help="Provider name: xai or codex"),
) -> None:
    """Print a current provider access token."""
    from fast_agent.auth.providers import get_oauth_provider
    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error

    try:
        handler = get_oauth_provider(provider)
        token = handler.access_token()
        if token is None:
            raise ProviderKeyError(
                f"{handler.display_name} OAuth token not configured",
                f"Run `fast-agent auth provider login {handler.id}` first.",
            )
        typer.echo(token)
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc), err=True)
        raise typer.Exit(1) from exc


@provider_app.command("export")
def provider_export(
    provider: str = typer.Argument(..., help="Provider name: xai or codex"),
    output: str = typer.Argument(..., help="Destination provider auth JSON file"),
    force: bool = typer.Option(False, "--force", help="Replace an existing file"),
) -> None:
    """Export one refreshable provider credential."""
    from fast_agent.auth.providers import export_provider_credential
    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error

    path = Path(output).expanduser()
    if path.exists() and not force:
        typer.echo(f"Refusing to replace existing file: {path}", err=True)
        raise typer.Exit(1)
    try:
        export_provider_credential(provider, path)
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Exported {provider.strip().casefold()} credential to {path}")


@mcp_app.command("list")
def mcp_list(
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List configured MCP servers and their effective authentication."""
    settings = get_settings_or_exit(config_path)
    keyring_status = get_keyring_status()
    views = _mcp_server_views(settings, keyring_status=keyring_status)
    if json_output:
        _emit_json(
            {
                "keyring": _keyring_payload(keyring_status),
                "servers": [asdict(view) for view in views],
            }
        )
    else:
        _print_mcp_server_views(views, keyring_status=keyring_status)


@mcp_app.command("show")
def mcp_show(
    server: str = typer.Argument(..., help="Configured MCP server name"),
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Show the authentication configuration for one MCP server."""
    settings = get_settings_or_exit(config_path)
    _backfill_configured_credential_index(settings)
    server_config = _configured_server(
        settings,
        server,
        url_guidance=(
            "`auth mcp show` accepts a configured server name. "
            "Use `fast-agent auth mcp credentials` to inspect stored OAuth resources."
        ),
    )
    keyring_status = get_keyring_status()
    view = _mcp_server_view(
        server,
        server_config,
        keyring_status=keyring_status,
        servers_by_resource=_credential_consumers_by_resource(settings),
    )
    if json_output:
        _emit_json({"server": asdict(view)})
        return

    print_detail_line(console, "server", view.name)
    print_detail_line(console, "endpoint", view.endpoint or "-")
    print_detail_line(console, "transport", view.transport)
    print_detail_line(console, "management", view.management)
    print_detail_line(console, "authentication", _auth_mode_label(view))
    print_detail_line(console, "OAuth resource", view.resource or "-")
    print_detail_line(console, "credential", view.credential.replace("_", "-"))
    print_detail_line(console, "persistence", view.persistence or "-")
    print_detail_line(console, "shared with", ", ".join(view.shared_with) or "-")


@mcp_app.command("credentials")
def mcp_credentials(
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List indexed local MCP OAuth credentials."""
    settings = get_settings_or_exit(config_path)
    keyring_status = get_keyring_status()
    views = _credential_views(settings)
    if json_output:
        _emit_json(
            {
                "keyring": _keyring_payload(keyring_status),
                "credentials": [asdict(view) for view in views],
            }
        )
    else:
        _print_credential_views(views, keyring_status=keyring_status)


def _validated_endpoint(endpoint: str, transport: str | None) -> LoginTargetConfig:
    from fast_agent.cli.commands.url_parser import generate_server_name
    from fast_agent.config import MCPServerAuthSettings, MCPServerSettings

    try:
        parsed = urlparse(endpoint)
        parsed_port = parsed.port
    except ValueError as exc:
        typer.echo(f"Invalid --endpoint URL: {exc}", err=True)
        raise typer.Exit(2) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        typer.echo("--endpoint must be an absolute HTTP(S) URL.", err=True)
        raise typer.Exit(2)
    if parsed.username is not None or parsed.password is not None:
        typer.echo("--endpoint must not contain URL user information.", err=True)
        raise typer.Exit(2)
    del parsed_port

    normalized_transport = normalize_action_token(transport or "")
    if not normalized_transport:
        normalized_transport = "sse" if parsed.path.rstrip("/").endswith("/sse") else "http"
    if not uses_mcp_remote_transport(normalized_transport):
        typer.echo("--transport must be 'http' or 'sse'.", err=True)
        raise typer.Exit(2)
    resolved_transport = cast("Literal['http', 'sse']", normalized_transport)
    return LoginTargetConfig(
        server=MCPServerSettings.model_construct(
            name=generate_server_name(endpoint),
            transport=resolved_transport,
            url=endpoint,
            auth=MCPServerAuthSettings(),
        ),
        transport=resolved_transport,
        configured_name=None,
    )


def _resolve_login_target(
    server: str | None,
    *,
    endpoint: str | None,
    transport: str | None,
    config_path: str | None,
) -> LoginTargetConfig:
    server_name = strip_to_none(server)
    endpoint_url = strip_to_none(endpoint)
    if (server_name is None) == (endpoint_url is None):
        typer.echo("Provide exactly one configured SERVER or --endpoint <URL>.", err=True)
        raise typer.Exit(2)
    if endpoint_url is not None:
        return _validated_endpoint(endpoint_url, transport)
    assert server_name is not None
    if "://" in server_name:
        typer.echo(
            "MCP positional values are configured server names. "
            "Use --endpoint <URL> for an ad-hoc endpoint.",
            err=True,
        )
        raise typer.Exit(2)
    if transport is not None:
        typer.echo("--transport is only valid with --endpoint.", err=True)
        raise typer.Exit(2)

    settings = get_settings_or_exit(config_path)
    server_config = _configured_server(
        settings,
        server_name,
        url_guidance=(
            "MCP positional values are configured server names. "
            "Use --endpoint <URL> for an ad-hoc endpoint."
        ),
    )
    if not uses_mcp_remote_transport(server_config.transport):
        typer.echo("Only HTTP and SSE MCP servers support OAuth login.", err=True)
        raise typer.Exit(1)
    return LoginTargetConfig(
        server=server_config,
        transport=cast("Literal['http', 'sse']", server_config.transport),
        configured_name=server_name,
    )


async def _run_login_session(
    cfg: MCPServerSettings,
    timeout_seconds: float,
    configured_name: str | None,
) -> bool:
    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
    from fast_agent.mcp.client_gateway import open_request_scoped_client
    from fast_agent.mcp.failures import classify_mcp_failure, render_mcp_failure

    server_name = cfg.name or cfg.url or "mcp"
    callbacks = MCPClientCallbackRuntime(server_name=server_name, server_config=cfg)
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    login_timeout = asyncio.timeout(timeout_seconds)
    try:
        async with login_timeout:
            async with open_request_scoped_client(
                server_name=server_name,
                config=cfg,
                callback_runtime=callbacks,
                trigger_oauth=True,
            ) as connection:
                await connection.list_tools(cache_mode="refresh")
        return True
    except Exception as exc:
        if (
            isinstance(exc, TimeoutError)
            or login_timeout.expired()
            or asyncio.get_running_loop().time() >= deadline
        ):
            typer.echo(
                f"MCP OAuth login timed out after {timeout_seconds:g} seconds. "
                "Increase --timeout and retry.",
                err=True,
            )
            return False
        failure = classify_mcp_failure(
            exc,
            server_name=configured_name or server_name,
            origin="central" if configured_name else "session",
            surface="configured_attach" if configured_name else "terminal_connect",
            input_ref=configured_name or cfg.url or server_name,
            stage="auth",
        )
        typer.echo(render_mcp_failure(failure), err=True)
        return False


@mcp_app.command("login")
def mcp_login(
    server: str | None = typer.Argument(None, help="Configured MCP server name"),
    endpoint: str | None = typer.Option(
        None,
        "--endpoint",
        help="Exact ad-hoc MCP endpoint URL",
    ),
    transport: str | None = typer.Option(
        None,
        "--transport",
        help="Transport for --endpoint: http or sse",
    ),
    timeout_seconds: float = typer.Option(
        300.0,
        "--timeout",
        min=1.0,
        help="Maximum seconds to wait for login and MCP initialization",
    ),
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
) -> None:
    """Authenticate a configured server or exact ad-hoc endpoint with OAuth."""
    resolved = _resolve_login_target(
        server,
        endpoint=endpoint,
        transport=transport,
        config_path=config_path,
    )
    config = resolved.server
    if config.management == "provider":
        typer.echo("Provider-managed MCP servers do not use local OAuth credentials.", err=True)
        raise typer.Exit(1)
    if config.auth is not None and config.auth.persist == "memory":
        typer.echo(
            "Proactive login requires persistent keyring storage; "
            "this server is configured with auth.persist: memory.",
            err=True,
        )
        raise typer.Exit(1)
    if resolve_oauth_mode(config, trigger_oauth=True) != "force":
        typer.echo("OAuth is disabled for this server.", err=True)
        raise typer.Exit(1)

    keyring_status = get_keyring_status()
    if not keyring_status.writable:
        typer.echo(
            "Proactive login requires a writable OS keyring. "
            "Install or unlock a keyring backend, then retry.",
            err=True,
        )
        raise typer.Exit(1)

    resource = compute_server_identity(config)
    print_detail_line(console, "server", config.name or "-")
    print_detail_line(console, "endpoint", redact_mcp_url(config.url) if config.url else "-")
    print_detail_line(console, "OAuth resource", redact_mcp_url(resource))
    print_detail_line(console, "storage", keyring_status.name)
    if not run_sync(
        _run_login_session,
        config,
        timeout_seconds,
        resolved.configured_name,
    ):
        raise typer.Exit(1)
    typer.echo(f"Authenticated. Stored MCP credential for: {redact_mcp_url(resource)}")


def _resource_candidates(resource: str) -> tuple[str, ...]:
    try:
        parsed = urlparse(resource)
        _ = parsed.port
    except ValueError as exc:
        typer.echo(f"Invalid --resource URL: {exc}", err=True)
        raise typer.Exit(2) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        typer.echo("--resource must be an absolute HTTP(S) URL.", err=True)
        raise typer.Exit(2)
    if parsed.username is not None or parsed.password is not None:
        typer.echo("--resource must not contain URL user information.", err=True)
        raise typer.Exit(2)
    return oauth_resource_key_candidates(resource)


def _resolve_forget_resources(
    settings: Settings,
    *,
    server: str | None,
    resource: str | None,
    all_credentials: bool,
) -> list[str]:
    server_name = strip_to_none(server)
    resource_url = strip_to_none(resource)
    selectors = int(server_name is not None) + int(resource_url is not None) + int(all_credentials)
    if selectors != 1:
        typer.echo("Provide exactly one SERVER, --resource <URL>, or --all.", err=True)
        raise typer.Exit(2)
    if all_credentials:
        return list_keyring_credentials()
    if resource_url is not None:
        return list(_resource_candidates(resource_url))

    assert server_name is not None
    server_config = _configured_server(
        settings,
        server_name,
        url_guidance=(
            "`auth mcp forget` accepts a configured server name. "
            "Use --resource <URL> to forget a stored OAuth resource."
        ),
    )
    if _resource_for_server(server_config) is None:
        typer.echo(f"Configured MCP server '{server_name}' has no OAuth resource.", err=True)
        raise typer.Exit(1)
    return list(compute_server_identity_candidates(server_config))


def _print_forget_preview(
    resources: list[str],
    *,
    servers_by_resource: dict[str, tuple[str, ...]],
) -> None:
    console.print("[bold]Stored MCP credentials to forget:[/bold]")
    for resource in resources:
        console.print(f"  {redact_mcp_url(resource)}")
        canonical = _derive_base_server_url(resource) or resource
        servers = servers_by_resource.get(canonical, ())
        if servers:
            console.print(f"    configured servers: {', '.join(servers)}")
        else:
            console.print("    configured servers: [dim]none (orphaned)[/dim]")
    console.print(
        "\n[dim]This removes local OAuth tokens and client registration. "
        "Server configuration and runtime connections are unchanged.[/dim]"
    )


@mcp_app.command("forget")
def mcp_forget(
    server: str | None = typer.Argument(None, help="Configured MCP server name"),
    resource: str | None = typer.Option(
        None,
        "--resource",
        help="Stored OAuth resource URL",
    ),
    all_credentials: bool = typer.Option(
        False,
        "--all",
        help="Forget every indexed MCP OAuth credential",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt for confirmation"),
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
) -> None:
    """Forget local MCP OAuth tokens and client registration."""
    settings = get_settings_or_exit(config_path)
    _backfill_configured_credential_index(settings)
    resources = _resolve_forget_resources(
        settings,
        server=server,
        resource=resource,
        all_credentials=all_credentials,
    )
    targets = [
        resource_url for resource_url in resources if keyring_credential_present(resource_url)
    ]
    if not targets:
        typer.echo("No stored MCP credentials matched the selection.")
        return

    _print_forget_preview(
        targets,
        servers_by_resource=_credential_consumers_by_resource(settings),
    )
    if not _confirm_destructive_action(
        prompt="Forget these stored credentials?",
        yes=yes,
    ):
        typer.echo("Cancelled; no MCP credentials were removed.")
        return

    removed = sum(clear_keyring_token(resource_url) for resource_url in targets)
    typer.echo(
        f"Forgot {removed} stored MCP credential{'s' if removed != 1 else ''}."
        if removed
        else "No stored MCP credentials were removed."
    )


def _migration_error(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(2)


@app.command("login", hidden=True)
def legacy_provider_login(provider: str = typer.Argument(...)) -> None:
    _migration_error(
        "`fast-agent auth login` was removed in 0.10. "
        f"Use `fast-agent auth provider login {provider}`."
    )


@app.command("logout", hidden=True)
def legacy_provider_logout(
    provider: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    del yes
    _migration_error(
        "`fast-agent auth logout` was removed in 0.10. "
        f"Use `fast-agent auth provider logout {provider}`."
    )


@app.command("token", hidden=True)
def legacy_provider_token(provider: str = typer.Argument(...)) -> None:
    _migration_error(
        "`fast-agent auth token` was removed in 0.10. "
        f"Use `fast-agent auth provider token {provider}`."
    )


@app.command("export", hidden=True)
def legacy_provider_export(
    provider: str = typer.Argument(...),
    output: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force"),
) -> None:
    del force
    _migration_error(
        "`fast-agent auth export` was removed in 0.10. "
        f"Use `fast-agent auth provider export {provider} {output}`."
    )


@app.command("status", hidden=True)
def legacy_provider_status(provider: str | None = typer.Argument(None)) -> None:
    replacement = (
        f"`fast-agent auth provider show {provider}`"
        if provider
        else "`fast-agent auth provider list`"
    )
    _migration_error(
        "`fast-agent auth status` was removed in 0.10. "
        f"Use {replacement}, or `fast-agent auth` for the combined overview."
    )


@mcp_app.command("status", hidden=True)
def legacy_mcp_status(
    target: str | None = typer.Argument(None),
    config_path: str | None = typer.Option(None, "--config-path", "-c"),
) -> None:
    del config_path
    if target and "://" not in target:
        replacement = f"`fast-agent auth mcp show {target}`"
    elif target:
        replacement = "`fast-agent auth mcp credentials`"
    else:
        replacement = "`fast-agent auth mcp list`"
    _migration_error(f"`fast-agent auth mcp status` was removed in 0.10. Use {replacement}.")


@mcp_app.command("logout", hidden=True)
def legacy_mcp_logout(
    server: str | None = typer.Argument(None),
    resource: str | None = typer.Option(None, "--identity", "--resource"),
    all_credentials: bool = typer.Option(False, "--all"),
    config_path: str | None = typer.Option(None, "--config-path", "-c"),
) -> None:
    del config_path
    if all_credentials:
        replacement = "`fast-agent auth mcp forget --all`"
    elif resource:
        replacement = f"`fast-agent auth mcp forget --resource {redact_mcp_url(resource)}`"
    elif server:
        replacement = f"`fast-agent auth mcp forget {server}`"
    else:
        replacement = "`fast-agent auth mcp forget --help`"
    _migration_error(f"`fast-agent auth mcp logout` was removed in 0.10. Use {replacement}.")
