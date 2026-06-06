"""Authentication management commands for fast-agent.

Shows keyring backend, per-server OAuth token status, and provides a way to clear tokens.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import typer
from rich.table import Table

from fast_agent.cli.codex_oauth_display import (
    codex_oauth_expiry_display,
    codex_oauth_source_display,
)
from fast_agent.cli.command_support import get_settings_or_exit
from fast_agent.cli.display import print_detail_line, print_section_header
from fast_agent.core.keyring_utils import get_keyring_status
from fast_agent.mcp.oauth_client import (
    _derive_base_server_url,
    clear_keyring_token,
    compute_server_identity,
    keyring_token_present,
    list_keyring_tokens,
)
from fast_agent.ui.console import console
from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.async_utils import run_sync
from fast_agent.utils.text import strip_to_none
from fast_agent.utils.transports import uses_mcp_remote_transport

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings, Settings
    from fast_agent.mcp.oauth_client import OAuthClientProvider

app = typer.Typer(
    help=(
        "Manage OAuth tokens stored in the OS keyring for MCP HTTP/SSE servers "
        "(identity = base URL)."
    ),
    add_completion=False,
)


@dataclass(slots=True, frozen=True)
class AuthServerRow:
    name: str
    transport: str
    url: str
    persist: str
    oauth: bool
    has_token: bool
    identity: str


@dataclass(slots=True, frozen=True)
class LoginTargetConfig:
    server: MCPServerSettings
    transport: str


def _configured_mcp_servers(settings: Settings) -> dict[str, MCPServerSettings]:
    if settings.mcp is None:
        return {}
    return settings.mcp.servers


def _print_keyring_backend_status(*, backend: str, usable: bool) -> None:
    value = backend if usable and backend != "unavailable" else "not available"
    value_style = "green" if usable and backend != "unavailable" else "red"
    print_detail_line(console, "keyring backend", value, value_style=value_style)


def _server_rows_from_settings(settings: Settings) -> list[AuthServerRow]:
    rows = []
    for name, cfg in _configured_mcp_servers(settings).items():
        transport = cfg.transport
        if transport == "stdio":
            # STDIO servers do not use OAuth; skip in auth views
            continue
        auth = cfg.auth
        oauth_enabled = auth.oauth if auth is not None else True
        persist = auth.persist if auth is not None else "keyring"
        identity = compute_server_identity(cfg)
        # token presence only meaningful if persist is keyring and transport is remote MCP
        is_remote_transport = uses_mcp_remote_transport(transport)
        has_token = False
        if persist == "keyring" and is_remote_transport and oauth_enabled:
            has_token = keyring_token_present(identity)
        rows.append(
            AuthServerRow(
                name=name,
                transport=transport,
                url=cfg.url or "",
                persist=persist,
                oauth=oauth_enabled and is_remote_transport,
                has_token=has_token,
                identity=identity,
            )
        )
    return rows


def _servers_by_identity(settings: Settings) -> dict[str, list[str]]:
    """Group configured server names by derived identity (base URL)."""
    mapping: dict[str, list[str]] = {}
    for name, cfg in _configured_mcp_servers(settings).items():
        try:
            identity = compute_server_identity(cfg)
        except Exception:
            identity = name
        mapping.setdefault(identity, []).append(name)
    return mapping


def _resolve_status_identity(target: str, settings: Settings) -> str:
    if "://" in target:
        identity = _derive_base_server_url(target)
        if identity:
            return identity

    cfg = _configured_mcp_servers(settings).get(target)
    if cfg is None:
        typer.echo(f"Server '{target}' not found in config; treating as identity")
        return target
    return compute_server_identity(cfg)


def _print_target_status(
    *,
    identity: str,
    settings: Settings,
    backend: str,
    backend_usable: bool,
) -> None:
    present = keyring_token_present(identity) if backend_usable else False

    table = Table(show_header=True, box=None)
    table.add_column("Identity", header_style="bold")
    table.add_column("Token", header_style="bold")
    table.add_column("Servers", header_style="bold")
    by_id = _servers_by_identity(settings)
    servers_for_id = ", ".join(by_id.get(identity, [])) or "[dim]None[/dim]"
    token_disp = "[bold green]✓[/bold green]" if present else "[dim]✗[/dim]"
    table.add_row(identity, token_disp, servers_for_id)

    _print_keyring_backend_status(backend=backend, usable=backend_usable)
    console.print(table)
    console.print(
        "\n[dim]Run 'fast-agent auth clear --identity "
        f"{identity}[/dim][dim]' to remove this token, or 'fast-agent auth clear --all' to remove all.[/dim]"
    )


def _print_stored_tokens_status() -> None:
    tokens = list_keyring_tokens()
    token_table = Table(show_header=True, box=None)
    token_table.add_column("Stored Tokens (Identity)", header_style="bold")
    token_table.add_column("Present", header_style="bold")
    if tokens:
        for ident in tokens:
            token_table.add_row(ident, "[bold green]✓[/bold green]")
    else:
        token_table.add_row("[dim]None[/dim]", "[dim]✗[/dim]")
    console.print(token_table)


def _token_display_for_server_row(row: AuthServerRow, *, backend_usable: bool) -> str:
    if row.persist == "keyring" and row.oauth:
        if backend_usable:
            return "[bold green]✓[/bold green]" if row.has_token else "[dim]✗[/dim]"
        return "[red]not available[/red]"
    if row.persist == "memory" and row.oauth:
        return "[yellow]memory[/yellow]"
    return "[dim]✗[/dim]"


def _print_configured_servers_status(
    rows: list[AuthServerRow], *, backend_usable: bool
) -> None:
    if not rows:
        return

    map_table = Table(show_header=True, box=None)
    map_table.add_column("Server", header_style="bold")
    map_table.add_column("Transport", header_style="bold")
    map_table.add_column("OAuth", header_style="bold")
    map_table.add_column("Persist", header_style="bold")
    map_table.add_column("Token", header_style="bold")
    map_table.add_column("Identity", header_style="bold")
    for row in rows:
        oauth_status = "[green]on[/green]" if row.oauth else "[dim]off[/dim]"
        persist_disp = (
            f"[green]{row.persist}[/green]"
            if row.persist == "keyring"
            else f"[yellow]{row.persist}[/yellow]"
        )
        map_table.add_row(
            row.name,
            row.transport.upper(),
            oauth_status,
            persist_disp,
            _token_display_for_server_row(row, backend_usable=backend_usable),
            row.identity,
        )
    console.print(map_table)


def _codex_token_status() -> dict[str, object] | None:
    with suppress(Exception):
        from fast_agent.llm.provider.openai.codex_oauth import get_codex_token_status

        return get_codex_token_status()
    return None


def _print_codex_oauth_status() -> None:
    from datetime import datetime

    codex_status = _codex_token_status()
    if codex_status is None:
        return

    codex_table = Table(show_header=True, box=None)
    codex_table.add_column("Codex OAuth", style="white", header_style="bold")
    codex_table.add_column("Token", header_style="bold")
    codex_table.add_column("Source", header_style="bold")
    codex_table.add_column("Expires", header_style="bold")

    if not codex_status.get("present"):
        token_display = "[dim]Not configured[/dim]"
        source_display = "[dim]-[/dim]"
        expires_display = "[dim]-[/dim]"
    else:
        token_display = "[bold green]Present[/bold green]"
        source_display = codex_oauth_source_display(codex_status)
        expires_display = codex_oauth_expiry_display(codex_status, datetime_type=datetime)

    codex_table.add_row("Token", token_display, source_display, expires_display)
    print_section_header(console, "Codex OAuth", color="blue")
    console.print(codex_table)


def _echo_missing_login_target() -> None:
    typer.echo("Provide a server name or identity URL to log in.")
    typer.echo(
        "Example: `fast-agent auth login my-server` "
        "or `fast-agent auth login https://example.com`."
    )
    typer.echo("Run `fast-agent auth login --help` for more details.")


def _validated_identity_transport(transport: str | None) -> Literal["http", "sse"]:
    resolved_transport = normalize_action_token(transport or "http")
    if not uses_mcp_remote_transport(resolved_transport):
        typer.echo("--transport must be 'http' or 'sse'")
        raise typer.Exit(1)
    return cast("Literal['http', 'sse']", resolved_transport)


def _login_config_from_identity(target: str, transport: str | None) -> LoginTargetConfig:
    from fast_agent.config import MCPServerAuthSettings, MCPServerSettings

    base = _derive_base_server_url(target)
    if not base:
        typer.echo("Invalid identity URL")
        raise typer.Exit(1)

    resolved_transport = _validated_identity_transport(transport)
    endpoint = base + ("/mcp" if resolved_transport == "http" else "/sse")
    server_transport = cast("Literal['stdio', 'sse', 'http']", resolved_transport)
    return LoginTargetConfig(
        server=MCPServerSettings(
            name=base,
            transport=server_transport,
            url=endpoint,
            auth=MCPServerAuthSettings(),
        ),
        transport=resolved_transport,
    )


def _login_config_from_server_name(target: str, config_path: str | None) -> LoginTargetConfig:
    settings = get_settings_or_exit(config_path)
    cfg = _configured_mcp_servers(settings).get(target)
    if cfg is None:
        typer.echo(f"Server '{target}' not found in config")
        raise typer.Exit(1)
    if cfg.transport == "stdio":
        typer.echo("STDIO servers do not support OAuth")
        raise typer.Exit(1)
    return LoginTargetConfig(server=cfg, transport=cfg.transport)


def _resolve_login_target(
    target: str | None, *, transport: str | None, config_path: str | None
) -> LoginTargetConfig:
    stripped_target = strip_to_none(target)
    if stripped_target is None:
        _echo_missing_login_target()
        raise typer.Exit(1)

    if "://" in stripped_target:
        return _login_config_from_identity(stripped_target, transport)
    return _login_config_from_server_name(stripped_target, config_path)


async def _run_login_session(
    cfg: MCPServerSettings,
    provider: OAuthClientProvider,
    resolved_transport: str,
) -> bool:
    try:
        # Use appropriate transport; connect and initialize a minimal session.
        if resolved_transport == "http":
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with (
                streamablehttp_client(cfg.url or "", cfg.headers, auth=provider) as (
                    read_stream,
                    write_stream,
                    _get_session_id,
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                return True
        if resolved_transport == "sse":
            from mcp.client.session import ClientSession
            from mcp.client.sse import sse_client

            async with (
                sse_client(cfg.url or "", cfg.headers, auth=provider) as (
                    read_stream,
                    write_stream,
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                return True
        return False
    except Exception as e:
        # Surface concise error; detailed logging is in the library.
        typer.echo(f"Login failed: {e}")
        return False


@app.command()
def status(
    target: str | None = typer.Argument(None, help="Identity (base URL) or server name"),
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
) -> None:
    """Show keyring backend and token status for configured MCP servers (identity = base URL)."""
    settings = get_settings_or_exit(config_path)
    keyring_status = get_keyring_status()
    backend = keyring_status.name
    backend_usable = keyring_status.available

    # Single-target view if target provided
    if target:
        _print_target_status(
            identity=_resolve_status_identity(target, settings),
            settings=settings,
            backend=backend,
            backend_usable=backend_usable,
        )
        return

    # Full status view
    _print_keyring_backend_status(backend=backend, usable=backend_usable)
    _print_stored_tokens_status()
    _print_configured_servers_status(
        _server_rows_from_settings(settings), backend_usable=backend_usable
    )
    _print_codex_oauth_status()

    console.print(
        "\n[dim]Run 'fast-agent auth clear --identity <identity>' to remove a token, or 'fast-agent auth clear --all' to remove all.\nCodex OAuth: 'fast-agent auth codexplan' (login) or 'fast-agent auth codex-clear' (remove).[/dim]"
    )


@app.command()
def clear(
    server: str | None = typer.Argument(None, help="Server name to clear (from config)"),
    identity: str | None = typer.Option(
        None, "--identity", help="Token identity (base URL) to clear"
    ),
    all: bool = typer.Option(False, "--all", help="Clear tokens for all identities in keyring"),
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
) -> None:
    """Clear stored OAuth tokens from the keyring by server name or identity (base URL)."""
    targets_identities: list[str] = []
    if all:
        targets_identities = list_keyring_tokens()
    elif identity:
        targets_identities = [identity]
    elif server:
        settings = get_settings_or_exit(config_path)
        rows = _server_rows_from_settings(settings)
        match = next((r for r in rows if r.name == server), None)
        if not match:
            typer.echo(f"Server '{server}' not found in config")
            raise typer.Exit(1)
        targets_identities = [match.identity]
    else:
        typer.echo("Provide --identity, a server name, or use --all")
        raise typer.Exit(1)

    # Confirm destructive action
    if not typer.confirm("Remove the selected keyring tokens?", default=False):
        raise typer.Exit()

    removed_any = False
    for ident in targets_identities:
        if clear_keyring_token(ident):
            removed_any = True
    if removed_any:
        typer.echo("Tokens removed.")
    else:
        typer.echo("No tokens found or nothing removed.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
) -> None:
    """Default to showing status if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        try:
            status(target=None, config_path=config_path)
        except typer.Exit:
            raise
        except Exception as e:
            typer.echo(f"Error showing auth status: {e}")


@app.command("codex-login")
def codex_login() -> None:
    """Start OAuth flow for Codex and store tokens in the keyring."""
    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error
    from fast_agent.llm.provider.openai.codex_oauth import login_codex_oauth

    try:
        login_codex_oauth()
        typer.echo("Codex OAuth login complete. Tokens stored in keyring.")
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc))
        raise typer.Exit(1) from exc


@app.command("codexplan")
def codexplan() -> None:
    """Ensure Codex OAuth tokens are present; optionally start login."""
    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error
    from fast_agent.core.keyring_utils import get_keyring_status
    from fast_agent.llm.provider.openai.codex_oauth import (
        get_codex_token_status,
        login_codex_oauth,
    )

    keyring_status = get_keyring_status()
    if not keyring_status.writable:
        if not keyring_status.available:
            detail = "No usable keyring backend was detected."
        else:
            detail = f"Keyring backend '{keyring_status.name}' is not writable."
        typer.echo(
            "Keyring backend not writable; cannot store Codex OAuth tokens. "
            f"{detail}"
        )
        raise typer.Exit(1)

    status = get_codex_token_status()
    if status.get("present"):
        typer.echo("Codex OAuth token already present in keyring.")
        return

    if not typer.confirm("No Codex OAuth token found. Start OAuth login flow now?", default=True):
        raise typer.Exit(1)

    try:
        login_codex_oauth()
        typer.echo("Codex OAuth login complete. Tokens stored in keyring.")
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc))
        raise typer.Exit(1) from exc


@app.command("codex-clear")
def codex_clear() -> None:
    """Remove Codex OAuth tokens from the keyring."""
    from fast_agent.core.keyring_utils import get_keyring_status
    from fast_agent.llm.provider.openai.codex_oauth import clear_codex_tokens

    status = get_keyring_status()
    if not status.available:
        typer.echo("Keyring backend not available; nothing to clear.")
        raise typer.Exit(1)
    if not status.writable:
        typer.echo(
            "Keyring backend not writable; deletion may fail. "
            f"Detected backend '{status.name}'."
        )

    if not typer.confirm("Remove Codex OAuth tokens from keyring?", default=False):
        raise typer.Exit()

    if clear_codex_tokens():
        typer.echo("Codex OAuth tokens removed.")
    else:
        typer.echo("No Codex OAuth tokens found.")


@app.command()
def login(
    target: str | None = typer.Argument(
        None, help="Server name (from config) or identity (base URL)"
    ),
    transport: str | None = typer.Option(
        None, "--transport", help="Transport for identity mode: http or sse"
    ),
    config_path: str | None = typer.Option(
        None,
        "--config-path",
        "-c",
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to config file",
    ),
) -> None:
    """Start OAuth flow and store tokens in the keyring for a server.

    Accepts either a configured server name or an identity (base URL).
    For identity mode, default transport is 'http' (uses <identity>/mcp).
    """
    # Resolve to a minimal MCPServerSettings
    from fast_agent.mcp.oauth_client import build_oauth_provider

    resolved = _resolve_login_target(target, transport=transport, config_path=config_path)

    # Build OAuth provider
    provider = build_oauth_provider(resolved.server)
    if provider is None:
        typer.echo("OAuth is disabled or misconfigured for this server/identity")
        raise typer.Exit(1)

    ok = bool(
        run_sync(
            _run_login_session,
            resolved.server,
            provider,
            resolved.transport,
        )
    )
    if ok:
        from fast_agent.mcp.oauth_client import compute_server_identity

        ident = compute_server_identity(resolved.server)
        typer.echo(f"Authenticated. Tokens stored for identity: {ident}")
    else:
        raise typer.Exit(1)
