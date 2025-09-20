"""Authentication management commands for fast-agent.

Shows keyring backend, per-server OAuth token status, and provides a way to clear tokens.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import typer
from rich.table import Table

from fast_agent.config import Settings, get_settings
from fast_agent.mcp.oauth_client import (
    _derive_base_server_url,
    clear_keyring_token,
    compute_server_identity,
    list_keyring_tokens,
)
from fast_agent.ui.console import console

app = typer.Typer(help="Manage OAuth authentication state for MCP servers")


def _get_keyring_status() -> tuple[str, bool]:
    """Return (backend_name, usable) where usable=False for the fail backend or missing keyring."""
    try:
        import keyring

        kr = keyring.get_keyring()
        name = getattr(kr, "name", kr.__class__.__name__)
        try:
            from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

            return name, not isinstance(kr, FailKeyring)
        except Exception:
            # If fail backend marker cannot be imported, assume usable
            return name, True
    except Exception:
        return "unavailable", False


def _get_keyring_backend_name() -> str:
    # Backwards-compat helper; prefer _get_keyring_status in new code
    name, _ = _get_keyring_status()
    return name


def _keyring_get_password(service: str, username: str) -> str | None:
    try:
        import keyring

        return keyring.get_password(service, username)
    except Exception:
        return None


def _keyring_delete_password(service: str, username: str) -> bool:
    try:
        import keyring

        keyring.delete_password(service, username)
        return True
    except Exception:
        return False


def _server_rows_from_settings(settings: Settings):
    rows = []
    mcp = getattr(settings, "mcp", None)
    servers = getattr(mcp, "servers", {}) if mcp else {}
    for name, cfg in servers.items():
        transport = getattr(cfg, "transport", "")
        if transport == "stdio":
            # STDIO servers do not use OAuth; skip in auth views
            continue
        url = getattr(cfg, "url", None)
        auth = getattr(cfg, "auth", None)
        oauth_enabled = getattr(auth, "oauth", True) if auth is not None else True
        persist = getattr(auth, "persist", "keyring") if auth is not None else "keyring"
        identity = compute_server_identity(cfg)
        # token presence only meaningful if persist is keyring and transport is http/sse
        has_token = False
        if persist == "keyring" and transport in ("http", "sse") and oauth_enabled:
            has_token = (
                _keyring_get_password("fast-agent-mcp", f"oauth:tokens:{identity}") is not None
            )
        rows.append(
            {
                "name": name,
                "transport": transport,
                "url": url or "",
                "persist": persist,
                "oauth": oauth_enabled and transport in ("http", "sse"),
                "has_token": has_token,
                "identity": identity,
            }
        )
    return rows


def _servers_by_identity(settings: Settings) -> Dict[str, List[str]]:
    """Group configured server names by derived identity (base URL)."""
    mapping: Dict[str, List[str]] = {}
    mcp = getattr(settings, "mcp", None)
    servers = getattr(mcp, "servers", {}) if mcp else {}
    for name, cfg in servers.items():
        try:
            identity = compute_server_identity(cfg)
        except Exception:
            identity = name
        mapping.setdefault(identity, []).append(name)
    return mapping


@app.command()
def status(
    target: Optional[str] = typer.Argument(None, help="Identity (base URL) or server name"),
    config_path: Optional[str] = typer.Option(None, "--config-path", "-c"),
) -> None:
    """Show keyring backend and token status for configured MCP servers."""
    settings = get_settings(config_path)
    backend, backend_usable = _get_keyring_status()

    # Single-target view if target provided
    if target:
        settings = get_settings(config_path)
        identity = _derive_base_server_url(target) if "://" in target else None
        if not identity:
            servers = getattr(getattr(settings, "mcp", None), "servers", {}) or {}
            cfg = servers.get(target)
            if not cfg:
                typer.echo(f"Server '{target}' not found in config; treating as identity")
                identity = target
            else:
                identity = compute_server_identity(cfg)

        # Direct presence check
        present = False
        if backend_usable:
            try:
                import keyring

                present = (
                    keyring.get_password("fast-agent-mcp", f"oauth:tokens:{identity}") is not None
                )
            except Exception:
                present = False

        table = Table(show_header=True, box=None)
        table.add_column("Identity", header_style="bold")
        table.add_column("Token", header_style="bold")
        table.add_column("Servers", header_style="bold")
        by_id = _servers_by_identity(settings)
        servers_for_id = ", ".join(by_id.get(identity, [])) or "[dim]None[/dim]"
        token_disp = "[bold green]✓[/bold green]" if present else "[dim]✗[/dim]"
        table.add_row(identity, token_disp, servers_for_id)

        if backend_usable and backend != "unavailable":
            console.print(f"Keyring backend: [green]{backend}[/green]")
        else:
            console.print("Keyring backend: [red]not available[/red]")
        console.print(table)
        console.print(
            "\n[dim]Run 'fast-agent auth clear --identity "
            f"{identity}[/dim][dim]' to remove this token, or 'fast-agent auth clear --all' to remove all.[/dim]"
        )
        return

    # Full status view
    if backend_usable and backend != "unavailable":
        console.print(f"Keyring backend: [green]{backend}[/green]")
    else:
        console.print("Keyring backend: [red]not available[/red]")

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

    rows = _server_rows_from_settings(settings)
    if rows:
        map_table = Table(show_header=True, box=None)
        map_table.add_column("Server", header_style="bold")
        map_table.add_column("Transport", header_style="bold")
        map_table.add_column("OAuth", header_style="bold")
        map_table.add_column("Persist", header_style="bold")
        map_table.add_column("Token", header_style="bold")
        map_table.add_column("Identity", header_style="bold")
        for row in rows:
            oauth_status = "[green]on[/green]" if row["oauth"] else "[dim]off[/dim]"
            persist = row["persist"]
            persist_disp = (
                f"[green]{persist}[/green]"
                if persist == "keyring"
                else f"[yellow]{persist}[/yellow]"
            )
            # Direct presence check for each identity so status works even without index
            has_token = False
            token_disp = "[dim]✗[/dim]"
            if persist == "keyring" and row["oauth"]:
                if backend_usable:
                    try:
                        import keyring

                        has_token = (
                            keyring.get_password(
                                "fast-agent-mcp", f"oauth:tokens:{row['identity']}"
                            )
                            is not None
                        )
                    except Exception:
                        has_token = False
                    token_disp = "[bold green]✓[/bold green]" if has_token else "[dim]✗[/dim]"
                else:
                    token_disp = "[red]not available[/red]"
            elif persist == "memory" and row["oauth"]:
                token_disp = "[yellow]memory[/yellow]"
            map_table.add_row(
                row["name"],
                row["transport"].upper(),
                oauth_status,
                persist_disp,
                token_disp,
                row["identity"],
            )
        console.print(map_table)

    console.print(
        "\n[dim]Run 'fast-agent auth clear --identity <identity>' to remove a token, or 'fast-agent auth clear --all' to remove all.[/dim]"
    )


@app.command()
def clear(
    server: Optional[str] = typer.Argument(None, help="Server name to clear (from config)"),
    identity: Optional[str] = typer.Option(
        None, "--identity", help="Token identity (base URL) to clear"
    ),
    all: bool = typer.Option(False, "--all", help="Clear tokens for all identities in keyring"),
    config_path: Optional[str] = typer.Option(None, "--config-path", "-c"),
) -> None:
    """Clear stored OAuth tokens from the keyring."""
    targets_identities: list[str] = []
    if all:
        targets_identities = list_keyring_tokens()
    elif identity:
        targets_identities = [identity]
    elif server:
        settings = get_settings(config_path)
        rows = _server_rows_from_settings(settings)
        match = next((r for r in rows if r["name"] == server), None)
        if not match:
            typer.echo(f"Server '{server}' not found in config")
            raise typer.Exit(1)
        targets_identities = [match["identity"]]
    else:
        typer.echo("Provide --identity, a server name, or use --all")
        raise typer.Exit(1)

    # Confirm destructive action
    if not typer.confirm("Remove tokens for the selected server(s) from keyring?", default=False):
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
    ctx: typer.Context, config_path: Optional[str] = typer.Option(None, "--config-path", "-c")
) -> None:
    """Default to showing status if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        try:
            status(target=None, config_path=config_path)
        except Exception as e:
            typer.echo(f"Error showing auth status: {e}")


@app.command()
def login(
    target: Optional[str] = typer.Argument(
        None, help="Server name (from config) or identity (base URL)"
    ),
    transport: Optional[str] = typer.Option(
        None, "--transport", help="Transport for identity mode: http or sse"
    ),
    config_path: Optional[str] = typer.Option(None, "--config-path", "-c"),
) -> None:
    """Start OAuth flow and store tokens for a server.

    Accepts either a configured server name or an identity (base URL).
    For identity mode, default transport is 'http' (uses <identity>/mcp).
    """
    # Resolve to a minimal MCPServerSettings
    from fast_agent.config import MCPServerAuthSettings, MCPServerSettings
    from fast_agent.mcp.oauth_client import build_oauth_provider

    cfg = None
    resolved_transport = None

    if target is None or not target.strip():
        typer.echo("Provide a server name or identity URL to log in.")
        typer.echo(
            "Example: `fast-agent auth login my-server` "
            "or `fast-agent auth login https://example.com`."
        )
        typer.echo("Run `fast-agent auth login --help` for more details.")
        raise typer.Exit(1)

    target = target.strip()

    if "://" in target:
        # Identity mode
        base = _derive_base_server_url(target)
        if not base:
            typer.echo("Invalid identity URL")
            raise typer.Exit(1)
        resolved_transport = (transport or "http").lower()
        if resolved_transport not in ("http", "sse"):
            typer.echo("--transport must be 'http' or 'sse'")
            raise typer.Exit(1)
        endpoint = base + ("/mcp" if resolved_transport == "http" else "/sse")
        cfg = MCPServerSettings(
            name=base,
            transport=resolved_transport,
            url=endpoint,
            auth=MCPServerAuthSettings(),
        )
    else:
        # Server name mode
        settings = get_settings(config_path)
        servers = getattr(getattr(settings, "mcp", None), "servers", {}) or {}
        cfg = servers.get(target)
        if not cfg:
            typer.echo(f"Server '{target}' not found in config")
            raise typer.Exit(1)
        resolved_transport = getattr(cfg, "transport", "")
        if resolved_transport == "stdio":
            typer.echo("STDIO servers do not support OAuth")
            raise typer.Exit(1)

    # Build OAuth provider
    provider = build_oauth_provider(cfg)
    if provider is None:
        typer.echo("OAuth is disabled or misconfigured for this server/identity")
        raise typer.Exit(1)

    async def _run_login():
        try:
            # Use appropriate transport; connect and initialize a minimal session
            if resolved_transport == "http":
                from mcp.client.session import ClientSession
                from mcp.client.streamable_http import streamablehttp_client

                async with streamablehttp_client(
                    cfg.url or "",
                    getattr(cfg, "headers", None),
                    auth=provider,
                ) as (read_stream, write_stream, _get_session_id):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        return True
            elif resolved_transport == "sse":
                from mcp.client.session import ClientSession
                from mcp.client.sse import sse_client

                async with sse_client(
                    cfg.url or "",
                    getattr(cfg, "headers", None),
                    auth=provider,
                ) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        return True
            else:
                return False
        except Exception as e:
            # Surface concise error; detailed logging is in the library
            typer.echo(f"Login failed: {e}")
            return False

    import asyncio

    ok = asyncio.run(_run_login())
    if ok:
        from fast_agent.mcp.oauth_client import compute_server_identity

        ident = compute_server_identity(cfg)
        typer.echo(f"Authenticated. Tokens stored for identity: {ident}")
    else:
        raise typer.Exit(1)
