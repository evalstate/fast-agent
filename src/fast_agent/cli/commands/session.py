"""Persisted session maintenance commands."""

from __future__ import annotations

from pathlib import Path

import typer

from fast_agent.cli.command_support import ensure_context_object
from fast_agent.cli.home_helpers import resolve_home_option
from fast_agent.session import SessionManager

app = typer.Typer(
    help="Maintain persisted sessions.",
    add_completion=False,
)


@app.callback()
def session() -> None:
    """Maintain persisted sessions."""


@app.command()
def prune(
    ctx: typer.Context,
    empty: bool = typer.Option(
        False,
        "--empty",
        help="Remove sessions containing only disposable startup metadata.",
    ),
) -> None:
    """Remove disposable persisted sessions."""
    if not empty:
        typer.echo("Specify what to prune with --empty.", err=True)
        raise typer.Exit(2)

    home_value = ensure_context_object(ctx).get("home")
    home = home_value if isinstance(home_value, Path) else None
    manager = SessionManager(home_override=resolve_home_option(ctx, home))
    removed = manager.prune_empty_sessions()
    noun = "session" if removed == 1 else "sessions"
    typer.echo(f"Removed {removed} empty {noun}.")
