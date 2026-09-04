"""Persisted session maintenance commands."""

from __future__ import annotations

from pathlib import Path

import typer

from fast_agent.cli.command_support import ensure_context_object
from fast_agent.cli.home_helpers import resolve_home_option
from fast_agent.session import SessionManager
from fast_agent.session.locking import SessionCheckpointBusyError

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
    manager = SessionManager(
        home_override=resolve_home_option(ctx, home),
        surface="maintenance",
    )
    try:
        result = manager.prune_empty_sessions_result()
    finally:
        manager.close()
    removed = result.removed
    noun = "session" if removed == 1 else "sessions"
    typer.echo(f"Removed {removed} empty {noun}.")
    if result.busy:
        typer.echo(f"Skipped {len(result.busy)} active sessions.", err=True)


@app.command("fork")
def fork_session(
    ctx: typer.Context,
    session_id: str = typer.Argument(help="Persisted source session ID or list number."),
    title: str | None = typer.Option(None, "--title", help="Title for the forked session."),
) -> None:
    """Fork the latest committed checkpoint of a persisted session."""
    home_value = ensure_context_object(ctx).get("home")
    home = home_value if isinstance(home_value, Path) else None
    manager = SessionManager(
        home_override=resolve_home_option(ctx, home),
        surface="maintenance",
    )
    try:
        forked = manager.fork_session(session_id, title=title)
        if forked is None:
            typer.echo(f"Session not found: {session_id}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Forked session {session_id} as {forked.info.name}.")
    except SessionCheckpointBusyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    finally:
        manager.close()
