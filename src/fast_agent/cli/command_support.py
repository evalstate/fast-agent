"""Shared helpers for top-level CLI command state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import typer


def ensure_context_object(ctx: typer.Context) -> dict[str, Any]:
    """Return a mutable context object dictionary for a Typer command tree."""
    if isinstance(ctx.obj, dict):
        return ctx.obj
    if ctx.obj is None:
        ctx.obj = {}
        return ctx.obj
    return {}


def resolve_context_string_option(
    ctx: typer.Context,
    *,
    key: str,
    command_value: str | None = None,
) -> str | None:
    """Resolve a string option from the current command, then the shared context."""
    if command_value:
        return command_value
    ctx_value = ensure_context_object(ctx).get(key)
    if isinstance(ctx_value, str) and ctx_value.strip():
        return ctx_value
    return None


def resolve_context_path_option(
    ctx: typer.Context,
    *,
    key: str,
    command_value: Path | None = None,
) -> Path | None:
    """Resolve a path option from the current command, then the shared context."""
    if command_value is not None:
        return command_value
    ctx_value = ensure_context_object(ctx).get(key)
    if isinstance(ctx_value, Path):
        return ctx_value
    if isinstance(ctx_value, str) and ctx_value.strip():
        return Path(ctx_value)
    return None
