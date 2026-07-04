from __future__ import annotations

import os
from pathlib import Path

import typer


def _coerce_workspace_path(value: Path | str | object | None) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    return None


def _parent_workspace_option(ctx: typer.Context | None) -> Path | None:
    if ctx is None:
        return None
    if isinstance(ctx.obj, dict):
        resolved = _coerce_workspace_path(ctx.obj.get("workspace"))
        if resolved is not None:
            return resolved
    if ctx.parent is None:
        return None
    return _coerce_workspace_path(ctx.parent.params.get("workspace"))


def _resolve_workspace_path(value: Path) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        return (Path.cwd() / expanded).resolve()
    return expanded.resolve()


def _set_workspace_for_context(ctx: typer.Context | None, resolved: Path) -> None:
    previous_cwd = Path.cwd()
    os.chdir(resolved)
    if ctx is None:
        return

    def restore_workspace() -> None:
        os.chdir(previous_cwd)

    ctx.call_on_close(restore_workspace)


def resolve_workspace_option(
    ctx: typer.Context | None,
    workspace: Path | None,
    *,
    change_cwd: bool = True,
) -> Path | None:
    resolved = _coerce_workspace_path(workspace)
    if resolved is None and ctx is not None:
        resolved = _parent_workspace_option(ctx)

    if resolved is None:
        return None

    resolved = _resolve_workspace_path(resolved)
    if not resolved.is_dir():
        raise typer.BadParameter(
            f"Workspace does not exist or is not a directory: {resolved}",
            param_hint="--workspace",
        )
    if change_cwd:
        _set_workspace_for_context(ctx, resolved)
    return resolved
