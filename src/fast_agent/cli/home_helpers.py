from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.constants import FAST_AGENT_RUNTIME_HOME

if TYPE_CHECKING:
    import typer


def _coerce_home_path(value: Path | str | object | None) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    return None


def _parent_home_option(ctx: typer.Context | None) -> Path | None:
    if ctx is None or ctx.parent is None:
        return None
    return _coerce_home_path(ctx.parent.params.get("home"))


def _resolve_home_path(value: Path) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        return (Path.cwd() / expanded).resolve()
    return expanded.resolve()


def _restore_env_value(name: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous_value


def _set_home_for_context(ctx: typer.Context | None, resolved: Path) -> None:
    previous_home = os.environ.get("FAST_AGENT_HOME")
    previous_runtime_env = os.environ.get(FAST_AGENT_RUNTIME_HOME)
    os.environ["FAST_AGENT_HOME"] = str(resolved)
    os.environ[FAST_AGENT_RUNTIME_HOME] = str(resolved)
    if ctx is None:
        return

    def restore_home() -> None:
        _restore_env_value("FAST_AGENT_HOME", previous_home)
        _restore_env_value(FAST_AGENT_RUNTIME_HOME, previous_runtime_env)

    ctx.call_on_close(restore_home)


def resolve_home_option(
    ctx: typer.Context | None,
    home: Path | None,
    *,
    set_env_var: bool = True,
) -> Path | None:
    resolved = _coerce_home_path(home)
    if resolved is None and ctx is not None:
        resolved = _parent_home_option(ctx)

    if resolved is None:
        return None

    resolved = _resolve_home_path(resolved)
    if set_env_var:
        _set_home_for_context(ctx, resolved)
    return resolved
