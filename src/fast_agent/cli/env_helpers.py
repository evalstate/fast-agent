from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.constants import FAST_AGENT_RUNTIME_ENVIRONMENT

if TYPE_CHECKING:
    import typer


def _coerce_env_dir_path(value: Path | str | object | None) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    return None


def _parent_env_dir_option(ctx: typer.Context | None) -> Path | None:
    if ctx is None or ctx.parent is None:
        return None
    return _coerce_env_dir_path(ctx.parent.params.get("env"))


def _resolve_env_dir_path(value: Path) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        return (Path.cwd() / expanded).resolve()
    return expanded.resolve()


def _restore_env_value(name: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous_value


def _set_environment_dir_for_context(ctx: typer.Context | None, resolved: Path) -> None:
    previous_runtime_env = os.environ.get(FAST_AGENT_RUNTIME_ENVIRONMENT)
    previous_legacy_env = os.environ.get("ENVIRONMENT_DIR")
    os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = str(resolved)
    os.environ["ENVIRONMENT_DIR"] = str(resolved)
    if ctx is None:
        return

    def restore_environment_dir() -> None:
        _restore_env_value(FAST_AGENT_RUNTIME_ENVIRONMENT, previous_runtime_env)
        _restore_env_value("ENVIRONMENT_DIR", previous_legacy_env)

    ctx.call_on_close(restore_environment_dir)


def resolve_environment_dir_option(
    ctx: typer.Context | None,
    env_dir: Path | None,
    *,
    set_env_var: bool = True,
) -> Path | None:
    resolved = _coerce_env_dir_path(env_dir)
    if resolved is None and ctx is not None:
        resolved = _parent_env_dir_option(ctx)

    if resolved is None:
        return None

    resolved = _resolve_env_dir_path(resolved)
    if set_env_var:
        _set_environment_dir_for_context(ctx, resolved)
    return resolved
