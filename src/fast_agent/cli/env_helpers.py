from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.constants import FAST_AGENT_RUNTIME_ENVIRONMENT

if TYPE_CHECKING:
    import typer


def resolve_environment_dir_option(
    ctx: typer.Context | None,
    env_dir: Path | None,
    *,
    set_env_var: bool = True,
) -> Path | None:
    resolved = env_dir
    if resolved is not None and not isinstance(resolved, (Path, str)):
        resolved = None

    if resolved is None and ctx is not None:
        parent = ctx.parent
        if parent is not None:
            value = parent.params.get("env")
            if isinstance(value, Path):
                resolved = value
            elif isinstance(value, str):
                resolved = Path(value)

    if isinstance(resolved, str):
        resolved = Path(resolved)

    if isinstance(resolved, Path):
        resolved = resolved.expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if set_env_var:
            previous_runtime_env = os.environ.get(FAST_AGENT_RUNTIME_ENVIRONMENT)
            previous_legacy_env = os.environ.get("ENVIRONMENT_DIR")
            os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = str(resolved)
            os.environ["ENVIRONMENT_DIR"] = str(resolved)
            if ctx is not None:

                def restore_environment_dir() -> None:
                    if previous_runtime_env is None:
                        os.environ.pop(FAST_AGENT_RUNTIME_ENVIRONMENT, None)
                    else:
                        os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = previous_runtime_env
                    if previous_legacy_env is None:
                        os.environ.pop("ENVIRONMENT_DIR", None)
                    else:
                        os.environ["ENVIRONMENT_DIR"] = previous_legacy_env

                ctx.call_on_close(restore_environment_dir)
        return resolved

    return None
