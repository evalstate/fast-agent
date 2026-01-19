from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def resolve_environment_dir_option(
    ctx: typer.Context | None,
    env_dir: Path | None,
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
        os.environ["ENVIRONMENT_DIR"] = str(resolved)
        return resolved

    return None
