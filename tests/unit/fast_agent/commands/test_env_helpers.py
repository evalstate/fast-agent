from __future__ import annotations

import os
from pathlib import Path

from fast_agent.cli.env_helpers import resolve_environment_dir_option


def test_resolve_environment_dir_option_returns_absolute_path(tmp_path: Path) -> None:
    original_env = os.environ.get("ENVIRONMENT_DIR")
    original_cwd = Path.cwd()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(workspace)
        resolved = resolve_environment_dir_option(None, Path(".dev"))
        assert resolved == (workspace / ".dev").resolve()
        assert os.environ.get("ENVIRONMENT_DIR") == str((workspace / ".dev").resolve())
    finally:
        os.chdir(original_cwd)
        if original_env is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env


def test_resolve_environment_dir_option_can_skip_environment_mutation(tmp_path: Path) -> None:
    original_env = os.environ.get("ENVIRONMENT_DIR")
    original_cwd = Path.cwd()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    os.environ["ENVIRONMENT_DIR"] = "do-not-change"
    try:
        os.chdir(workspace)
        resolved = resolve_environment_dir_option(
            None,
            Path(".dev"),
            set_env_var=False,
        )
        assert resolved == (workspace / ".dev").resolve()
        assert os.environ.get("ENVIRONMENT_DIR") == "do-not-change"
    finally:
        os.chdir(original_cwd)
        if original_env is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env
