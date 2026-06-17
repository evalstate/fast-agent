from __future__ import annotations

import os
from pathlib import Path

from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.constants import FAST_AGENT_RUNTIME_ENVIRONMENT


def test_resolve_environment_dir_option_returns_absolute_path(tmp_path: Path) -> None:
    original_env = os.environ.get("ENVIRONMENT_DIR")
    original_runtime_env = os.environ.get(FAST_AGENT_RUNTIME_ENVIRONMENT)
    original_cwd = Path.cwd()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(workspace)
        resolved = resolve_environment_dir_option(None, Path(".dev"))
        assert resolved == (workspace / ".dev").resolve()
        assert os.environ.get("ENVIRONMENT_DIR") == str((workspace / ".dev").resolve())
        assert os.environ.get(FAST_AGENT_RUNTIME_ENVIRONMENT) == str((workspace / ".dev").resolve())
    finally:
        os.chdir(original_cwd)
        if original_env is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env
        if original_runtime_env is None:
            os.environ.pop(FAST_AGENT_RUNTIME_ENVIRONMENT, None)
        else:
            os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = original_runtime_env


def test_resolve_environment_dir_option_can_skip_environment_mutation(tmp_path: Path) -> None:
    original_env = os.environ.get("ENVIRONMENT_DIR")
    original_runtime_env = os.environ.get(FAST_AGENT_RUNTIME_ENVIRONMENT)
    original_cwd = Path.cwd()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    os.environ["ENVIRONMENT_DIR"] = "do-not-change"
    os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = "do-not-change"
    try:
        os.chdir(workspace)
        resolved = resolve_environment_dir_option(
            None,
            Path(".dev"),
            set_env_var=False,
        )
        assert resolved == (workspace / ".dev").resolve()
        assert os.environ.get("ENVIRONMENT_DIR") == "do-not-change"
        assert os.environ.get(FAST_AGENT_RUNTIME_ENVIRONMENT) == "do-not-change"
    finally:
        os.chdir(original_cwd)
        if original_env is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env
        if original_runtime_env is None:
            os.environ.pop(FAST_AGENT_RUNTIME_ENVIRONMENT, None)
        else:
            os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = original_runtime_env
