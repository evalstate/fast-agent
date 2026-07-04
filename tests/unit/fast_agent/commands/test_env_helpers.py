from __future__ import annotations

import os
from pathlib import Path

from fast_agent.cli.home_helpers import resolve_home_option
from fast_agent.constants import FAST_AGENT_RUNTIME_HOME


def test_resolve_home_option_returns_absolute_path(tmp_path: Path) -> None:
    original_env = os.environ.get("FAST_AGENT_HOME")
    original_runtime_env = os.environ.get(FAST_AGENT_RUNTIME_HOME)
    original_cwd = Path.cwd()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    os.environ.pop("FAST_AGENT_HOME", None)
    try:
        os.chdir(workspace)
        resolved = resolve_home_option(None, Path(".dev"))
        assert resolved == (workspace / ".dev").resolve()
        assert os.environ.get("FAST_AGENT_HOME") == str((workspace / ".dev").resolve())
        assert os.environ.get(FAST_AGENT_RUNTIME_HOME) == str((workspace / ".dev").resolve())
    finally:
        os.chdir(original_cwd)
        if original_env is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = original_env
        if original_runtime_env is None:
            os.environ.pop(FAST_AGENT_RUNTIME_HOME, None)
        else:
            os.environ[FAST_AGENT_RUNTIME_HOME] = original_runtime_env


def test_resolve_home_option_can_skip_environment_mutation(tmp_path: Path) -> None:
    original_env = os.environ.get("FAST_AGENT_HOME")
    original_runtime_env = os.environ.get(FAST_AGENT_RUNTIME_HOME)
    original_cwd = Path.cwd()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    os.environ["FAST_AGENT_HOME"] = "do-not-change"
    os.environ[FAST_AGENT_RUNTIME_HOME] = "do-not-change"
    try:
        os.chdir(workspace)
        resolved = resolve_home_option(
            None,
            Path(".dev"),
            set_env_var=False,
        )
        assert resolved == (workspace / ".dev").resolve()
        assert os.environ.get("FAST_AGENT_HOME") == "do-not-change"
        assert os.environ.get(FAST_AGENT_RUNTIME_HOME) == "do-not-change"
    finally:
        os.chdir(original_cwd)
        if original_env is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = original_env
        if original_runtime_env is None:
            os.environ.pop(FAST_AGENT_RUNTIME_HOME, None)
        else:
            os.environ[FAST_AGENT_RUNTIME_HOME] = original_runtime_env
