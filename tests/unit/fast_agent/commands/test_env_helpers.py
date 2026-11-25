from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from fast_agent.cli.home_helpers import resolve_home_option, resolve_workspace_and_home_options
from fast_agent.constants import FAST_AGENT_RUNTIME_HOME

if TYPE_CHECKING:
    import pytest


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


def test_resolve_workspace_without_explicit_home_does_not_leak_fast_agent_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.setenv(FAST_AGENT_RUNTIME_HOME, "/leaked/runtime/home")

    app = typer.Typer()
    app.command()(lambda: None)
    ctx = typer.Context(typer.main.get_command(app))
    with ctx:
        resolved_workspace, resolved_home = resolve_workspace_and_home_options(
            ctx,
            workspace=workspace,
            home=None,
        )
        assert resolved_workspace == workspace.resolve()
        assert resolved_home == (workspace / ".fast-agent").resolve()
        assert "FAST_AGENT_HOME" not in os.environ
        assert FAST_AGENT_RUNTIME_HOME not in os.environ

    assert os.environ.get(FAST_AGENT_RUNTIME_HOME) == "/leaked/runtime/home"
