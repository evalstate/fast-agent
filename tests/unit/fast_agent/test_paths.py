from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import fast_agent.config as config_module
from fast_agent.config import Settings
from fast_agent.paths import (
    default_skill_paths,
    resolve_home_dir,
    resolve_home_paths,
    resolve_settings_start_path,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_settings_start_path_uses_project_config_parent(tmp_path: Path) -> None:
    settings = Settings()
    settings._config_file = str(tmp_path / "project" / "fast-agent.yaml")

    assert resolve_settings_start_path(settings) == (tmp_path / "project").resolve()


def test_resolve_settings_start_path_uses_env_config_parent_project(tmp_path: Path) -> None:
    settings = Settings()
    settings._config_file = str(tmp_path / "project" / ".fast-agent" / "fast-agent.yaml")

    assert resolve_settings_start_path(settings) == (tmp_path / "project").resolve()


def test_resolve_settings_start_path_uses_absolute_home_parent(
    tmp_path: Path,
) -> None:
    settings = Settings(home=str(tmp_path / "project" / ".fast-agent"))

    assert resolve_settings_start_path(settings) == (tmp_path / "project").resolve()


def test_resolve_settings_start_path_ignores_fast_agent_home_override_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_repo_home = tmp_path / "other-repo" / ".fast-agent"
    other_repo_home.mkdir(parents=True)
    (other_repo_home / "fast-agent.yaml").write_text("default_model: haiku\n", encoding="utf-8")

    workspace = tmp_path / "current-workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("FAST_AGENT_HOME", str(other_repo_home))
    config_module._settings = None

    try:
        settings = config_module.get_settings()
        assert resolve_settings_start_path(settings) == workspace.resolve()
    finally:
        config_module._settings = None


def test_resolve_settings_start_path_keeps_explicit_config_parent_with_home_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_home = tmp_path / "shared-home" / ".fast-agent"
    override_home.mkdir(parents=True)
    explicit_project = tmp_path / "explicit-project"
    explicit_project.mkdir()
    explicit_config = explicit_project / "fast-agent.yaml"
    explicit_config.write_text("default_model: haiku\n", encoding="utf-8")

    workspace = tmp_path / "elsewhere"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("FAST_AGENT_HOME", str(override_home))
    config_module._settings = None

    try:
        settings = config_module.get_settings(explicit_config)
        assert resolve_settings_start_path(settings) == explicit_project.resolve()
    finally:
        config_module._settings = None


@pytest.mark.parametrize("target_name", ["fast-agent-root", ".fast-agent"])
def test_resolve_settings_start_path_keeps_workspace_for_symlinked_default_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    cache_home = tmp_path / "cache" / target_name
    cache_home.mkdir(parents=True)
    (cache_home / "fast-agent.yaml").write_text("default_model: haiku\n", encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".fast-agent").symlink_to(cache_home, target_is_directory=True)
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    config_module._settings = None

    try:
        settings = config_module.get_settings()
        assert resolve_settings_start_path(settings) == workspace.resolve()
    finally:
        config_module._settings = None


def test_resolve_home_dir_uses_fast_agent_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("FAST_AGENT_HOME", str(home))

    assert resolve_home_dir(Settings(), cwd=tmp_path) == home.resolve()


def test_resolve_home_dir_override_wins_over_fast_agent_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "from-home"))

    assert (
        resolve_home_dir(Settings(), cwd=tmp_path, override=tmp_path / "from-cli")
        == (tmp_path / "from-cli").resolve()
    )


def test_resolve_home_dir_settings_home_wins_over_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "from-home"))

    assert (
        resolve_home_dir(Settings(home="from-settings"), cwd=tmp_path)
        == (tmp_path / "from-settings").resolve()
    )


def test_resolve_home_dir_settings_home_wins_over_cached_home(
    tmp_path: Path,
) -> None:
    settings = Settings(home="configured-home")
    settings._fast_agent_home = str(tmp_path / ".fast-agent")

    assert resolve_home_dir(settings, cwd=tmp_path) == (tmp_path / "configured-home").resolve()


def test_resolve_home_dir_uses_settings_selected_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    settings._fast_agent_home = str(tmp_path / "selected-home")
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "ambient-home"))

    assert resolve_home_dir(settings, cwd=tmp_path) == (tmp_path / "selected-home").resolve()


def test_resolve_home_paths_uses_get_settings_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_home = tmp_path / "selected-home"
    ambient_home = tmp_path / "ambient-home"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAST_AGENT_HOME", str(ambient_home))
    config_module._settings = None

    try:
        settings = config_module.get_settings(home=selected_home)

        assert resolve_home_paths(settings, cwd=tmp_path).root == selected_home.resolve()
        assert (
            resolve_home_paths(settings, cwd=tmp_path).processes
            == selected_home.resolve() / "processes"
        )
    finally:
        config_module._settings = None


def test_resolve_home_paths_rejects_no_home_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    settings._fast_agent_no_home = True
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "ambient-home"))

    with pytest.raises(ValueError, match="fast-agent home is disabled"):
        resolve_home_paths(settings, cwd=tmp_path)


def test_default_skill_paths_use_settings_selected_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    settings._fast_agent_home = str(tmp_path / "selected-home")
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "ambient-home"))

    paths = default_skill_paths(settings, cwd=tmp_path)

    assert paths[0] == (tmp_path / "selected-home" / "skills").resolve()
    assert (tmp_path / "ambient-home" / "skills").resolve() not in paths


def test_default_skill_paths_skip_home_when_no_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    settings._fast_agent_no_home = True
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "ambient-home"))

    paths = default_skill_paths(settings, cwd=tmp_path)

    assert (tmp_path / ".fast-agent" / "skills").resolve() not in paths
    assert (tmp_path / "ambient-home" / "skills").resolve() not in paths
    assert paths == [
        (tmp_path / ".agents" / "skills").resolve(),
        (tmp_path / ".claude" / "skills").resolve(),
    ]
