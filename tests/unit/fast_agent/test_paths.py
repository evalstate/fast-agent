from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import fast_agent.config as config_module
from fast_agent.config import Settings
from fast_agent.paths import default_skill_paths, resolve_environment_dir, resolve_environment_paths

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_environment_dir_uses_fast_agent_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("FAST_AGENT_HOME", str(home))
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)

    assert resolve_environment_dir(Settings(), cwd=tmp_path) == home.resolve()


def test_resolve_environment_dir_override_wins_over_fast_agent_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "from-home"))

    assert (
        resolve_environment_dir(Settings(), cwd=tmp_path, override=tmp_path / "from-cli")
        == (tmp_path / "from-cli").resolve()
    )


def test_resolve_environment_dir_settings_environment_dir_wins_over_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "from-home"))

    assert (
        resolve_environment_dir(Settings(environment_dir="from-settings"), cwd=tmp_path)
        == (tmp_path / "from-settings").resolve()
    )


def test_resolve_environment_dir_settings_environment_dir_wins_over_cached_home(
    tmp_path: Path,
) -> None:
    settings = Settings(environment_dir="configured-home")
    settings._fast_agent_home = str(tmp_path / ".fast-agent")

    assert resolve_environment_dir(settings, cwd=tmp_path) == (
        tmp_path / "configured-home"
    ).resolve()


def test_resolve_environment_dir_uses_settings_selected_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    settings._fast_agent_home = str(tmp_path / "selected-home")
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "ambient-home"))

    assert resolve_environment_dir(settings, cwd=tmp_path) == (tmp_path / "selected-home").resolve()


def test_resolve_environment_paths_uses_get_settings_env_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_home = tmp_path / "selected-home"
    ambient_home = tmp_path / "ambient-home"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAST_AGENT_HOME", str(ambient_home))
    config_module._settings = None

    try:
        settings = config_module.get_settings(env_dir=selected_home)

        assert resolve_environment_paths(settings, cwd=tmp_path).root == selected_home.resolve()
    finally:
        config_module._settings = None


def test_resolve_environment_paths_rejects_noenv_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    settings._fast_agent_noenv = True
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "ambient-home"))

    with pytest.raises(ValueError, match="fast-agent home is disabled"):
        resolve_environment_paths(settings, cwd=tmp_path)


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


def test_default_skill_paths_skip_home_when_noenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    settings._fast_agent_noenv = True
    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "ambient-home"))

    paths = default_skill_paths(settings, cwd=tmp_path)

    assert (tmp_path / ".fast-agent" / "skills").resolve() not in paths
    assert (tmp_path / "ambient-home" / "skills").resolve() not in paths
    assert paths == [
        (tmp_path / ".agents" / "skills").resolve(),
        (tmp_path / ".claude" / "skills").resolve(),
    ]
