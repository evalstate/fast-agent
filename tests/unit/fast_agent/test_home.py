from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.core.exceptions import ConfigFileError, FastAgentError
from fast_agent.home import (
    AmbiguousConfigFilesError,
    AmbiguousSecretsFilesError,
    FastAgentHome,
    build_child_environment,
    discover_config_files,
    find_config_in_directory,
    find_secrets_in_directory,
    resolve_fast_agent_home,
)

if TYPE_CHECKING:
    from pathlib import Path


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x: y\n", encoding="utf-8")
    return path


def test_resolve_home_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_AGENT_HOME", "from-fast-agent-home")

    home = resolve_fast_agent_home(cwd=tmp_path, cli_override="from-cli")

    assert home == FastAgentHome((tmp_path / "from-cli").resolve(), "cli")


def test_resolve_home_uses_fast_agent_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAST_AGENT_HOME", "from-fast-agent-home")

    home = resolve_fast_agent_home(cwd=tmp_path)

    assert home == FastAgentHome((tmp_path / "from-fast-agent-home").resolve(), "FAST_AGENT_HOME")


def test_resolve_home_uses_default_when_fast_agent_home_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)

    default_home = resolve_fast_agent_home(cwd=tmp_path)
    assert default_home == FastAgentHome((tmp_path / ".fast-agent").resolve(), "default")


def test_no_home_disables_home_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAST_AGENT_HOME", "ignored")

    assert resolve_fast_agent_home(cwd=tmp_path, no_home=True) is None


def test_discovers_preferred_config_in_home(tmp_path: Path) -> None:
    home_path = tmp_path / ".fast-agent"
    config = touch(home_path / "fast-agent.yaml")
    secrets = touch(home_path / "fast-agent.secrets.yaml")
    home = FastAgentHome(home_path.resolve(), "default")

    result = discover_config_files(cwd=tmp_path, home=home)

    assert result.home == home
    assert result.config_path == config.resolve()
    assert result.secrets_path == secrets.resolve()


def test_discovers_preferred_config_in_cwd_without_home(tmp_path: Path) -> None:
    config = touch(tmp_path / "fast-agent.yaml")

    result = discover_config_files(cwd=tmp_path, home=None)

    assert result.config_path == config.resolve()


def test_discovers_legacy_and_transitional_aliases(tmp_path: Path) -> None:
    legacy_home = tmp_path / "legacy-home"
    transitional_home = tmp_path / "transitional-home"
    legacy_config = touch(legacy_home / "fastagent.config.yaml")
    transitional_config = touch(transitional_home / "fast-agent.config.yaml")

    legacy_result = discover_config_files(
        cwd=tmp_path,
        home=FastAgentHome(legacy_home.resolve(), "cli"),
    )
    transitional_result = discover_config_files(
        cwd=tmp_path,
        home=FastAgentHome(transitional_home.resolve(), "cli"),
    )

    assert legacy_result.config_path == legacy_config.resolve()
    assert transitional_result.config_path == transitional_config.resolve()


def test_parent_directories_are_ignored(tmp_path: Path) -> None:
    parent_config = touch(tmp_path / "fast-agent.yaml")
    nested = tmp_path / "app" / "nested"
    nested.mkdir(parents=True)

    result = discover_config_files(cwd=nested, home=None)

    assert result.config_path is None
    assert result.secrets_path is None
    assert parent_config.exists()


def test_same_directory_config_ambiguity_errors(tmp_path: Path) -> None:
    touch(tmp_path / "fast-agent.yaml")
    touch(tmp_path / "fastagent.config.yaml")

    with pytest.raises(AmbiguousConfigFilesError) as exc_info:
        find_config_in_directory(tmp_path)

    assert isinstance(exc_info.value, ConfigFileError)
    assert isinstance(exc_info.value, FastAgentError)
    assert exc_info.value.directory == tmp_path.resolve()
    assert [path.name for path in exc_info.value.candidates] == [
        "fast-agent.yaml",
        "fastagent.config.yaml",
    ]


def test_same_directory_secrets_ambiguity_errors(tmp_path: Path) -> None:
    touch(tmp_path / "fast-agent.secrets.yaml")
    touch(tmp_path / "fastagent.secrets.yaml")

    with pytest.raises(AmbiguousSecretsFilesError) as exc_info:
        find_secrets_in_directory(tmp_path)

    assert exc_info.value.directory == tmp_path.resolve()
    assert [path.name for path in exc_info.value.candidates] == [
        "fast-agent.secrets.yaml",
        "fastagent.secrets.yaml",
    ]


def test_explicit_config_path_is_exact_and_uses_same_directory_secrets(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    config_dir = tmp_path / "config-dir"
    config = touch(config_dir / "custom.yaml")
    secrets = touch(config_dir / "fast-agent.secrets.yaml")
    touch(tmp_path / "fast-agent.secrets.yaml")

    result = discover_config_files(cwd=cwd, home=None, explicit_config_path=config)

    assert result.config_path == config.resolve()
    assert result.secrets_path == secrets.resolve()


def test_config_selected_secrets_come_only_from_config_directory(tmp_path: Path) -> None:
    home_path = tmp_path / ".fast-agent"
    config = touch(home_path / "fast-agent.yaml")
    cwd_secrets = touch(tmp_path / "fast-agent.secrets.yaml")

    result = discover_config_files(
        cwd=tmp_path,
        home=FastAgentHome(home_path.resolve(), "default"),
    )

    assert result.config_path == config.resolve()
    assert result.secrets_path is None
    assert cwd_secrets.exists()


def test_secrets_only_searches_home_then_cwd(tmp_path: Path) -> None:
    home_path = tmp_path / ".fast-agent"
    home_secrets = touch(home_path / "fast-agent.secrets.yaml")
    touch(tmp_path / "fast-agent.secrets.yaml")

    home_result = discover_config_files(
        cwd=tmp_path,
        home=FastAgentHome(home_path.resolve(), "default"),
    )
    cwd_result = discover_config_files(cwd=tmp_path, home=None)

    assert home_result.config_path is None
    assert home_result.secrets_path == home_secrets.resolve()
    assert cwd_result.secrets_path == (tmp_path / "fast-agent.secrets.yaml").resolve()


def test_home_and_cwd_same_directory_is_searched_once(tmp_path: Path) -> None:
    config = touch(tmp_path / "fast-agent.yaml")
    home = FastAgentHome(tmp_path.resolve(), "cli")

    result = discover_config_files(cwd=tmp_path, home=home)

    assert result.config_path == config.resolve()


def test_child_environment_exports_runtime_home_and_legacy_alias(tmp_path: Path) -> None:
    env = build_child_environment(
        active_home=tmp_path / ".fast-agent",
        base={"PATH": "/bin"},
        overrides={"EXTRA": "1"},
    )

    assert env["PATH"] == "/bin"
    assert env["EXTRA"] == "1"
    assert env["FAST_AGENT_RUNTIME_HOME"] == str((tmp_path / ".fast-agent").resolve())
    assert env["FAST_AGENT_HOME"] == str((tmp_path / ".fast-agent").resolve())


def test_no_home_child_environment_strips_runtime_home_aliases(tmp_path: Path) -> None:
    env = build_child_environment(
        active_home=tmp_path / ".fast-agent",
        no_home=True,
        base={
            "PATH": "/bin",
            "FAST_AGENT_RUNTIME_HOME": "inherited",
            "FAST_AGENT_HOME": "inherited",
        },
        overrides={
            "FAST_AGENT_RUNTIME_HOME": "override",
            "FAST_AGENT_HOME": "override",
        },
    )

    assert env == {"PATH": "/bin"}
