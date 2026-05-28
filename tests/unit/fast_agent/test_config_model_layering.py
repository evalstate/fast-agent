from __future__ import annotations

import os
from pathlib import Path

import yaml

import fast_agent.config as config_module
from fast_agent.config import Settings, get_settings, update_global_settings


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_get_settings_preserves_manually_installed_global_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_yaml(workspace / ".fast-agent" / "fast-agent.yaml", {"default_model": "disk-default"})
    monkeypatch.chdir(workspace)

    previous_settings = config_module._settings
    try:
        manual_settings = Settings(
            default_model="manual-default",
            environment_dir=str(tmp_path / ".manual-fast-agent"),
        )
        update_global_settings(manual_settings)

        settings = get_settings()

        assert settings is manual_settings
        assert settings.default_model == "manual-default"
        assert settings.environment_dir == str(tmp_path / ".manual-fast-agent")
    finally:
        config_module._settings = previous_settings


def test_get_settings_prefers_env_config_over_cwd_and_legacy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "child"
    env_dir = nested / ".fast-agent"
    workspace.mkdir(parents=True)
    nested.mkdir()

    _write_yaml(workspace / "fastagent.config.yaml", {"default_model": "legacy-default"})
    _write_yaml(nested / "fastagent.config.yaml", {"default_model": "cwd-default"})
    _write_yaml(env_dir / "fastagent.config.yaml", {"default_model": "env-default"})

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    previous_settings = config_module._settings
    try:
        os.chdir(nested)
        os.environ.pop("ENVIRONMENT_DIR", None)
        config_module._settings = None

        settings = get_settings()

        assert settings.default_model == "env-default"
        assert settings._config_file == str(env_dir / "fastagent.config.yaml")
    finally:
        os.chdir(previous_cwd)
        config_module._settings = previous_settings
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_get_settings_env_dir_argument_wins_over_fast_agent_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    fast_agent_home = workspace / ".from-fast-agent-home"
    cli_env = workspace / ".from-cli-env"
    workspace.mkdir(parents=True)

    _write_yaml(fast_agent_home / "fast-agent.yaml", {"default_model": "wrong-home"})
    _write_yaml(cli_env / "fast-agent.yaml", {"default_model": "right-home"})
    monkeypatch.setenv("FAST_AGENT_HOME", str(fast_agent_home))
    monkeypatch.chdir(workspace)

    previous_settings = config_module._settings
    try:
        config_module._settings = None

        settings = get_settings(env_dir=cli_env)

        assert settings.default_model == "right-home"
        assert settings._config_file == str(cli_env / "fast-agent.yaml")
    finally:
        config_module._settings = previous_settings


def test_get_settings_recomputes_when_env_dir_argument_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    default_env = workspace / ".fast-agent"
    cli_env = workspace / ".custom-env"
    workspace.mkdir(parents=True)

    _write_yaml(default_env / "fast-agent.yaml", {"default_model": "default-env"})
    _write_yaml(cli_env / "fast-agent.yaml", {"default_model": "cli-env"})
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    monkeypatch.chdir(workspace)

    previous_settings = config_module._settings
    try:
        config_module._settings = None

        cached_settings = get_settings()
        selected_settings = get_settings(env_dir=cli_env)

        assert cached_settings.default_model == "default-env"
        assert selected_settings.default_model == "cli-env"
        assert selected_settings._config_file == str(cli_env / "fast-agent.yaml")
    finally:
        config_module._settings = previous_settings


def test_get_settings_recomputes_when_env_dir_override_removed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    default_env = workspace / ".fast-agent"
    cli_env = workspace / ".custom-env"
    workspace.mkdir(parents=True)

    _write_yaml(default_env / "fast-agent.yaml", {"default_model": "default-env"})
    _write_yaml(cli_env / "fast-agent.yaml", {"default_model": "cli-env"})
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    monkeypatch.chdir(workspace)

    previous_settings = config_module._settings
    try:
        config_module._settings = None

        cli_settings = get_settings(env_dir=cli_env)
        default_settings = get_settings()

        assert cli_settings.default_model == "cli-env"
        assert default_settings.default_model == "default-env"
        assert default_settings._config_file == str(default_env / "fast-agent.yaml")
    finally:
        config_module._settings = previous_settings


def test_get_settings_recomputes_when_noenv_argument_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    _write_yaml(env_dir / "fast-agent.yaml", {"default_model": "env-default"})
    _write_yaml(workspace / "fast-agent.yaml", {"default_model": "cwd-default"})
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    monkeypatch.chdir(workspace)

    previous_settings = config_module._settings
    try:
        config_module._settings = None

        cached_settings = get_settings()
        selected_settings = get_settings(noenv=True)

        assert cached_settings.default_model == "env-default"
        assert selected_settings.default_model == "cwd-default"
        assert selected_settings._fast_agent_home is None
        assert selected_settings._fast_agent_noenv is True
    finally:
        config_module._settings = previous_settings


def test_get_settings_prefers_cwd_config_when_env_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "child"
    workspace.mkdir(parents=True)
    nested.mkdir()

    _write_yaml(workspace / "fastagent.config.yaml", {"default_model": "legacy-default"})
    _write_yaml(nested / "fastagent.config.yaml", {"default_model": "cwd-default"})

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    previous_settings = config_module._settings
    try:
        os.chdir(nested)
        os.environ.pop("ENVIRONMENT_DIR", None)
        config_module._settings = None

        settings = get_settings()

        assert settings.default_model == "cwd-default"
        assert settings._config_file == str(nested / "fastagent.config.yaml")
    finally:
        os.chdir(previous_cwd)
        config_module._settings = previous_settings
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_get_settings_ignores_parent_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "child" / "grandchild"
    workspace.mkdir(parents=True)
    nested.mkdir(parents=True)

    _write_yaml(workspace / "fastagent.config.yaml", {"default_model": "legacy-default"})

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    previous_settings = config_module._settings
    try:
        os.chdir(nested)
        os.environ.pop("ENVIRONMENT_DIR", None)
        config_module._settings = None

        settings = get_settings()

        assert settings.default_model is None
        assert settings._config_file is None
    finally:
        os.chdir(previous_cwd)
        config_module._settings = previous_settings
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_get_settings_pairs_secrets_with_selected_config_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "child"
    env_dir = nested / ".fast-agent"
    workspace.mkdir(parents=True)
    nested.mkdir()

    _write_yaml(workspace / "fastagent.config.yaml", {"default_model": "legacy-default"})
    _write_yaml(nested / "fastagent.config.yaml", {"default_model": "cwd-default"})
    _write_yaml(env_dir / "fastagent.secrets.yaml", {"default_model": "secret-default"})

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    previous_settings = config_module._settings
    try:
        os.chdir(nested)
        os.environ.pop("ENVIRONMENT_DIR", None)
        config_module._settings = None

        settings = get_settings()

        assert settings.default_model == "cwd-default"
        assert settings._secrets_file is None
    finally:
        os.chdir(previous_cwd)
        config_module._settings = previous_settings
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
