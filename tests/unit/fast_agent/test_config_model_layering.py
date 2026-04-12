from __future__ import annotations

import os
from pathlib import Path

import yaml

import fast_agent.config as config_module
from fast_agent.config import get_settings


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


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


def test_get_settings_falls_back_to_parent_config_as_legacy_lookup(tmp_path: Path) -> None:
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

        assert settings.default_model == "legacy-default"
        assert settings._config_file == str(workspace / "fastagent.config.yaml")
    finally:
        os.chdir(previous_cwd)
        config_module._settings = previous_settings
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_get_settings_keeps_secrets_last_with_env_cwd_legacy_discovery(tmp_path: Path) -> None:
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

        assert settings.default_model == "secret-default"
        assert settings._secrets_file == str(env_dir / "fastagent.secrets.yaml")
    finally:
        os.chdir(previous_cwd)
        config_module._settings = previous_settings
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
