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


def test_get_settings_layers_model_settings_project_then_env(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    _write_yaml(
        workspace / "fastagent.config.yaml",
        {
            "default_model": "project-default",
            "model_aliases": {
                "system": {
                    "fast": "project-fast",
                    "code": "project-code",
                },
                "project": {
                    "only": "project-only",
                },
            },
        },
    )
    _write_yaml(
        env_dir / "fastagent.config.yaml",
        {
            "default_model": "env-default",
            "model_aliases": {
                "system": {
                    "fast": "env-fast",
                },
                "env": {
                    "only": "env-only",
                },
            },
        },
    )

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    previous_settings = config_module._settings
    try:
        os.chdir(workspace)
        os.environ["ENVIRONMENT_DIR"] = str(env_dir)
        config_module._settings = None

        settings = get_settings()

        assert settings.default_model == "env-default"
        assert settings.model_aliases["system"]["fast"] == "env-fast"
        assert settings.model_aliases["system"]["code"] == "project-code"
        assert settings.model_aliases["project"]["only"] == "project-only"
        assert settings.model_aliases["env"]["only"] == "env-only"
    finally:
        os.chdir(previous_cwd)
        config_module._settings = previous_settings
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_get_settings_keeps_secrets_last_for_model_settings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    _write_yaml(
        workspace / "fastagent.config.yaml",
        {
            "default_model": "project-default",
            "model_aliases": {
                "system": {
                    "fast": "project-fast",
                }
            },
        },
    )
    _write_yaml(
        env_dir / "fastagent.config.yaml",
        {
            "default_model": "env-default",
            "model_aliases": {
                "system": {
                    "fast": "env-fast",
                }
            },
        },
    )
    _write_yaml(
        env_dir / "fastagent.secrets.yaml",
        {
            "default_model": "secret-default",
            "model_aliases": {
                "system": {
                    "fast": "secret-fast",
                }
            },
        },
    )

    previous_cwd = Path.cwd()
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    previous_settings = config_module._settings
    try:
        os.chdir(workspace)
        os.environ["ENVIRONMENT_DIR"] = str(env_dir)
        config_module._settings = None

        settings = get_settings()

        assert settings.default_model == "secret-default"
        assert settings.model_aliases["system"]["fast"] == "secret-fast"
    finally:
        os.chdir(previous_cwd)
        config_module._settings = previous_settings
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
