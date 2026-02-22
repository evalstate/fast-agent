from __future__ import annotations

import yaml

from fast_agent.llm.model_alias_config import ModelAliasConfigService


def _write_yaml(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if isinstance(loaded, dict):
        return loaded
    return {}


def test_set_alias_dry_run_does_not_mutate_target_file(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    service = ModelAliasConfigService(cwd=workspace, env_dir=env_dir)

    result = service.set_alias(
        "$system.fast",
        "claude-haiku-4-5",
        target="env",
        dry_run=True,
    )

    assert result.target_path == env_dir / "fastagent.config.yaml"
    assert result.applied is False
    assert result.dry_run is True
    assert result.changes[0].old is None
    assert result.changes[0].new == "claude-haiku-4-5"
    assert result.target_path.exists() is False


def test_set_alias_writes_env_target_and_creates_config_file(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    service = ModelAliasConfigService(cwd=workspace, env_dir=env_dir)

    result = service.set_alias("$system.fast", "claude-haiku-4-5", target="env")

    assert result.applied is True
    assert result.target_path == env_dir / "fastagent.config.yaml"
    saved = _read_yaml(result.target_path)
    assert saved["model_aliases"]["system"]["fast"] == "claude-haiku-4-5"


def test_unset_alias_writes_project_target(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    project_config = workspace / "fastagent.config.yaml"
    _write_yaml(
        project_config,
        {
            "model_aliases": {
                "system": {
                    "fast": "claude-haiku-4-5",
                    "code": "claude-sonnet-4-5",
                }
            }
        },
    )

    service = ModelAliasConfigService(cwd=workspace, env_dir=workspace / ".fast-agent")

    result = service.unset_alias("$system.fast", target="project")

    assert result.applied is True
    saved = _read_yaml(project_config)
    assert "fast" not in saved["model_aliases"]["system"]
    assert saved["model_aliases"]["system"]["code"] == "claude-sonnet-4-5"


def test_list_aliases_uses_project_env_and_secrets_layering(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    _write_yaml(
        workspace / "fastagent.config.yaml",
        {
            "model_aliases": {
                "system": {
                    "fast": "project-fast",
                    "code": "project-code",
                }
            }
        },
    )
    _write_yaml(
        env_dir / "fastagent.config.yaml",
        {
            "model_aliases": {
                "system": {
                    "fast": "env-fast",
                }
            }
        },
    )
    _write_yaml(
        env_dir / "fastagent.secrets.yaml",
        {
            "model_aliases": {
                "system": {
                    "code": "secret-code",
                }
            }
        },
    )

    service = ModelAliasConfigService(cwd=workspace, env_dir=env_dir)
    aliases = service.list_aliases()

    assert aliases["system"]["fast"] == "env-fast"
    assert aliases["system"]["code"] == "secret-code"
