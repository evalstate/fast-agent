from __future__ import annotations

import yaml

from fast_agent.llm.model_reference_config import ModelReferenceConfigService


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


def test_set_reference_dry_run_does_not_mutate_target_file(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    service = ModelReferenceConfigService(start_path=workspace, env_dir=env_dir)

    result = service.set_reference(
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


def test_set_reference_writes_env_target_and_creates_config_file(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    service = ModelReferenceConfigService(start_path=workspace, env_dir=env_dir)

    result = service.set_reference("$system.fast", "claude-haiku-4-5", target="env")

    assert result.applied is True
    assert result.target_path == env_dir / "fastagent.config.yaml"
    saved = _read_yaml(result.target_path)
    assert saved["model_references"]["system"]["fast"] == "claude-haiku-4-5"


def test_set_reference_preserves_existing_yaml_comments(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)
    config_path = env_dir / "fastagent.config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        (
            "# top comment\n"
            "model_references:\n"
            "  # shared references\n"
            "  system:\n"
            "    # keep this note\n"
            "    fast: claude-sonnet-4-5\n"
        ),
        encoding="utf-8",
    )

    service = ModelReferenceConfigService(start_path=workspace, env_dir=env_dir)

    result = service.set_reference("$system.fast", "claude-haiku-4-5", target="env")

    assert result.applied is True
    updated_text = config_path.read_text(encoding="utf-8")
    assert "# top comment" in updated_text
    assert "# shared references" in updated_text
    assert "# keep this note" in updated_text
    assert "fast: claude-haiku-4-5" in updated_text


def test_unset_reference_writes_project_target(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    project_config = workspace / "fastagent.config.yaml"
    _write_yaml(
        project_config,
        {
            "model_references": {
                "system": {
                    "fast": "claude-haiku-4-5",
                    "code": "claude-sonnet-4-5",
                }
            }
        },
    )

    service = ModelReferenceConfigService(start_path=workspace, env_dir=workspace / ".fast-agent")

    result = service.unset_reference("$system.fast", target="project")

    assert result.applied is True
    saved = _read_yaml(project_config)
    assert "fast" not in saved["model_references"]["system"]
    assert saved["model_references"]["system"]["code"] == "claude-sonnet-4-5"


def test_list_references_uses_project_env_and_secrets_layering(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    _write_yaml(
        workspace / "fastagent.config.yaml",
        {
            "model_references": {
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
            "model_references": {
                "system": {
                    "fast": "env-fast",
                }
            }
        },
    )
    _write_yaml(
        env_dir / "fastagent.secrets.yaml",
        {
            "model_references": {
                "system": {
                    "code": "secret-code",
                }
            }
        },
    )

    service = ModelReferenceConfigService(start_path=workspace, env_dir=env_dir)
    references = service.list_references()

    assert references["system"]["fast"] == "env-fast"
    assert references["system"]["code"] == "secret-code"
