"""Model alias config layering and targeted mutation helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from fast_agent.config import (
    Settings,
    deep_merge,
    find_fastagent_config_files,
    find_project_config_file,
    load_layered_model_settings,
    load_yaml_mapping,
    resolve_config_search_root,
    resolve_environment_config_file,
)
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.model_resolution import parse_model_alias_token

ModelAliasWriteTarget = Literal["env", "project"]


@dataclass(frozen=True)
class ModelAliasConfigPaths:
    """Resolved paths used by model alias mutation helpers."""

    project_read_path: Path | None
    project_write_path: Path
    env_path: Path
    secrets_path: Path | None


@dataclass(frozen=True)
class ModelAliasChange:
    """Single mutation preview entry."""

    key_path: str
    old: str | None
    new: str | None


@dataclass(frozen=True)
class ModelAliasMutationResult:
    """Result payload for set/unset mutations."""

    target: ModelAliasWriteTarget
    target_path: Path
    dry_run: bool
    applied: bool
    changes: tuple[ModelAliasChange, ...]


class ModelAliasConfigService:
    """Read/write service for ``model_aliases`` config subtree operations."""

    def __init__(
        self,
        *,
        cwd: Path | None = None,
        env_dir: str | Path | None = None,
    ) -> None:
        self._cwd = (cwd or Path.cwd()).resolve()
        self._env_dir = env_dir if env_dir is not None else os.getenv("ENVIRONMENT_DIR")
        self.paths = _discover_paths(start_path=self._cwd, env_dir=self._env_dir)

    def list_aliases(self) -> dict[str, dict[str, str]]:
        """Return effective aliases from layered project/env config (+ secrets overlay)."""
        layered_model_settings = load_layered_model_settings(
            start_path=self._cwd,
            env_dir=self._env_dir,
        )

        if self.paths.secrets_path and self.paths.secrets_path.exists():
            secrets_payload = load_yaml_mapping(self.paths.secrets_path)
            layered_model_settings = _merge_model_settings(layered_model_settings, secrets_payload)

        aliases_payload = layered_model_settings.get("model_aliases", {})
        validated = Settings(model_aliases=aliases_payload)
        return validated.model_aliases

    def set_alias(
        self,
        token: str,
        model_spec: str,
        *,
        target: ModelAliasWriteTarget = "env",
        dry_run: bool = False,
    ) -> ModelAliasMutationResult:
        """Set a model alias token at the selected write target."""
        try:
            namespace, key = parse_model_alias_token(token)
        except ModelConfigError as exc:
            raise ValueError(exc.details) from exc

        normalized_model = model_spec.strip()
        if not normalized_model:
            raise ValueError("model-spec must be a non-empty string")

        # Reuse existing Settings validator semantics for alias values.
        Settings(model_aliases={namespace: {key: normalized_model}})

        target_path = self._resolve_target_path(target)
        document = _load_round_trip_yaml(target_path)
        old_value = _extract_alias_value(document, namespace, key)

        _set_alias_value(document, namespace, key, normalized_model)
        new_value = normalized_model
        changed = old_value != new_value

        if changed and not dry_run:
            _atomic_write_round_trip_yaml(document, target_path)

        return ModelAliasMutationResult(
            target=target,
            target_path=target_path,
            dry_run=dry_run,
            applied=changed and (not dry_run),
            changes=(
                ModelAliasChange(
                    key_path=f"model_aliases.{namespace}.{key}",
                    old=old_value,
                    new=new_value,
                ),
            ),
        )

    def unset_alias(
        self,
        token: str,
        *,
        target: ModelAliasWriteTarget = "env",
        dry_run: bool = False,
    ) -> ModelAliasMutationResult:
        """Unset a model alias token from the selected write target."""
        try:
            namespace, key = parse_model_alias_token(token)
        except ModelConfigError as exc:
            raise ValueError(exc.details) from exc

        target_path = self._resolve_target_path(target)
        document = _load_round_trip_yaml(target_path)
        old_value = _extract_alias_value(document, namespace, key)

        _unset_alias_value(document, namespace, key)
        changed = old_value is not None

        if changed and not dry_run:
            _atomic_write_round_trip_yaml(document, target_path)

        return ModelAliasMutationResult(
            target=target,
            target_path=target_path,
            dry_run=dry_run,
            applied=changed and (not dry_run),
            changes=(
                ModelAliasChange(
                    key_path=f"model_aliases.{namespace}.{key}",
                    old=old_value,
                    new=None,
                ),
            ),
        )

    def _resolve_target_path(self, target: ModelAliasWriteTarget) -> Path:
        if target == "env":
            return self.paths.env_path
        if target == "project":
            return self.paths.project_write_path
        raise ValueError("target must be 'env' or 'project'")


def _discover_paths(*, start_path: Path, env_dir: str | Path | None) -> ModelAliasConfigPaths:
    project_read_path = find_project_config_file(start_path)
    project_write_path = project_read_path or (start_path / "fastagent.config.yaml")
    env_path = resolve_environment_config_file(start_path, env_dir=env_dir)

    search_root = resolve_config_search_root(start_path, env_dir=env_dir)
    _, secrets_path = find_fastagent_config_files(search_root)

    return ModelAliasConfigPaths(
        project_read_path=project_read_path,
        project_write_path=project_write_path,
        env_path=env_path,
        secrets_path=secrets_path,
    )


def _merge_model_settings(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    if "default_model" in overlay:
        merged["default_model"] = overlay["default_model"]

    if "model_aliases" in overlay:
        overlay_aliases = overlay["model_aliases"]
        base_aliases = merged.get("model_aliases")
        if isinstance(base_aliases, dict) and isinstance(overlay_aliases, dict):
            merged["model_aliases"] = deep_merge(base_aliases, overlay_aliases)
        else:
            merged["model_aliases"] = overlay_aliases

    return merged


_ROUND_TRIP_YAML = YAML()
_ROUND_TRIP_YAML.preserve_quotes = True


def _load_round_trip_yaml(path: Path) -> CommentedMap:
    if not path.exists():
        return CommentedMap()

    with open(path, "r", encoding="utf-8") as handle:
        payload = _ROUND_TRIP_YAML.load(handle)

    if payload is None:
        return CommentedMap()
    if isinstance(payload, CommentedMap):
        return payload
    if isinstance(payload, dict):
        return CommentedMap(payload)

    raise ValueError(f"Top-level YAML at {path} must be a mapping")


def _extract_alias_value(document: CommentedMap, namespace: str, key: str) -> str | None:
    aliases = document.get("model_aliases")
    if not isinstance(aliases, dict):
        return None

    namespace_entries = aliases.get(namespace)
    if not isinstance(namespace_entries, dict):
        return None

    value = namespace_entries.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _set_alias_value(document: CommentedMap, namespace: str, key: str, model_spec: str) -> None:
    aliases = document.get("model_aliases")
    if not isinstance(aliases, dict):
        aliases = CommentedMap()
        document["model_aliases"] = aliases

    namespace_entries = aliases.get(namespace)
    if not isinstance(namespace_entries, dict):
        namespace_entries = CommentedMap()
        aliases[namespace] = namespace_entries

    namespace_entries[key] = model_spec


def _unset_alias_value(document: CommentedMap, namespace: str, key: str) -> None:
    aliases = document.get("model_aliases")
    if not isinstance(aliases, dict):
        return

    namespace_entries = aliases.get(namespace)
    if not isinstance(namespace_entries, dict):
        return

    if key in namespace_entries:
        del namespace_entries[key]

    if len(namespace_entries) == 0 and namespace in aliases:
        del aliases[namespace]

    if len(aliases) == 0 and "model_aliases" in document:
        del document["model_aliases"]


def _atomic_write_round_trip_yaml(document: CommentedMap, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            _ROUND_TRIP_YAML.dump(document, handle)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)

        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
