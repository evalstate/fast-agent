"""Model reference config layering and targeted mutation helpers."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from fast_agent.config import (
    Settings,
    deep_merge,
    load_layered_model_settings,
    load_yaml_mapping,
)
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.model_resolution import parse_model_reference_token
from fast_agent.home import (
    PREFERRED_CONFIG_FILENAME,
    discover_config_files,
    find_config_in_directory,
    resolve_fast_agent_home,
)
from fast_agent.paths import resolve_settings_start_path

ModelReferenceWriteTarget = Literal["env", "project"]


@dataclass(frozen=True)
class ModelReferenceConfigPaths:
    """Resolved paths used by model reference mutation helpers."""

    project_read_path: Path | None
    project_write_path: Path
    env_path: Path
    secrets_path: Path | None


@dataclass(frozen=True)
class ModelReferenceChange:
    """Single mutation preview entry."""

    key_path: str
    old: str | None
    new: str | None


@dataclass(frozen=True)
class ModelReferenceMutationResult:
    """Result payload for set/unset mutations."""

    target: ModelReferenceWriteTarget
    target_path: Path
    dry_run: bool
    applied: bool
    changes: tuple[ModelReferenceChange, ...]


class ModelReferenceConfigService:
    """Read/write service for ``model_references`` config subtree operations."""

    def __init__(
        self,
        *,
        start_path: Path | None = None,
        env_dir: str | Path | None = None,
        project_write_path: Path | None = None,
    ) -> None:
        self._start_path = (start_path or Path.cwd()).resolve()
        self._env_dir = env_dir
        self.paths = _discover_paths(
            start_path=self._start_path,
            env_dir=self._env_dir,
            project_write_path=project_write_path,
        )

    def load_effective_model_settings(self) -> dict[str, Any]:
        """Return layered model settings merged with any secrets overlay."""
        layered_model_settings = load_layered_model_settings(
            start_path=self._start_path,
            env_dir=self._env_dir,
        )

        if self.paths.secrets_path and self.paths.secrets_path.exists():
            secrets_payload = load_yaml_mapping(self.paths.secrets_path)
            layered_model_settings = _merge_model_settings(layered_model_settings, secrets_payload)

        return layered_model_settings

    def list_references(self) -> dict[str, dict[str, str]]:
        """Return effective references from layered project/env config (+ secrets overlay)."""
        layered_model_settings = self.load_effective_model_settings()
        references_payload = layered_model_settings.get("model_references", {})
        validated = Settings(model_references=references_payload)
        return validated.model_references

    def list_references_tolerant(self) -> dict[str, dict[str, str]]:
        """Return only valid, non-empty reference entries from effective settings."""
        references_payload = self.load_effective_model_settings().get("model_references", {})
        return _extract_valid_references(references_payload)

    def set_reference(
        self,
        token: str,
        model_spec: str,
        *,
        target: ModelReferenceWriteTarget = "env",
        dry_run: bool = False,
    ) -> ModelReferenceMutationResult:
        """Set a model reference token at the selected write target."""
        try:
            namespace, key = parse_model_reference_token(token)
        except ModelConfigError as exc:
            raise ValueError(exc.details) from exc

        normalized_model = model_spec.strip()
        if not normalized_model:
            raise ValueError("model-spec must be a non-empty string")

        # Reuse existing Settings validator semantics for reference values.
        Settings(model_references={namespace: {key: normalized_model}})

        target_path = self._resolve_target_path(target)
        document = _load_round_trip_yaml(target_path)
        old_value = _extract_reference_value(document, namespace, key)

        _set_reference_value(document, namespace, key, normalized_model)
        new_value = normalized_model
        changed = old_value != new_value

        if changed and not dry_run:
            _atomic_write_round_trip_yaml(document, target_path)

        return ModelReferenceMutationResult(
            target=target,
            target_path=target_path,
            dry_run=dry_run,
            applied=changed and (not dry_run),
            changes=(
                ModelReferenceChange(
                    key_path=f"model_references.{namespace}.{key}",
                    old=old_value,
                    new=new_value,
                ),
            ),
        )

    def unset_reference(
        self,
        token: str,
        *,
        target: ModelReferenceWriteTarget = "env",
        dry_run: bool = False,
    ) -> ModelReferenceMutationResult:
        """Unset a model reference token from the selected write target."""
        try:
            namespace, key = parse_model_reference_token(token)
        except ModelConfigError as exc:
            raise ValueError(exc.details) from exc

        target_path = self._resolve_target_path(target)
        document = _load_round_trip_yaml(target_path)
        old_value = _extract_reference_value(document, namespace, key)

        _unset_reference_value(document, namespace, key)
        changed = old_value is not None

        if changed and not dry_run:
            _atomic_write_round_trip_yaml(document, target_path)

        return ModelReferenceMutationResult(
            target=target,
            target_path=target_path,
            dry_run=dry_run,
            applied=changed and (not dry_run),
            changes=(
                ModelReferenceChange(
                    key_path=f"model_references.{namespace}.{key}",
                    old=old_value,
                    new=None,
                ),
            ),
        )

    def _resolve_target_path(self, target: ModelReferenceWriteTarget) -> Path:
        target_paths: dict[ModelReferenceWriteTarget, Path] = {
            "env": self.paths.env_path,
            "project": self.paths.project_write_path,
        }
        if target in target_paths:
            return target_paths[target]
        raise ValueError("target must be 'env' or 'project'")


def resolve_model_reference_start_path(
    *,
    settings: Settings | None = None,
    fallback_path: Path | None = None,
) -> Path:
    """Resolve the deterministic base path for model reference config discovery."""
    return resolve_settings_start_path(settings, fallback_path=fallback_path)


def _discover_paths(
    *,
    start_path: Path,
    env_dir: str | Path | None,
    project_write_path: Path | None = None,
) -> ModelReferenceConfigPaths:
    resolved_project_write_path = (
        project_write_path.expanduser().resolve() if project_write_path is not None else None
    )
    project_read_path = find_config_in_directory(start_path)
    if resolved_project_write_path is not None and resolved_project_write_path.exists():
        project_read_path = resolved_project_write_path
    resolved_project_path = resolved_project_write_path or project_read_path or (
        start_path / PREFERRED_CONFIG_FILENAME
    )
    home = resolve_fast_agent_home(cwd=start_path, cli_override=env_dir)
    env_root = home.path if home is not None else start_path
    env_path = find_config_in_directory(env_root) or (env_root / PREFERRED_CONFIG_FILENAME)
    secrets_path = discover_config_files(cwd=start_path, home=home).secrets_path

    return ModelReferenceConfigPaths(
        project_read_path=project_read_path,
        project_write_path=resolved_project_path,
        env_path=env_path,
        secrets_path=secrets_path,
    )


def _merge_model_settings(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    if "default_model" in overlay:
        merged["default_model"] = overlay["default_model"]

    if "model_references" in overlay:
        overlay_references = overlay["model_references"]
        base_references = merged.get("model_references")
        if isinstance(base_references, dict) and isinstance(overlay_references, dict):
            merged["model_references"] = deep_merge(base_references, overlay_references)
        else:
            merged["model_references"] = overlay_references

    return merged


def _extract_valid_references(references_payload: Any) -> dict[str, dict[str, str]]:
    if not isinstance(references_payload, dict):
        return {}

    valid_references: dict[str, dict[str, str]] = {}
    for namespace, entries in references_payload.items():
        if not isinstance(namespace, str) or not isinstance(entries, dict):
            continue

        normalized_entries: dict[str, str] = {}
        for key, raw_value in entries.items():
            if not isinstance(key, str) or not isinstance(raw_value, str):
                continue
            if not _is_valid_reference_name(namespace, key):
                continue
            model_value = raw_value.strip()
            if not model_value:
                continue
            normalized_entries[key] = model_value

        if normalized_entries:
            valid_references[namespace] = normalized_entries

    return valid_references


def _is_valid_reference_name(namespace: str, key: str) -> bool:
    try:
        parse_model_reference_token(f"${namespace}.{key}")
    except ModelConfigError:
        return False
    return True


_ROUND_TRIP_YAML = YAML()
_ROUND_TRIP_YAML.preserve_quotes = True


def _load_round_trip_yaml(path: Path) -> CommentedMap:
    if not path.exists():
        return CommentedMap()

    with path.open("r", encoding="utf-8") as handle:
        payload = _ROUND_TRIP_YAML.load(handle)

    if payload is None:
        return CommentedMap()
    if isinstance(payload, CommentedMap):
        return payload
    if isinstance(payload, dict):
        return CommentedMap(payload)

    raise ValueError(f"Top-level YAML at {path} must be a mapping")


def _extract_reference_value(document: CommentedMap, namespace: str, key: str) -> str | None:
    references = document.get("model_references")
    if not isinstance(references, dict):
        return None

    namespace_entries = references.get(namespace)
    if not isinstance(namespace_entries, dict):
        return None

    value = namespace_entries.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _set_reference_value(document: CommentedMap, namespace: str, key: str, model_spec: str) -> None:
    references = document.get("model_references")
    if not isinstance(references, dict):
        references = CommentedMap()
        document["model_references"] = references

    namespace_entries = references.get(namespace)
    if not isinstance(namespace_entries, dict):
        namespace_entries = CommentedMap()
        references[namespace] = namespace_entries

    namespace_entries[key] = model_spec


def _unset_reference_value(document: CommentedMap, namespace: str, key: str) -> None:
    references = document.get("model_references")
    if not isinstance(references, dict):
        return

    namespace_entries = references.get(namespace)
    if not isinstance(namespace_entries, dict):
        return

    if key in namespace_entries:
        del namespace_entries[key]

    if not namespace_entries and namespace in references:
        del references[namespace]

    if not references and "model_references" in document:
        del document["model_references"]


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

        temp_path.replace(path)
    finally:
        if temp_path and temp_path.exists():
            with suppress(Exception):
                temp_path.unlink()
