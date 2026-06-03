"""Plugin manifest parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fast_agent.command_actions.config import parse_plugin_command_action_specs
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.plugins.models import PLUGIN_MANIFEST_FILENAME, PluginManifest
from fast_agent.utils.text import strip_to_none


class _PluginManifestModel(BaseModel):
    schema_version: int = 1
    name: str
    version: str | None = None
    description: str | None = None
    commands: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("plugin name must not be empty")
        return cleaned

    @field_validator("version", "description")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        return strip_to_none(value)


def load_plugin_manifest(plugin_dir: Path) -> PluginManifest:
    manifest_path = plugin_dir / PLUGIN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{PLUGIN_MANIFEST_FILENAME} not found in {plugin_dir}")

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise AgentConfigError(f"Plugin manifest must be a mapping: {manifest_path}")

    model = _PluginManifestModel.model_validate(data)
    if model.schema_version != 1:
        raise AgentConfigError(
            f"Unsupported plugin schema_version: {model.schema_version}",
            f"Manifest: {manifest_path}",
        )

    commands = parse_plugin_command_action_specs(model.commands, source=str(manifest_path)) or {}
    commands = {
        name: spec.__class__(
            name=spec.name,
            description=spec.description,
            handler=_resolve_handler(spec.handler, plugin_dir),
            input_hint=spec.input_hint,
            key=spec.key,
        )
        for name, spec in commands.items()
    }
    return PluginManifest(
        schema_version=model.schema_version,
        name=model.name,
        version=model.version,
        description=model.description,
        commands=commands,
        path=manifest_path,
    )


def _resolve_handler(handler: str, plugin_dir: Path) -> str:
    module, separator, func = handler.rpartition(":")
    if not separator:
        return handler
    module_path = Path(module)
    if module_path.is_absolute():
        raise AgentConfigError(f"Plugin handler must be relative to plugin root: {handler}")
    plugin_root = plugin_dir.resolve()
    resolved = (plugin_root / module_path).resolve()
    try:
        resolved.relative_to(plugin_root)
    except ValueError as exc:
        raise AgentConfigError(f"Plugin handler escapes plugin root: {handler}") from exc
    return f"{resolved.as_posix()}:{func}"
