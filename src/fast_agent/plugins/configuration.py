"""Plugin settings helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from fast_agent.marketplace import registry_urls
from fast_agent.marketplace.source_urls import normalize_marketplace_url
from fast_agent.plugins.models import DEFAULT_PLUGIN_MARKETPLACE_URL, DEFAULT_PLUGIN_REGISTRIES

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.config import Settings


def get_marketplace_url(settings: "Settings | None" = None) -> str:
    url = settings.plugins.marketplace_url if settings is not None else None
    if not url and settings is not None and settings.plugins.marketplace_urls:
        url = settings.plugins.marketplace_urls[0]
    return normalize_marketplace_url(url or DEFAULT_PLUGIN_MARKETPLACE_URL)


def get_manager_directory(settings: "Settings | None" = None, *, cwd: Path | None = None) -> Path:
    from fast_agent.paths import resolve_environment_paths

    return resolve_environment_paths(settings, cwd=cwd).plugins


def resolve_registries(settings: "Settings | None" = None) -> list[str]:
    configured = settings.plugins.marketplace_urls if settings is not None else None
    active = settings.plugins.marketplace_url if settings is not None else None
    return registry_urls.resolve_registry_urls(
        configured,
        default_urls=DEFAULT_PLUGIN_REGISTRIES,
        active_url=active,
    )


def enable_plugin_in_config(path: Path, name: str) -> None:
    data = _read_config(path)
    plugins = data.get("plugins")
    if not isinstance(plugins, dict):
        plugins = {}
    enabled = plugins.get("enabled")
    if not isinstance(enabled, list):
        enabled = []
    if name not in enabled:
        enabled.append(name)
    plugins["enabled"] = enabled
    data["plugins"] = plugins
    _write_config(path, data)


def disable_plugin_in_config(path: Path, name: str) -> None:
    data = _read_config(path)
    plugins = data.get("plugins")
    if not isinstance(plugins, dict):
        return
    enabled = plugins.get("enabled")
    if isinstance(enabled, list):
        plugins["enabled"] = [entry for entry in enabled if entry != name]
    data["plugins"] = plugins
    _write_config(path, data)


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _write_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
