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


def enabled_plugins_by_scope(settings: "Settings | None" = None) -> tuple[list[str], list[str]]:
    """Return ``(home_enabled, project_enabled)`` for the active config.

    Mirrors the load scoping used at startup: home plugins load first, then
    project plugins override them. Used by tooling that needs to know which
    scope a given enabled plugin name loads from.
    """
    from fast_agent.config import _enabled_plugin_sources, get_settings

    if settings is None:
        settings = get_settings()
    sources = _enabled_plugin_sources(settings)
    return list(sources.home), list(sources.project)


def installed_plugin_roots(
    settings: "Settings | None" = None,
    *,
    project_plugins: "Path | None" = None,
) -> list[tuple[str, "Path"]]:
    """Return ``(scope, plugins_dir)`` roots that are installed to, in load order.

    ``project`` first (overrides), then ``global`` (FAST_AGENT_HOME / ~/.fast-agent)
    when distinct from the project root.
    """
    from pathlib import Path

    from fast_agent.utils.text import strip_to_none

    if project_plugins is None:
        project_plugins = get_manager_directory(settings)
    roots: list[tuple[str, Path]] = [("project", project_plugins)]
    global_home = (
        strip_to_none(settings._fast_agent_global_plugin_home) if settings is not None else None
    )
    if global_home is None:
        return roots
    global_plugins = Path(global_home).expanduser() / "plugins"
    if global_plugins.resolve() != project_plugins.resolve():
        roots.append(("global", global_plugins))
    return roots


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
