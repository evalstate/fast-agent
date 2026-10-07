"""Plugin update detection from published marketplace metadata.

Each marketplace entry publishes ``path_oid``, the git tree id of the plugin
directory, which installs record as ``installed_path_oid``. Comparing the two
needs one fetch per marketplace and no git operations, so it is cheap enough
for the startup update check. ``plugins update`` remains the authoritative check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_agent.plugins.marketplace import parse_marketplace_plugins
from fast_agent.plugins.operations import list_local_plugins
from fast_agent.utils.versions import requirement_satisfied

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from fast_agent.plugins.models import InstalledPluginSource, MarketplacePlugin


@dataclass(frozen=True, slots=True)
class PluginUpdateNotice:
    name: str
    installed_version: str | None
    available_version: str | None
    requires_fast_agent: str | None
    compatible: bool


def find_plugin_updates(
    plugin_roots: Iterable[Path],
    *,
    fetch_payload: Callable[[str], Any],
    fast_agent_version: str | None,
) -> list[PluginUpdateNotice]:
    """Return installed marketplace plugins whose published contents differ.

    Plugins installed in several roots are reported once (first root wins).
    """
    installed: dict[str, tuple[str | None, str, InstalledPluginSource]] = {}
    for root in plugin_roots:
        for plugin in list_local_plugins(destination_root=root):
            if plugin.source is None or plugin.source.source_url is None:
                continue
            version = plugin.manifest.version if plugin.manifest is not None else None
            installed.setdefault(plugin.name, (version, plugin.source.source_url, plugin.source))

    published: dict[str, dict[tuple[str, str, str | None], MarketplacePlugin]] = {}
    notices: list[PluginUpdateNotice] = []
    for name, (version, source_url, source) in installed.items():
        if source_url not in published:
            published[source_url] = {
                (entry.repo_url, entry.repo_path, entry.repo_ref): entry
                for entry in parse_marketplace_plugins(
                    fetch_payload(source_url), source_url=source_url
                )
            }
        entry = published[source_url].get((source.repo_url, source.repo_path, source.repo_ref))
        if (
            entry is None
            or entry.path_oid is None
            or source.installed_path_oid in (None, entry.path_oid)
        ):
            continue
        notices.append(
            PluginUpdateNotice(
                name=name,
                installed_version=version,
                available_version=entry.version,
                requires_fast_agent=entry.requires_fast_agent,
                compatible=entry.requires_fast_agent is None
                or fast_agent_version is None
                or requirement_satisfied(entry.requires_fast_agent, fast_agent_version),
            )
        )
    return notices
