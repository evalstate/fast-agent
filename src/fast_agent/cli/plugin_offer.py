"""One-time offer to install the recommended plugin bundle on a first interactive run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fast_agent.cli.update_check import CHECK_ERRORS, should_check_now
from fast_agent.config import find_config_in_directory
from fast_agent.home import PREFERRED_CONFIG_FILENAME
from fast_agent.plugins.bundles import RECOMMENDED_BUNDLE, bundle_members, install_plugin_bundle
from fast_agent.plugins.marketplace import parse_marketplace_plugins
from fast_agent.plugins.operations import list_local_plugins

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

OFFER_MARKER_FILENAME = ".plugin_bundle_offer"
INSTALL_LATER_HINT = f"fast-agent plugins add --bundle {RECOMMENDED_BUNDLE} --global"
_ANSWERED = "answered"
_RETRY = "retry"


def offer_recommended_plugins(
    *,
    global_home: Path,
    plugin_roots: Sequence[Path],
    marketplace_url: str,
    fetch_payload: Callable[[str], Any],
    confirm: Callable[[str], bool],
    echo: Callable[[str], None],
    now: float | None = None,
) -> None:
    """Ask once, globally, whether to install the recommended bundle.

    Users who already have plugins are never asked. A failed marketplace fetch is
    retried at most daily; once the question is answered it is never asked again.
    """
    marker = global_home / OFFER_MARKER_FILENAME
    if marker.exists() and (
        marker.read_text(encoding="utf-8").strip() == _ANSWERED
        or not should_check_now(marker, now=now)
    ):
        return
    if any(list_local_plugins(destination_root=root) for root in plugin_roots):
        return

    try:
        plugins = parse_marketplace_plugins(
            fetch_payload(marketplace_url), source_url=marketplace_url
        )
    except CHECK_ERRORS:
        _write_marker(marker, _RETRY)
        return
    members = bundle_members(plugins, RECOMMENDED_BUNDLE)
    if not members:
        _write_marker(marker, _RETRY)
        return

    listing = "\n".join(
        f"  • {plugin.name}" + (f" — {plugin.description}" if plugin.description else "")
        for plugin in members
    )
    accepted = confirm(f"Install the recommended fast-agent plugins?\n{listing}\n")
    _write_marker(marker, _ANSWERED)
    if not accepted:
        echo(f"Skipped. Install them later with: {INSTALL_LATER_HINT}")
        return

    config_path = find_config_in_directory(global_home) or global_home / PREFERRED_CONFIG_FILENAME
    try:
        results = install_plugin_bundle(
            plugins,
            RECOMMENDED_BUNDLE,
            destination_root=global_home / "plugins",
            config_path=config_path,
        )
    except Exception as exc:
        echo(f"Could not install recommended plugins: {exc}\nRetry with: {INSTALL_LATER_HINT}")
        return
    echo(f"Installed {', '.join(result.name for result in results)} (enabled in {config_path})")


def _write_marker(marker: Path, state: str) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(state + "\n", encoding="utf-8")
