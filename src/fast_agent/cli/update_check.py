"""Lightweight CLI update checker for fast-agent."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from fast_agent.core.exceptions import FastAgentError
from fast_agent.paths import resolve_home_dir
from fast_agent.utils.text import strip_str_to_none
from fast_agent.utils.versions import compare_releases, release_tuple

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fast_agent.plugins.updates import PluginUpdateNotice

logger = logging.getLogger(__name__)

CHECK_ERRORS = (
    FastAgentError,
    OSError,
    ValueError,
    TimeoutError,
    HTTPError,
    URLError,
    json.JSONDecodeError,
)

PACKAGE_NAME = "fast-agent-mcp"
DEFAULT_UPDATE_COMMAND = "uv tool install -U fast-agent-mcp"
PLUGIN_UPDATE_COMMAND = "fast-agent plugins update all --yes"
DEFAULT_TIMEOUT_SECONDS = 1.5
DEFAULT_INTERVAL_SECONDS = 24 * 3600
UPDATE_CHECK_MARKER_FILENAME = ".check_for_update_done"
_PRERELEASE_OR_DEV_PATTERN = re.compile(
    r"(?:(?<=\d)(?:a|b|rc|dev|alpha|beta|pre|preview)\d*"
    r"|[._-](?:a|b|rc|dev|alpha|beta|pre|preview)\d*)",
    re.IGNORECASE,
)


def get_installed_version(package_name: str = PACKAGE_NAME) -> str | None:
    """Return the installed package version, or ``None`` when unavailable."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def is_prerelease_or_dev(version: str) -> bool:
    """Return True for dev or prerelease versions that should skip checks."""
    return _PRERELEASE_OR_DEV_PATTERN.search(version) is not None


def _resolve_home_root(
    home: Path | None,
    *,
    cwd: Path | None = None,
) -> Path:
    base = cwd or Path.cwd()
    return resolve_home_dir(cwd=base, override=home)


def resolve_update_check_marker_path(
    home: Path | None,
    *,
    cwd: Path | None = None,
) -> Path | None:
    """Return the marker path used to rate-limit update checks."""
    home_root = _resolve_home_root(home, cwd=cwd)
    if not home_root.is_dir():
        return None
    return home_root / UPDATE_CHECK_MARKER_FILENAME


def should_run_update_check(*, disabled: bool) -> bool:
    """Return True when the CLI should attempt an update check."""
    return not disabled


def should_check_now(
    marker_path: Path,
    *,
    now: float | None = None,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
) -> bool:
    """Return True when the marker file is missing or older than the interval."""
    if not marker_path.exists():
        return True
    current_time = time.time() if now is None else now
    return (current_time - marker_path.stat().st_mtime) >= interval_seconds


def mark_check_complete(marker_path: Path) -> None:
    """Touch the marker file, creating parent directories as needed."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.touch()


def is_newer_version(latest_version: str, current_version: str) -> bool:
    """Return True when ``latest_version`` is newer than ``current_version``."""
    latest_tuple = release_tuple(latest_version)
    current_tuple = release_tuple(current_version)
    if latest_tuple is None or current_tuple is None:
        return False
    return compare_releases(latest_tuple, current_tuple) > 0


def _fetch_latest_version_from_pypi(
    package_name: str = PACKAGE_NAME,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    url = f"https://pypi.org/pypi/{package_name}/json"
    with urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    info = payload.get("info")
    if not isinstance(info, dict):
        raise ValueError("PyPI response is missing package info")
    version = strip_str_to_none(info.get("version"))
    if version is None:
        raise ValueError("PyPI response is missing package version")
    return version


def format_update_notice(
    *,
    latest_version: str,
    update_command: str = DEFAULT_UPDATE_COMMAND,
) -> str:
    """Format a rich-markup notice for CLI and TUI startup display."""
    return (
        "fast-agent [cyan]"
        f"{latest_version}[/cyan] is available "
        f" "
        f"[dim][bold]({update_command})[/bold][/dim]"
    )


def _resolve_latest_version(
    *,
    package_name: str,
    timeout_seconds: float,
    fetch_latest_version: Callable[[], str] | None,
) -> str:
    if fetch_latest_version is not None:
        return fetch_latest_version()
    return _fetch_latest_version_from_pypi(
        package_name,
        timeout_seconds=timeout_seconds,
    )


def default_plugin_roots(home: Path | None, *, cwd: Path | None = None) -> list[Path]:
    """Project and global plugin directories for the startup plugin update check."""
    from fast_agent.config import resolve_global_plugin_home_path

    roots = [_resolve_home_root(home, cwd=cwd) / "plugins"]
    try:
        global_home = resolve_global_plugin_home_path(
            fast_agent_home=os.getenv("FAST_AGENT_HOME"),
            home=Path.home(),
            cwd=cwd or Path.cwd(),
        )
    except RuntimeError:
        global_home = None
    if global_home is not None:
        roots.append(global_home / "plugins")
    return list(dict.fromkeys(root.resolve() for root in roots))


def fetch_marketplace_payload(url: str, *, timeout_seconds: float) -> Any:
    from fast_agent.marketplace.fetch import load_local_marketplace_payload

    if (local_payload := load_local_marketplace_payload(url)) is not None:
        return local_payload
    with urlopen(url, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _find_plugin_updates(
    plugin_roots: Sequence[Path],
    *,
    fast_agent_version: str | None,
    fetch_marketplace: Callable[[str], Any],
) -> list[PluginUpdateNotice]:
    """Best effort: a failed plugin check must not hide the fast-agent notice."""
    from fast_agent.plugins.updates import find_plugin_updates

    try:
        return find_plugin_updates(
            plugin_roots,
            fetch_payload=fetch_marketplace,
            fast_agent_version=fast_agent_version,
        )
    except CHECK_ERRORS:
        logger.debug("Skipping plugin update notice after check failure.", exc_info=True)
        return []


def format_plugin_update_notice(updates: Sequence[PluginUpdateNotice]) -> str:
    """Format a rich-markup notice listing plugin updates."""
    from rich.markup import escape

    def describe(update: PluginUpdateNotice) -> str:
        versions = " → ".join(
            escape(version)
            for version in (update.installed_version, update.available_version)
            if version is not None
        )
        label = f"[cyan]{escape(update.name)}[/cyan] {versions}".rstrip()
        if update.compatible:
            return label
        return f"{label} [dim](needs fast-agent {escape(update.requires_fast_agent or '')})[/dim]"

    notice = "Plugin updates: " + ", ".join(describe(update) for update in updates)
    if any(update.compatible for update in updates):
        notice += f" [dim][bold]({PLUGIN_UPDATE_COMMAND})[/bold][/dim]"
    return notice


def _check_for_update_notice(
    *,
    home: Path | None,
    package_name: str = PACKAGE_NAME,
    current_version: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    now: float | None = None,
    fetch_latest_version: Callable[[], str] | None = None,
    plugin_roots: Sequence[Path] = (),
    fetch_marketplace: Callable[[str], Any] | None = None,
) -> str | None:
    installed_version = current_version or get_installed_version(package_name)
    check_release = installed_version is not None and not is_prerelease_or_dev(installed_version)
    if not check_release and not plugin_roots:
        return None

    marker_path = resolve_update_check_marker_path(home)
    if marker_path is not None and not should_check_now(
        marker_path,
        now=now,
        interval_seconds=interval_seconds,
    ):
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        latest_future = (
            pool.submit(
                _resolve_latest_version,
                package_name=package_name,
                timeout_seconds=timeout_seconds,
                fetch_latest_version=fetch_latest_version,
            )
            if check_release
            else None
        )
        plugins_future = pool.submit(
            _find_plugin_updates,
            plugin_roots,
            fast_agent_version=installed_version,
            fetch_marketplace=fetch_marketplace
            or partial(fetch_marketplace_payload, timeout_seconds=timeout_seconds),
        )
        latest_version = latest_future.result() if latest_future is not None else None
        plugin_updates = plugins_future.result()

    if marker_path is not None:
        mark_check_complete(marker_path)
    notices: list[str] = []
    if (
        latest_version is not None
        and installed_version is not None
        and not is_prerelease_or_dev(latest_version)
        and is_newer_version(latest_version, installed_version)
    ):
        notices.append(format_update_notice(latest_version=latest_version))
    if plugin_updates:
        notices.append(format_plugin_update_notice(plugin_updates))
    return "\n".join(notices) or None


def check_for_update_notice(
    *,
    home: Path | None,
    package_name: str = PACKAGE_NAME,
    current_version: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    now: float | None = None,
    fetch_latest_version: Callable[[], str] | None = None,
    plugin_roots: Sequence[Path] = (),
    fetch_marketplace: Callable[[str], Any] | None = None,
) -> str | None:
    """Return a formatted update notice, swallowing network/cache errors.

    ``plugin_roots`` opts in to checking installed plugins against their marketplaces.
    """
    try:
        return _check_for_update_notice(
            home=home,
            package_name=package_name,
            current_version=current_version,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
            now=now,
            fetch_latest_version=fetch_latest_version,
            plugin_roots=plugin_roots,
            fetch_marketplace=fetch_marketplace,
        )
    except CHECK_ERRORS:
        logger.debug("Skipping update notice after check failure.", exc_info=True)
        return None
