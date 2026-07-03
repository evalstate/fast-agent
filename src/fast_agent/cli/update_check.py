"""Lightweight CLI update checker for fast-agent."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from fast_agent.core.exceptions import FastAgentError
from fast_agent.paths import resolve_home_dir
from fast_agent.utils.text import strip_str_to_none

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

PACKAGE_NAME = "fast-agent-mcp"
DEFAULT_UPDATE_COMMAND = "uv tool install -U fast-agent-mcp"
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


def _parse_release_tuple(version: str) -> tuple[int, ...] | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer_version(latest_version: str, current_version: str) -> bool:
    """Return True when ``latest_version`` is newer than ``current_version``."""
    latest_tuple = _parse_release_tuple(latest_version)
    current_tuple = _parse_release_tuple(current_version)
    if latest_tuple is None or current_tuple is None:
        return False

    width = max(len(latest_tuple), len(current_tuple))
    normalized_latest = latest_tuple + (0,) * (width - len(latest_tuple))
    normalized_current = current_tuple + (0,) * (width - len(current_tuple))
    return normalized_latest > normalized_current


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


def _check_for_update_notice(
    *,
    home: Path | None,
    package_name: str = PACKAGE_NAME,
    current_version: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    now: float | None = None,
    fetch_latest_version: Callable[[], str] | None = None,
) -> str | None:
    installed_version = current_version or get_installed_version(package_name)
    if installed_version is None or is_prerelease_or_dev(installed_version):
        return None

    marker_path = resolve_update_check_marker_path(home)
    if marker_path is not None and not should_check_now(
        marker_path,
        now=now,
        interval_seconds=interval_seconds,
    ):
        return None

    latest_version = _resolve_latest_version(
        package_name=package_name,
        timeout_seconds=timeout_seconds,
        fetch_latest_version=fetch_latest_version,
    )
    if marker_path is not None:
        mark_check_complete(marker_path)
    if is_prerelease_or_dev(latest_version):
        return None
    if not is_newer_version(latest_version, installed_version):
        return None

    return format_update_notice(
        latest_version=latest_version,
    )


def check_for_update_notice(
    *,
    home: Path | None,
    package_name: str = PACKAGE_NAME,
    current_version: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    now: float | None = None,
    fetch_latest_version: Callable[[], str] | None = None,
) -> str | None:
    """Return a formatted update notice, swallowing network/cache errors."""
    try:
        return _check_for_update_notice(
            home=home,
            package_name=package_name,
            current_version=current_version,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
            now=now,
            fetch_latest_version=fetch_latest_version,
        )
    except (
        FastAgentError,
        OSError,
        ValueError,
        TimeoutError,
        HTTPError,
        URLError,
        json.JSONDecodeError,
    ):
        logger.debug("Skipping update notice after check failure.", exc_info=True)
        return None
