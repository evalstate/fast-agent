"""Best-effort git provenance capture for persisted sessions."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse


@dataclass(frozen=True, slots=True)
class GitMetadata:
    cwd: str
    repository_root: str
    commit: str
    captured_at: datetime
    remote_url: str | None = None
    github_repository: str | None = None
    branch: str | None = None
    dirty: bool = False


def capture_git_metadata(cwd: Path) -> GitMetadata | None:
    """Return git metadata for ``cwd`` when it is inside a git repository."""
    resolved_cwd = cwd.expanduser().resolve()
    repository_root = _git_output(resolved_cwd, "rev-parse", "--show-toplevel")
    commit = _git_output(resolved_cwd, "rev-parse", "HEAD")
    if repository_root is None or commit is None:
        return None

    remote_url = _sanitize_remote_url(_git_output(resolved_cwd, "remote", "get-url", "origin"))
    return GitMetadata(
        cwd=str(resolved_cwd),
        repository_root=str(Path(repository_root).expanduser().resolve()),
        commit=commit,
        captured_at=datetime.now(timezone.utc),
        remote_url=remote_url,
        github_repository=_github_repository(remote_url),
        branch=_git_output(resolved_cwd, "branch", "--show-current") or None,
        dirty=bool(_git_output(resolved_cwd, "status", "--porcelain")),
    )


def _git_output(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(cwd), *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _sanitize_remote_url(remote_url: str | None) -> str | None:
    if remote_url is None:
        return None
    parsed = urlparse(remote_url)
    if parsed.scheme and parsed.netloc and "@" in parsed.netloc:
        netloc = parsed.netloc.rsplit("@", 1)[1]
        return urlunparse(parsed._replace(netloc=netloc))
    return remote_url


def _github_repository(remote_url: str | None) -> str | None:
    if remote_url is None:
        return None

    owner_repo: str | None = None
    parsed = urlparse(remote_url)
    host = parsed.hostname or ""
    if host == "github.com":
        owner_repo = parsed.path.lstrip("/")
    elif remote_url.startswith("git@github.com:"):
        owner_repo = remote_url.split(":", 1)[1]

    if owner_repo is None:
        return None
    owner_repo = owner_repo.removesuffix(".git").strip("/")
    parts = owner_repo.split("/")
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])
