"""Shared marketplace source parsing and git helper utilities."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypeVar
from urllib.parse import urlparse
from uuid import uuid4

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping, Sequence

EntryT = TypeVar("EntryT")


@dataclass(frozen=True)
class ParsedInstalledSourceFields:
    source_origin: Literal["remote", "local"]
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None
    installed_commit: str | None
    installed_path_oid: str | None
    installed_revision: str
    installed_at: str
    content_fingerprint: str


def normalize_marketplace_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            org, repo, _, ref = parts[:4]
            file_path = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{file_path}"
    return url


def candidate_marketplace_urls(url: str) -> list[str]:
    normalized = url.strip()
    if not normalized:
        return []

    parsed = urlparse(normalized)
    if parsed.scheme in {"file", ""} and parsed.netloc == "":
        path = Path(parsed.path).expanduser()
        if path.exists() and path.is_dir():
            claude_plugin = path / ".claude-plugin" / "marketplace.json"
            if claude_plugin.exists():
                return [claude_plugin.as_posix()]
            fallback = path / "marketplace.json"
            if fallback.exists():
                return [fallback.as_posix()]
        return [normalized]

    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            org, repo = parts[:2]
            if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
                ref = parts[3]
                base_path = "/".join(parts[4:])
                return _github_marketplace_candidates(org, repo, ref, base_path)
            if len(parts) == 2:
                return [
                    *_github_marketplace_candidates(org, repo, "main", ""),
                    *_github_marketplace_candidates(org, repo, "master", ""),
                ]

    return [normalized]


def _github_marketplace_candidates(org: str, repo: str, ref: str, base_path: str) -> list[str]:
    suffixes = _marketplace_path_candidates(base_path)
    return [
        f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{suffix}"
        for suffix in suffixes
    ]


def _marketplace_path_candidates(base_path: str) -> list[str]:
    cleaned = base_path.strip().strip("/")
    if not cleaned:
        return [".claude-plugin/marketplace.json", "marketplace.json"]

    path = PurePosixPath(cleaned)
    if path.name.lower() == "marketplace.json":
        return [str(path)]
    if path.name == ".claude-plugin":
        return [str(path / "marketplace.json")]

    return [
        str(path / ".claude-plugin" / "marketplace.json"),
        str(path / "marketplace.json"),
    ]


def parse_github_url(url: str | None) -> tuple[str, str | None, str] | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
            org, repo, _, ref = parts[:4]
            file_path = "/".join(parts[4:])
            return f"https://github.com/{org}/{repo}", ref, file_path
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            org, repo, ref = parts[:3]
            file_path = "/".join(parts[3:])
            return f"https://github.com/{org}/{repo}", ref, file_path
    return None


def parse_ls_remote_commit(output: str) -> str | None:
    fallback: str | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        commit = parts[0].strip()
        if not commit:
            continue
        ref = parts[1].strip() if len(parts) > 1 else ""
        if ref.endswith("^{}"):
            return commit
        if fallback is None:
            fallback = commit
    return fallback


def load_local_marketplace_payload(url: str) -> Any | None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        return read_json_file(path)
    if parsed.scheme in {"http", "https"}:
        return None
    candidate = Path(url).expanduser()
    if candidate.exists():
        return read_json_file(candidate)
    return None


def read_json_file(path: Path) -> Any:
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def resolve_local_repo(repo_url: str) -> Path | None:
    parsed = urlparse(repo_url)
    if parsed.scheme == "file":
        repo_path = Path(parsed.path)
    elif parsed.scheme in {"http", "https", "ssh"}:
        return None
    else:
        repo_path = Path(repo_url)

    repo_path = repo_path.expanduser()
    if not repo_path.is_absolute():
        repo_path = repo_path.resolve()
    if repo_path.exists():
        return repo_path
    return None


def derive_local_repo_root(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    if parsed.scheme in {"http", "https", "ssh"}:
        return None

    if parsed.scheme == "file":
        path = Path(parsed.path)
    else:
        path = Path(source_url)

    path = path.expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if not path.exists():
        return None

    if path.is_file() and path.name == "marketplace.json":
        if path.parent.name == ".claude-plugin":
            repo_root = path.parent.parent
        else:
            repo_root = path.parent
        if repo_root.exists():
            return str(repo_root)

    if path.is_dir():
        return str(path)

    return None


def resolve_git_commit(repo_root: Path, revision: str | None) -> str | None:
    rev = revision or "HEAD"
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", f"{rev}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    values = result.stdout.strip().splitlines()
    if not values:
        return None
    commit = values[0].strip()
    return commit or None


def resolve_git_path_oid(repo_root: Path, commit: str, repo_path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", f"{commit}:{repo_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    values = result.stdout.strip().splitlines()
    if not values:
        return None
    path_oid = values[0].strip()
    return path_oid or None


def run_git(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git command failed: {' '.join(args)}\n{stderr}")


def resolve_source_path_oid(
    *,
    repo_url: str,
    repo_ref: str | None,
    repo_path: str,
    commit: str,
    path_cache: "MutableMapping[tuple[str, str | None, str, str], tuple[str | None, Any, str | None]]",
    resolve_local_repo_fn: "Callable[[str], Path | None]" = resolve_local_repo,
    resolve_git_path_oid_fn: "Callable[[Path, str, str], str | None]" = resolve_git_path_oid,
) -> tuple[str | None, Any, str | None]:
    cache_key = (repo_url, repo_ref, repo_path, commit)
    cached = path_cache.get(cache_key)
    if cached is not None:
        return cached

    local_repo = resolve_local_repo_fn(repo_url)
    if local_repo is not None:
        path_oid = resolve_git_path_oid_fn(local_repo, commit, repo_path)
        if path_oid is None:
            resolved = (None, "source_path_missing", f"path missing at revision {commit}: {repo_path}")
            path_cache[cache_key] = resolved
            return resolved
        resolved = (path_oid, None, None)
        path_cache[cache_key] = resolved
        return resolved

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        clone_args = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
        ]
        if repo_ref:
            clone_args.extend(["--branch", repo_ref])
        clone_args.extend([repo_url, str(tmp_path)])

        result = subprocess.run(clone_args, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            if repo_ref and "Remote branch" in stderr and "not found" in stderr:
                resolved = (None, "source_ref_missing", f"ref not found: {repo_ref}")
            else:
                resolved = (None, "source_unreachable", stderr or "unable to reach source")
            path_cache[cache_key] = resolved
            return resolved

        path_oid = resolve_git_path_oid_fn(tmp_path, commit, repo_path)
        if path_oid is None:
            resolved = (None, "source_path_missing", f"path missing at revision {commit}: {repo_path}")
            path_cache[cache_key] = resolved
            return resolved

    resolved = (path_oid, None, None)
    path_cache[cache_key] = resolved
    return resolved


def parse_installed_source_fields(
    payload: "Mapping[str, Any]",
    *,
    expected_schema_version: int,
    normalize_repo_path: "Callable[[str], str | None]",
) -> ParsedInstalledSourceFields:
    schema_version = payload.get("schema_version")
    if schema_version != expected_schema_version:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    installed_via = payload.get("installed_via")
    if not isinstance(installed_via, str) or installed_via.strip() != "marketplace":
        raise ValueError("installed_via must be 'marketplace'")

    source_origin_raw = payload.get("source_origin")
    if source_origin_raw not in {"remote", "local"}:
        raise ValueError("source_origin must be 'remote' or 'local'")

    repo_url = payload.get("repo_url")
    if not isinstance(repo_url, str) or not repo_url.strip():
        raise ValueError("repo_url is required")

    repo_ref_value = payload.get("repo_ref")
    repo_ref: str | None
    if repo_ref_value is None:
        repo_ref = None
    elif isinstance(repo_ref_value, str):
        repo_ref = repo_ref_value.strip() or None
    else:
        raise ValueError("repo_ref must be a string or null")

    repo_path_raw = payload.get("repo_path")
    if not isinstance(repo_path_raw, str):
        raise ValueError("repo_path is required")
    repo_path = normalize_repo_path(repo_path_raw)
    if not repo_path:
        raise ValueError("repo_path is invalid")

    source_url_value = payload.get("source_url")
    source_url: str | None
    if source_url_value is None:
        source_url = None
    elif isinstance(source_url_value, str):
        source_url = source_url_value.strip() or None
    else:
        raise ValueError("source_url must be a string or null")

    installed_commit_value = payload.get("installed_commit")
    installed_commit: str | None
    if installed_commit_value is None:
        installed_commit = None
    elif isinstance(installed_commit_value, str) and installed_commit_value.strip():
        installed_commit = installed_commit_value.strip()
    else:
        raise ValueError("installed_commit must be a non-empty string or null")

    installed_path_oid_value = payload.get("installed_path_oid")
    installed_path_oid: str | None
    if installed_path_oid_value is None:
        installed_path_oid = None
    elif isinstance(installed_path_oid_value, str) and installed_path_oid_value.strip():
        installed_path_oid = installed_path_oid_value.strip()
    else:
        raise ValueError("installed_path_oid must be a non-empty string or null")

    installed_revision = payload.get("installed_revision")
    if not isinstance(installed_revision, str) or not installed_revision.strip():
        raise ValueError("installed_revision is required")

    installed_at = payload.get("installed_at")
    if not isinstance(installed_at, str) or not installed_at.strip():
        raise ValueError("installed_at is required")

    content_fingerprint = payload.get("content_fingerprint")
    if not isinstance(content_fingerprint, str) or not content_fingerprint.startswith("sha256:"):
        raise ValueError("content_fingerprint must be a sha256 fingerprint")

    return ParsedInstalledSourceFields(
        source_origin=source_origin_raw,
        repo_url=repo_url.strip(),
        repo_ref=repo_ref,
        repo_path=repo_path,
        source_url=source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision.strip(),
        installed_at=installed_at.strip(),
        content_fingerprint=content_fingerprint,
    )


def normalize_marketplace_payload(
    data: Any,
    info: Any,
    *,
    extract_entries: "Callable[[Any], list[dict[str, Any]]]",
) -> dict[str, list[dict[str, Any]]]:
    entries = extract_entries(data)
    context = getattr(info, "context", None) or {}
    source_url = context.get("source_url")
    repo_url = context.get("repo_url")
    repo_ref = context.get("repo_ref")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if source_url and "source_url" not in entry:
            entry["source_url"] = source_url
        if repo_url and "repo_url" not in entry and "repo" not in entry:
            entry["repo_url"] = repo_url
        if repo_ref and "repo_ref" not in entry and "ref" not in entry:
            entry["repo_ref"] = repo_ref
    return {"entries": entries}


async def fetch_marketplace_entries_with_source(
    url: str,
    *,
    candidate_urls: "Callable[[str], Sequence[str]]",
    normalize_url: "Callable[[str], str]",
    load_local_payload: "Callable[[str], Any | None]",
    parse_payload: "Callable[[Any, str | None], list[EntryT]]",
) -> tuple[list[EntryT], str]:
    candidates = candidate_urls(url)
    last_error: Exception | None = None
    for candidate in candidates:
        normalized = normalize_url(candidate)
        local_payload = load_local_payload(normalized)
        if local_payload is not None:
            return parse_payload(local_payload, normalized), normalized
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(normalized)
                response.raise_for_status()
                data = response.json()
            return parse_payload(data, normalized), normalized
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if last_error is not None:
        raise last_error

    return [], normalize_url(url)


def atomic_replace_directory(*, existing_dir: Path, staged_dir: Path) -> None:
    existing_dir = existing_dir.resolve()
    staged_dir = staged_dir.resolve()
    parent = existing_dir.parent
    backup_dir = parent / f".{existing_dir.name}.backup-{uuid4().hex}"

    os.replace(existing_dir, backup_dir)
    try:
        os.replace(staged_dir, existing_dir)
    except Exception:
        os.replace(backup_dir, existing_dir)
        raise
    shutil.rmtree(backup_dir)
