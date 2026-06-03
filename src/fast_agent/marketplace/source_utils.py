"""Shared marketplace source parsing and git helper utilities."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, Generic, Literal, Protocol, TypeVar
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from fast_agent.io.path_uri import file_uri_to_path
from fast_agent.utils.text import strip_casefold, strip_str_to_none

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
    from urllib.parse import ParseResult

    from pydantic import ValidationInfo

EntryT = TypeVar("EntryT")
StatusT = TypeVar("StatusT")
OriginT = TypeVar("OriginT")
SourceT = TypeVar("SourceT")
SourceUpdateStatus = Literal["up_to_date", "update_available"]
CacheKeyT = TypeVar("CacheKeyT")
CacheValueT = TypeVar("CacheValueT")
_GITHUB_WEB_HOSTS = frozenset({"github.com", "www.github.com"})
_GITHUB_RAW_HOST = "raw.githubusercontent.com"
_GITHUB_SOURCE_PATH_ANCHORS = frozenset(
    {
        ".claude-plugin",
        "cards",
        "marketplace.json",
        "packs",
        "plugins",
        "skills",
        "SKILL.md",
    }
)
_GITHUB_COMMON_SINGLE_SEGMENT_REFS = frozenset(
    {
        "dev",
        "develop",
        "development",
        "gh-pages",
        "main",
        "master",
        "trunk",
    }
)
_GITHUB_COMMON_REPO_PATH_PREFIXES = frozenset(
    {
        ".github",
        "docs",
        "examples",
        "resources",
        "src",
        "tests",
    }
)
_GITHUB_COMMON_SLASH_BRANCH_PREFIXES = frozenset(
    {
        "bugfix",
        "feature",
        "fix",
        "hotfix",
        "release",
    }
)
_SCP_LIKE_GIT_SOURCE_RE = r"^[^@\s]+@[^:\s]+:[^\s]+$"
_SHA256_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MARKETPLACE_REPO_URL_KEYS = ("repo", "repository", "git", "repo_url")
MARKETPLACE_REPO_REF_KEYS = ("repo_ref", "ref", "branch", "tag", "revision", "commit")


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


class InstalledSourcePayloadFields(Protocol):
    schema_version: int
    installed_via: str
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


@dataclass(frozen=True, slots=True)
class ParsedGitHubUrl:
    repo_url: str
    repo_ref: str | None
    repo_path: str


@dataclass(frozen=True, slots=True)
class SourceRevision(Generic[StatusT]):
    revision: str | None
    status: StatusT | None = None
    detail: str | None = None

    def __iter__(self) -> Iterator[str | StatusT | None]:
        yield self.revision
        yield self.status
        yield self.detail


@dataclass(frozen=True, slots=True)
class SourcePathOid(Generic[StatusT]):
    path_oid: str | None
    status: StatusT | None = None
    detail: str | None = None

    def __iter__(self) -> Iterator[str | StatusT | None]:
        yield self.path_oid
        yield self.status
        yield self.detail


def _remember(
    cache: "MutableMapping[CacheKeyT, CacheValueT]",
    key: CacheKeyT,
    value: CacheValueT,
) -> CacheValueT:
    cache[key] = value
    return value


@dataclass(frozen=True, slots=True)
class SourceCopyResult(Generic[OriginT]):
    origin: OriginT
    commit: str | None
    path_oid: str | None


@dataclass(frozen=True, slots=True)
class InstalledSourceReadResult(Generic[SourceT]):
    source: SourceT | None = None
    error: str | None = None

    def __iter__(self) -> Iterator[SourceT | str | None]:
        yield self.source
        yield self.error


@dataclass(frozen=True, slots=True)
class SourceUpdateDecision:
    status: SourceUpdateStatus
    detail: str


@dataclass(frozen=True, slots=True)
class MarketplaceSourceContext:
    source_url: str | None = None
    repo_url: str | None = None
    repo_ref: str | None = None

    def as_validation_context(self) -> dict[str, str | None]:
        return {
            "source_url": self.source_url,
            "repo_url": self.repo_url,
            "repo_ref": self.repo_ref,
        }


@dataclass(frozen=True, slots=True)
class MarketplaceRepoFields:
    repo_url: str | None = None
    repo_ref: str | None = None
    repo_path: str | None = None


@dataclass(frozen=True, slots=True)
class _SourceLocation:
    path: Path | None
    is_remote: bool = False


def read_installed_source_file(
    sidecar_path: Path,
    *,
    parse_payload: "Callable[[dict[str, Any]], SourceT]",
) -> InstalledSourceReadResult[SourceT]:
    if not sidecar_path.exists():
        return InstalledSourceReadResult()

    try:
        payload = read_json_file(sidecar_path)
    except Exception as exc:
        return InstalledSourceReadResult(error=f"invalid json: {exc}")

    if not isinstance(payload, dict):
        return InstalledSourceReadResult(error="metadata root must be an object")

    try:
        source = parse_payload(payload)
    except ValueError as exc:
        return InstalledSourceReadResult(error=str(exc))
    return InstalledSourceReadResult(source=source)


def installed_source_payload(source: InstalledSourcePayloadFields) -> dict[str, Any]:
    return {
        "schema_version": source.schema_version,
        "installed_via": source.installed_via,
        "source_origin": source.source_origin,
        "repo_url": source.repo_url,
        "repo_ref": source.repo_ref,
        "repo_path": source.repo_path,
        "source_url": source.source_url,
        "installed_commit": source.installed_commit,
        "installed_path_oid": source.installed_path_oid,
        "installed_revision": source.installed_revision,
        "installed_at": source.installed_at,
        "content_fingerprint": source.content_fingerprint,
    }


def write_installed_source_file(
    sidecar_path: Path,
    source: InstalledSourcePayloadFields,
    *,
    extra_payload: "Mapping[str, Any] | None" = None,
) -> None:
    payload = installed_source_payload(source)
    if extra_payload:
        payload.update(extra_payload)
    sidecar_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compute_directory_content_fingerprint(
    directory: Path,
    *,
    sidecar_path: Path,
    ignore_path: "Callable[[Path], bool] | None" = None,
) -> str:
    digest = hashlib.sha256()
    root = directory.resolve()
    resolved_sidecar = sidecar_path.resolve()

    for path in sorted(root.rglob("*")):
        if path == resolved_sidecar:
            continue
        if not path.is_file():
            continue
        if ignore_path is not None and ignore_path(path):
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return f"sha256:{digest.hexdigest()}"


def decide_source_update_status(
    *,
    available_path_oid: str | None,
    current_path_oid: str | None,
    available_revision: str,
    current_revision: str,
    content_changed_detail: str,
) -> SourceUpdateDecision:
    if available_path_oid and current_path_oid:
        if available_path_oid != current_path_oid:
            return SourceUpdateDecision(
                status="update_available",
                detail=content_changed_detail,
            )
    elif available_revision != current_revision:
        return SourceUpdateDecision(
            status="update_available",
            detail="new revision available",
        )

    return SourceUpdateDecision(
        status="up_to_date",
        detail="already up to date",
    )


def normalize_marketplace_url(url: str) -> str:
    normalized = url.strip()
    parsed = urlparse(normalized)
    if is_github_web_host(parsed.netloc) or is_github_raw_host(parsed.netloc):
        parts = parsed.path.strip("/").split("/")
        is_github_blob = is_github_web_host(parsed.netloc) and (
            len(parts) >= 5 and parts[2] == "blob"
        )
        is_raw_github = is_github_raw_host(parsed.netloc)
        if is_github_blob or is_raw_github:
            parsed_source = parse_github_url(normalized)
            return _github_raw_content_url(parsed_source) or normalized
    return normalized


def is_github_web_host(host: str) -> bool:
    return strip_casefold(host) in _GITHUB_WEB_HOSTS


def is_github_raw_host(host: str) -> bool:
    return strip_casefold(host) == _GITHUB_RAW_HOST


def _path_name_matches(path: PurePosixPath, filename: str) -> bool:
    return strip_casefold(path.name) == strip_casefold(filename)


def github_raw_file_url(
    *,
    repo_url: str,
    repo_ref: str | None,
    repo_path: str,
) -> str | None:
    parsed_repo = urlparse(repo_url)
    repo_parts = parsed_repo.path.strip("/").split("/")
    if not is_github_web_host(parsed_repo.netloc) or len(repo_parts) != 2:
        return None
    org, repo = repo_parts
    ref = repo_ref or "main"
    return f"https://{_GITHUB_RAW_HOST}/{org}/{repo}/{ref}/{repo_path}"


def _github_raw_content_url(parsed_source: ParsedGitHubUrl | None) -> str | None:
    if parsed_source is None or parsed_source.repo_ref is None:
        return None
    return github_raw_file_url(
        repo_url=parsed_source.repo_url,
        repo_ref=parsed_source.repo_ref,
        repo_path=parsed_source.repo_path,
    )


def _is_local_marketplace_candidate(parsed: "ParseResult") -> bool:
    return parsed.scheme in {"file", ""} and parsed.netloc == ""


def _local_marketplace_path(parsed: "ParseResult") -> Path:
    if parsed.scheme == "file":
        return file_uri_to_path(parsed)
    return Path(parsed.path).expanduser()


def _candidate_local_marketplace_urls(normalized: str, parsed: "ParseResult") -> list[str]:
    path = _local_marketplace_path(parsed)
    if path.exists() and path.is_dir():
        claude_plugin = path / ".claude-plugin" / "marketplace.json"
        if claude_plugin.exists():
            return [claude_plugin.as_posix()]
        fallback = path / "marketplace.json"
        if fallback.exists():
            return [fallback.as_posix()]
    return [normalized]


def _candidate_github_marketplace_urls(normalized: str, parsed: "ParseResult") -> list[str]:
    if not is_github_web_host(parsed.netloc):
        return []

    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        return []

    org, repo = parts[:2]
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        parsed_source = parse_github_url(normalized)
        if parsed_source is not None and parsed_source.repo_ref is not None:
            return _github_marketplace_candidates(
                org,
                repo,
                parsed_source.repo_ref,
                parsed_source.repo_path,
            )

    if len(parts) == 2:
        return [
            *_github_marketplace_candidates(org, repo, "main", ""),
            *_github_marketplace_candidates(org, repo, "master", ""),
        ]

    return []


def candidate_marketplace_urls(url: str) -> list[str]:
    normalized = strip_str_to_none(url)
    if normalized is None:
        return []

    parsed = urlparse(normalized)
    if _is_local_marketplace_candidate(parsed):
        return _candidate_local_marketplace_urls(normalized, parsed)

    github_candidates = _candidate_github_marketplace_urls(normalized, parsed)
    if github_candidates:
        return github_candidates

    return [normalized]


def _github_marketplace_candidates(org: str, repo: str, ref: str, base_path: str) -> list[str]:
    suffixes = _marketplace_path_candidates(base_path)
    return [
        f"https://{_GITHUB_RAW_HOST}/{org}/{repo}/{ref}/{suffix}"
        for suffix in suffixes
    ]


def _marketplace_path_candidates(base_path: str) -> list[str]:
    cleaned = base_path.strip().strip("/")
    if not cleaned:
        return [".claude-plugin/marketplace.json", "marketplace.json"]

    path = PurePosixPath(cleaned)
    if _path_name_matches(path, "marketplace.json"):
        return [str(path)]
    if path.name == ".claude-plugin":
        return [str(path / "marketplace.json")]

    return [
        str(path / ".claude-plugin" / "marketplace.json"),
        str(path / "marketplace.json"),
    ]


def _split_github_ref_and_path(parts: "Sequence[str]") -> tuple[str, str] | None:
    if len(parts) < 2:
        return None

    if parts[0] in _GITHUB_COMMON_SINGLE_SEGMENT_REFS:
        return _github_ref_path_split(parts, repo_path_index=1)

    if parts[0] in _GITHUB_COMMON_SLASH_BRANCH_PREFIXES:
        prefixed_ref_split = _split_github_prefixed_ref(parts)
        if prefixed_ref_split is not None:
            return prefixed_ref_split

    if len(parts) >= 3 and parts[1] in _GITHUB_COMMON_REPO_PATH_PREFIXES:
        return _github_ref_path_split(parts, repo_path_index=1)

    anchor_split = _split_github_ref_at_source_path_anchor(parts)
    if anchor_split is not None:
        return anchor_split

    return _github_ref_path_split(parts, repo_path_index=1)


def _github_ref_path_split(
    parts: "Sequence[str]",
    *,
    repo_path_index: int,
) -> tuple[str, str] | None:
    ref = "/".join(parts[:repo_path_index])
    repo_path = "/".join(parts[repo_path_index:])
    if ref and repo_path:
        return ref, repo_path
    return None


def _split_github_prefixed_ref(parts: "Sequence[str]") -> tuple[str, str] | None:
    if len(parts) >= 3 and parts[1] in _GITHUB_SOURCE_PATH_ANCHORS:
        return _github_ref_path_split(parts, repo_path_index=2)
    if len(parts) >= 4 and parts[2] in _GITHUB_COMMON_REPO_PATH_PREFIXES:
        return _github_ref_path_split(parts, repo_path_index=2)
    return _split_github_ref_at_source_path_anchor(parts)


def _split_github_ref_at_source_path_anchor(
    parts: "Sequence[str]",
) -> tuple[str, str] | None:
    for index, part in enumerate(parts[1:], start=1):
        if part in _GITHUB_SOURCE_PATH_ANCHORS:
            ref = "/".join(parts[:index])
            repo_path = "/".join(parts[index:])
            if ref and repo_path:
                return ref, repo_path
    return None


def parse_github_url(url: str | None) -> ParsedGitHubUrl | None:
    normalized = strip_str_to_none(url)
    if normalized is None:
        return None
    parsed = urlparse(normalized)
    if is_github_web_host(parsed.netloc):
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
            return _parsed_github_source(parts[:2], parts[3:])
    if is_github_raw_host(parsed.netloc):
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            return _parsed_github_source(parts[:2], parts[2:])
    return None


def _parsed_github_source(
    repo_parts: "Sequence[str]",
    ref_and_path_parts: "Sequence[str]",
) -> ParsedGitHubUrl | None:
    if len(repo_parts) < 2:
        return None
    split_ref = _split_github_ref_and_path(ref_and_path_parts)
    if split_ref is None:
        return None
    org, repo = repo_parts[:2]
    ref, repo_path = split_ref
    return ParsedGitHubUrl(
        repo_url=f"https://github.com/{org}/{repo}",
        repo_ref=ref,
        repo_path=repo_path,
    )


def parse_ls_remote_commit(output: str) -> str | None:
    candidates = [
        candidate
        for line in output.splitlines()
        if (candidate := _ls_remote_ref_candidate(line)) is not None
    ]
    peeled_tag_commit = next(
        (commit for commit, ref in candidates if ref.strip().endswith("^{}")),
        None,
    )
    if peeled_tag_commit:
        return peeled_tag_commit
    return candidates[0][0] if candidates else None


def _ls_remote_ref_candidate(line: str) -> tuple[str, str] | None:
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2 or not _is_ls_remote_ref(parts[1]):
        return None
    return parts[0], parts[1]


def _is_ls_remote_ref(ref: str) -> bool:
    ref_name = ref.strip()
    return ref_name == "HEAD" or ref_name.startswith("refs/")


def load_local_marketplace_payload(url: str) -> Any | None:
    source = _source_location(url)
    if source.is_remote or source.path is None:
        return None
    candidate = source.path.expanduser()
    if candidate.exists():
        if candidate.is_dir():
            raise FileNotFoundError(f"marketplace.json not found in directory: {candidate}")
        return read_json_file(candidate)
    return None


def read_json_file(path: Path) -> Any:
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def resolve_local_repo(repo_url: str) -> Path | None:
    source = _source_location(repo_url)
    if source.is_remote or source.path is None:
        return None
    repo_path = source.path
    repo_path = repo_path.expanduser()
    if not repo_path.is_absolute():
        repo_path = repo_path.resolve()
    if repo_path.exists():
        return repo_path
    return None


def resolve_repo_subdir(repo_root: Path, repo_subdir: str, *, label: str) -> Path:
    repo_root = repo_root.resolve()
    source_dir = (repo_root / Path(repo_subdir)).resolve()
    try:
        source_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes repository root.") from exc
    return source_dir


def derive_local_repo_root(source_url: str) -> str | None:
    source = _source_location(source_url)
    if source.is_remote or source.path is None:
        return None

    path = source.path
    path = path.expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if not path.exists():
        return None

    if path.is_file() and path.name == "marketplace.json":
        repo_root = path.parent.parent if path.parent.name == ".claude-plugin" else path.parent
        if repo_root.exists():
            return str(repo_root)

    if path.is_dir():
        return str(path)

    return None


def _source_location(value: str) -> _SourceLocation:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return _SourceLocation(path=file_uri_to_path(parsed))
    if is_git_source_url(value):
        return _SourceLocation(path=None, is_remote=True)
    return _SourceLocation(path=Path(value))


def marketplace_source_context(source_url: str | None) -> MarketplaceSourceContext:
    if not source_url:
        return MarketplaceSourceContext()

    parsed = parse_github_url(source_url)
    if parsed:
        return MarketplaceSourceContext(
            source_url=source_url,
            repo_url=parsed.repo_url,
            repo_ref=parsed.repo_ref,
        )

    return MarketplaceSourceContext(
        source_url=source_url,
        repo_url=derive_local_repo_root(source_url),
        repo_ref=None,
    )


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


def resolve_git_path_oid_if_commit(
    repo_root: Path,
    commit: str | None,
    repo_path: str,
    *,
    resolve_git_path_oid_fn: "Callable[[Path, str, str], str | None]" = resolve_git_path_oid,
) -> str | None:
    if commit is None:
        return None
    return resolve_git_path_oid_fn(repo_root, commit, repo_path)


def is_git_source_dirty(repo_root: Path, source_path: Path) -> bool:
    try:
        relative_source = source_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False

    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            relative_source.as_posix(),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def run_git(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = subprocess_failure_detail(result)
        raise RuntimeError(f"Git command failed: {' '.join(args)}\n{stderr}")


def subprocess_failure_detail(
    result: subprocess.CompletedProcess[str],
    fallback: str = "",
) -> str:
    return result.stderr.strip() or result.stdout.strip() or fallback


def _sparse_clone_args(repo_url: str, repo_ref: str | None, destination_dir: Path) -> list[str]:
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
    clone_args.extend([repo_url, str(destination_dir)])
    return clone_args


def clone_sparse_checkout(
    *,
    repo_url: str,
    repo_ref: str | None,
    repo_subdir: str,
    destination_dir: Path,
    checkout_ref: str | None = None,
    run_git_fn: "Callable[[list[str]], None]" = run_git,
) -> None:
    run_git_fn(_sparse_clone_args(repo_url, repo_ref, destination_dir))
    run_git_fn(["git", "-C", str(destination_dir), "sparse-checkout", "set", repo_subdir])
    checkout_args = ["git", "-C", str(destination_dir), "checkout"]
    if checkout_ref:
        checkout_args.append(checkout_ref)
    run_git_fn(checkout_args)


def pinned_checkout_ref(pinned_revision: str | None, *, local_revision: str) -> str | None:
    if pinned_revision and pinned_revision != local_revision:
        return pinned_revision
    return None


def copy_git_path_from_commit(
    *,
    repo_root: Path,
    commit: str,
    repo_subdir: str,
    destination_dir: Path,
    missing_message: str,
) -> None:
    archive_pathspec = commit if repo_subdir in {"", ".", "./"} else f"{commit}:{repo_subdir}"
    result = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar", archive_pathspec],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise FileNotFoundError(stderr or missing_message)

    destination_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        archive.write(result.stdout)
        archive.seek(0)
        extract_tar_safely(archive, destination_dir)


def extract_tar_safely(archive_file: BinaryIO, destination_dir: Path) -> None:
    destination_root = destination_dir.resolve()
    with tarfile.open(fileobj=archive_file, mode="r:") as archive:
        for member in archive.getmembers():
            _validate_tar_member_path(member, destination_root)
        archive.extractall(destination_root, filter="data")


def _validate_tar_member_path(member: tarfile.TarInfo, destination_root: Path) -> None:
    target = (destination_root / member.name).resolve()
    try:
        target.relative_to(destination_root)
    except ValueError as exc:
        raise tarfile.TarError(
            f"Archive member escapes destination: {member.name}"
        ) from exc


def _missing_ref_revision(repo_ref: str, status: StatusT) -> SourceRevision[StatusT]:
    return SourceRevision(
        revision=None,
        status=status,
        detail=f"ref not found: {repo_ref}",
    )


def _local_source_revision(
    *,
    local_repo: Path,
    repo_ref: str | None,
    local_revision: str,
    source_ref_missing_status: StatusT,
    resolve_git_commit_fn: "Callable[[Path, str | None], str | None]",
) -> SourceRevision[StatusT]:
    revision = resolve_git_commit_fn(local_repo, repo_ref or "HEAD")
    if revision is None and repo_ref:
        return _missing_ref_revision(repo_ref, source_ref_missing_status)
    return SourceRevision(revision or local_revision)


def _empty_remote_revision(
    repo_ref: str | None,
    *,
    source_ref_missing_status: StatusT,
    source_unreachable_status: StatusT,
) -> SourceRevision[StatusT]:
    if repo_ref:
        return _missing_ref_revision(repo_ref, source_ref_missing_status)
    return SourceRevision(
        revision=None,
        status=source_unreachable_status,
        detail="unable to resolve source HEAD",
    )


def _remote_source_revision(
    *,
    repo_url: str,
    repo_ref: str | None,
    source_ref_missing_status: StatusT,
    source_unreachable_status: StatusT,
    run_subprocess_fn: "Callable[..., subprocess.CompletedProcess[str]]",
) -> SourceRevision[StatusT]:
    ls_remote_args = ["git", "ls-remote", repo_url, repo_ref or "HEAD"]
    result = run_subprocess_fn(ls_remote_args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return SourceRevision(
            revision=None,
            status=source_unreachable_status,
            detail=subprocess_failure_detail(result, "unable to reach source"),
        )

    output = result.stdout.strip()
    if not output:
        return _empty_remote_revision(
            repo_ref,
            source_ref_missing_status=source_ref_missing_status,
            source_unreachable_status=source_unreachable_status,
        )

    commit = parse_ls_remote_commit(output)
    if commit is None:
        return SourceRevision(
            revision=None,
            status=source_unreachable_status,
            detail="unable to resolve source revision",
        )

    return SourceRevision(commit)


def resolve_source_revision(
    *,
    repo_url: str,
    repo_ref: str | None,
    head_cache: "MutableMapping[tuple[str, str | None], SourceRevision[StatusT]]",
    local_revision: str,
    source_ref_missing_status: StatusT,
    source_unreachable_status: StatusT,
    resolve_local_repo_fn: "Callable[[str], Path | None]" = resolve_local_repo,
    resolve_git_commit_fn: "Callable[[Path, str | None], str | None]" = resolve_git_commit,
    run_subprocess_fn: "Callable[..., subprocess.CompletedProcess[str]]" = subprocess.run,
) -> SourceRevision[StatusT]:
    cache_key = (repo_url, repo_ref)
    cached = head_cache.get(cache_key)
    if cached is not None:
        return cached

    local_repo = resolve_local_repo_fn(repo_url)
    if local_repo is not None:
        return _remember(
            head_cache,
            cache_key,
            _local_source_revision(
                local_repo=local_repo,
                repo_ref=repo_ref,
                local_revision=local_revision,
                source_ref_missing_status=source_ref_missing_status,
                resolve_git_commit_fn=resolve_git_commit_fn,
            ),
        )

    return _remember(
        head_cache,
        cache_key,
        _remote_source_revision(
            repo_url=repo_url,
            repo_ref=repo_ref,
            source_ref_missing_status=source_ref_missing_status,
            source_unreachable_status=source_unreachable_status,
            run_subprocess_fn=run_subprocess_fn,
        ),
    )


def resolve_source_path_oid(
    *,
    repo_url: str,
    repo_ref: str | None,
    repo_path: str,
    commit: str,
    path_cache: "MutableMapping[tuple[str, str | None, str, str], SourcePathOid[StatusT]]",
    source_ref_missing_status: StatusT,
    source_unreachable_status: StatusT,
    source_path_missing_status: StatusT,
    resolve_local_repo_fn: "Callable[[str], Path | None]" = resolve_local_repo,
    resolve_git_path_oid_fn: "Callable[[Path, str, str], str | None]" = resolve_git_path_oid,
) -> SourcePathOid[StatusT]:
    cache_key = (repo_url, repo_ref, repo_path, commit)
    cached = path_cache.get(cache_key)
    if cached is not None:
        return cached

    local_repo = resolve_local_repo_fn(repo_url)
    if local_repo is not None:
        path_oid = resolve_git_path_oid_fn(local_repo, commit, repo_path)
        if path_oid is None:
            return _remember(
                path_cache,
                cache_key,
                SourcePathOid(
                    path_oid=None,
                    status=source_path_missing_status,
                    detail=f"path missing at revision {commit}: {repo_path}",
                ),
            )
        return _remember(path_cache, cache_key, SourcePathOid(path_oid))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        result = subprocess.run(
            _sparse_clone_args(repo_url, repo_ref, tmp_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = subprocess_failure_detail(result)
            if repo_ref and "Remote branch" in stderr and "not found" in stderr:
                path_result = SourcePathOid(
                    path_oid=None,
                    status=source_ref_missing_status,
                    detail=f"ref not found: {repo_ref}",
                )
            else:
                path_result = SourcePathOid(
                    path_oid=None,
                    status=source_unreachable_status,
                    detail=stderr or "unable to reach source",
                )
            return _remember(path_cache, cache_key, path_result)

        path_oid = resolve_git_path_oid_fn(tmp_path, commit, repo_path)
        if path_oid is None:
            return _remember(
                path_cache,
                cache_key,
                SourcePathOid(
                    path_oid=None,
                    status=source_path_missing_status,
                    detail=f"path missing at revision {commit}: {repo_path}",
                ),
            )

    return _remember(path_cache, cache_key, SourcePathOid(path_oid))


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

    repo_url = _required_installed_source_string(payload, "repo_url")
    repo_ref = _optional_installed_source_string(payload, "repo_ref")

    repo_path_raw = payload.get("repo_path")
    if not isinstance(repo_path_raw, str):
        raise ValueError("repo_path is required")
    repo_path = normalize_repo_path(repo_path_raw)
    if not repo_path:
        raise ValueError("repo_path is invalid")

    source_url = _optional_installed_source_string(payload, "source_url")
    installed_commit = _optional_installed_source_string(
        payload,
        "installed_commit",
        allow_blank=False,
    )
    installed_path_oid = _optional_installed_source_string(
        payload,
        "installed_path_oid",
        allow_blank=False,
    )
    installed_revision = _required_installed_source_string(payload, "installed_revision")
    installed_at = _required_installed_source_string(payload, "installed_at")

    content_fingerprint = _required_sha256_fingerprint(payload, "content_fingerprint")

    return ParsedInstalledSourceFields(
        source_origin=source_origin_raw,
        repo_url=repo_url,
        repo_ref=repo_ref,
        repo_path=repo_path,
        source_url=source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=installed_at,
        content_fingerprint=content_fingerprint,
    )


def _required_installed_source_string(payload: "Mapping[str, Any]", field_name: str) -> str:
    value = payload.get(field_name)
    normalized = strip_str_to_none(value)
    if normalized is None:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_installed_source_string(
    payload: "Mapping[str, Any]",
    field_name: str,
    *,
    allow_blank: bool = True,
) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    normalized = strip_str_to_none(value)
    if normalized is not None:
        return normalized
    if allow_blank:
        return None
    raise ValueError(f"{field_name} must be a non-empty string or null")


def _required_sha256_fingerprint(payload: "Mapping[str, Any]", field_name: str) -> str:
    value = payload.get(field_name)
    if isinstance(value, str) and _SHA256_FINGERPRINT_RE.fullmatch(value):
        return value
    raise ValueError(f"{field_name} must be a sha256 fingerprint")


def normalize_relative_repo_path(path: str, *, allow_current_dir: bool = False) -> str | None:
    raw = path.strip().replace("\\", "/")
    if not raw:
        return None
    if re.match(r"^[A-Za-z]:($|/)", raw):
        return None
    posix_path = PurePosixPath(raw)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        return None
    normalized = str(posix_path).lstrip("/")
    if normalized == "" or (normalized == "." and not allow_current_dir):
        return None
    return normalized


def repo_subdir_for_manifest_path(repo_path: str, manifest_filename: str) -> str:
    path_value = normalize_relative_repo_path(repo_path, allow_current_dir=True) or repo_path.strip()
    path = PurePosixPath(path_value)
    if _path_name_matches(path, manifest_filename):
        return str(path.parent)
    return str(path)


def repo_name_for_manifest_path(
    repo_path: str,
    manifest_filename: str,
    *,
    allow_current_dir: bool = False,
) -> str:
    path_value = (
        normalize_relative_repo_path(repo_path, allow_current_dir=allow_current_dir)
        or repo_path.strip()
    )
    path = PurePosixPath(path_value)
    if _path_name_matches(path, manifest_filename) and path.parent.name:
        return path.parent.name
    return path.name or path_value


def is_probable_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def is_git_source_url(value: str | None) -> bool:
    stripped = strip_str_to_none(value)
    if stripped is None:
        return False
    if is_probable_url(stripped):
        return True
    return re.match(_SCP_LIKE_GIT_SOURCE_RE, stripped) is not None


def first_nonempty_str(data: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if (normalized := strip_str_to_none(value)) is not None:
            return normalized
    return None


def explicit_entry_source_url(
    data: Mapping[str, Any],
    *keys: str,
    default_source_url: str | None,
) -> str | None:
    """Return an entry-provided source URL, excluding registry/context provenance."""
    for key in keys:
        value = data.get(key)
        if (source_url := strip_str_to_none(value)) is not None:
            if source_url != default_source_url:
                return source_url
    return None


def marketplace_repo_fields(
    data: Mapping[str, Any],
    *,
    repo_path_keys: "Sequence[str]",
    repo_url_keys: "Sequence[str]" = MARKETPLACE_REPO_URL_KEYS,
    repo_ref_keys: "Sequence[str]" = MARKETPLACE_REPO_REF_KEYS,
) -> MarketplaceRepoFields:
    """Extract and normalize common repository fields from a marketplace entry."""
    repo_url = first_nonempty_str(data, *repo_url_keys)
    repo_ref = first_nonempty_str(data, *repo_ref_keys)
    repo_path = first_nonempty_str(data, *repo_path_keys)

    parsed = parse_github_url(repo_url) if repo_url else None
    if parsed is None:
        return MarketplaceRepoFields(
            repo_url=repo_url,
            repo_ref=repo_ref,
            repo_path=repo_path,
        )

    return MarketplaceRepoFields(
        repo_url=parsed.repo_url,
        repo_ref=repo_ref or parsed.repo_ref,
        repo_path=repo_path or parsed.repo_path,
    )


def normalize_marketplace_payload(
    data: Any,
    info: "ValidationInfo",
    *,
    extract_entries: "Callable[[Any], list[dict[str, Any]]]",
) -> dict[str, list[dict[str, Any]]]:
    entries = extract_entries(data)
    context = info.context or {}
    source_url = context.get("source_url")
    repo_url = context.get("repo_url")
    repo_ref = context.get("repo_ref")
    normalized_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry = dict(entry)
        if source_url and "source_url" not in entry:
            entry["source_url"] = source_url
        if repo_url and "repo_url" not in entry and "repo" not in entry:
            entry["repo_url"] = repo_url
        if repo_ref and "repo_ref" not in entry and "ref" not in entry:
            entry["repo_ref"] = repo_ref
        normalized_entries.append(entry)
    return {"entries": normalized_entries}


def _dict_entries_from_sequence(values: "Sequence[Any]") -> list[dict[str, Any]]:
    return [entry for entry in values if isinstance(entry, dict)]


def extract_dict_entries(
    payload: Any,
    keys: "Sequence[str]",
    *,
    allow_mapping_values: bool = False,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _dict_entries_from_sequence(payload)

    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return _dict_entries_from_sequence(value)

        if allow_mapping_values and all(isinstance(value, dict) for value in payload.values()):
            return _dict_entries_from_sequence(list(payload.values()))

    raise ValueError("Unsupported marketplace payload format.")


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
        try:
            local_payload = load_local_payload(normalized)
            if local_payload is not None:
                return parse_payload(local_payload, normalized), normalized
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(normalized)
                response.raise_for_status()
                data = response.json()
            return parse_payload(data, normalized), normalized
        except Exception as exc:
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

    existing_dir.replace(backup_dir)
    try:
        staged_dir.replace(existing_dir)
    except Exception:
        backup_dir.replace(existing_dir)
        raise
    shutil.rmtree(backup_dir)
