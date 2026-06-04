"""Git-backed marketplace source helpers."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, TypeVar
from uuid import uuid4

from fast_agent.marketplace.fetch import _source_location
from fast_agent.marketplace.source_models import SourcePathOid, SourceRevision

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

StatusT = TypeVar("StatusT")
CacheKeyT = TypeVar("CacheKeyT")
CacheValueT = TypeVar("CacheValueT")


def _remember(
    cache: "MutableMapping[CacheKeyT, CacheValueT]",
    key: CacheKeyT,
    value: CacheValueT,
) -> CacheValueT:
    cache[key] = value
    return value


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


def resolve_local_repo(repo_url: str) -> Path | None:
    source = _source_location(repo_url)
    if source.is_remote or source.path is None:
        return None
    repo_path = source.path.expanduser()
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
