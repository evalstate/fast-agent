"""Direct skill source resolution.

Direct sources are ad-hoc local or GitHub SKILL.md locations that can be
installed without a marketplace file.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from fast_agent.io.path_uri import file_uri_to_path
from fast_agent.marketplace.source_utils import parse_github_url
from fast_agent.skills.models import MarketplaceSkill
from fast_agent.skills.registry import SkillManifest, SkillRegistry

if TYPE_CHECKING:
    from urllib.parse import ParseResult


DIRECT_SOURCE_TIMEOUT_SECONDS = 7.0
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


@dataclass(frozen=True)
class DirectSkillSource:
    skill: MarketplaceSkill


class DirectSkillSourceError(ValueError):
    """Raised when a direct skill source cannot be resolved."""


def is_direct_skill_source(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return False

    parsed = urlparse(cleaned)
    if _is_github_skill_url(parsed):
        return True
    if parsed.scheme == "file":
        return True
    if parsed.scheme in {"http", "https"}:
        return False

    return _is_path_like(cleaned) or Path(cleaned).expanduser().exists()


async def resolve_direct_skill_source(value: str) -> DirectSkillSource:
    cleaned = value.strip()
    if not cleaned:
        raise DirectSkillSourceError("Direct skill source is empty.")

    parsed = urlparse(cleaned)
    if _is_github_skill_url(parsed):
        return await _resolve_github_source(cleaned)
    if parsed.scheme == "file":
        return _resolve_local_source(file_uri_to_path(parsed), source_url=cleaned)
    if parsed.scheme in {"http", "https"}:
        raise DirectSkillSourceError("Only GitHub skill URLs are supported for direct installs.")

    return _resolve_local_source(Path(cleaned).expanduser(), source_url=cleaned)


def resolve_direct_skill_source_sync(value: str) -> DirectSkillSource:
    import asyncio

    return asyncio.run(resolve_direct_skill_source(value))


async def _resolve_github_source(url: str) -> DirectSkillSource:
    parsed_source = parse_github_url(url)
    if parsed_source is None:
        raise DirectSkillSourceError("Unsupported GitHub skill URL.")

    repo_url, repo_ref, repo_path = parsed_source
    skill_md_path = _skill_manifest_repo_path(repo_path)
    raw_url = _github_raw_url(repo_url, repo_ref, skill_md_path)

    # Deliberately peek only at SKILL.md before cloning: this gives direct installs an
    # early, cheap manifest/name validation boundary before copying arbitrary repo
    # contents. The later install still clones the selected path through the standard
    # marketplace installer/provenance path, so a moving branch could theoretically
    # change between this read and the clone; callers accept that trade-off for now.
    try:
        async with httpx.AsyncClient(timeout=DIRECT_SOURCE_TIMEOUT_SECONDS) as client:
            response = await client.get(raw_url)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise DirectSkillSourceError("Timed out reading SKILL.md from direct source.") from exc
    except httpx.HTTPError as exc:
        raise DirectSkillSourceError(f"Failed to read SKILL.md from direct source: {exc}") from exc

    manifest = _parse_manifest(response.text, display_path=Path(skill_md_path))
    _validate_manifest_name(manifest.name)

    install_repo_path = str(PurePosixPath(skill_md_path).parent)
    if install_repo_path in {"", "."}:
        install_repo_path = skill_md_path

    return DirectSkillSource(
        skill=MarketplaceSkill(
            name=manifest.name,
            description=manifest.description,
            repo_url=repo_url,
            repo_ref=repo_ref,
            repo_path=install_repo_path,
            source_url=url,
            install_dir_name_override=manifest.name,
        ),
    )


def _resolve_local_source(path: Path, *, source_url: str) -> DirectSkillSource:
    source_path = path.expanduser()
    if not source_path.exists():
        raise DirectSkillSourceError(f"Direct skill source not found: {source_path}")

    manifest_path = source_path / "SKILL.md" if source_path.is_dir() else source_path
    if manifest_path.name.lower() != "skill.md" or not manifest_path.is_file():
        raise DirectSkillSourceError("Direct skill source must be a SKILL.md file or directory.")

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        raise DirectSkillSourceError(f"Failed to read SKILL.md from direct source: {exc}") from exc

    manifest = _parse_manifest(manifest_text, display_path=manifest_path)
    _validate_manifest_name(manifest.name)

    source_dir = manifest_path.parent
    repo_root = _find_git_root(source_dir) or source_dir
    repo_path = source_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    repo_url = repo_root.as_posix()

    return DirectSkillSource(
        skill=MarketplaceSkill(
            name=manifest.name,
            description=manifest.description,
            repo_url=repo_url,
            repo_ref=None,
            repo_path=repo_path or ".",
            source_url=source_url,
            install_dir_name_override=manifest.name,
        ),
    )


def _parse_manifest(manifest_text: str, *, display_path: Path) -> SkillManifest:
    manifest, error = SkillRegistry.parse_manifest_text(manifest_text, path=display_path)
    if manifest is None:
        detail = error or "invalid SKILL.md frontmatter"
        raise DirectSkillSourceError(f"Invalid SKILL.md frontmatter: {detail}")
    return manifest


def _validate_manifest_name(name: str) -> None:
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise DirectSkillSourceError(
            "Does not meet Agent Skills specification: skill name must be 1-64 "
            "characters, lowercase letters, numbers, and hyphens only, and must not "
            "start or end with a hyphen."
        )


def _is_github_skill_url(parsed: "ParseResult") -> bool:
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
            path = "/".join(parts[4:])
            return _looks_like_skill_path(path)
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            path = "/".join(parts[3:])
            return _looks_like_skill_path(path)
    return False


def _looks_like_skill_path(path: str) -> bool:
    posix_path = PurePosixPath(path)
    return posix_path.name.lower() == "skill.md" or bool(path.strip("/"))


def _is_path_like(value: str) -> bool:
    return (
        value.startswith((".", "/", "~"))
        or "/" in value
        or "\\" in value
    )


def _skill_manifest_repo_path(repo_path: str) -> str:
    path = PurePosixPath(repo_path)
    if path.name.lower() == "skill.md":
        return str(path)
    return str(path / "SKILL.md")


def _github_raw_url(repo_url: str, repo_ref: str | None, repo_path: str) -> str:
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if parsed.netloc not in {"github.com", "www.github.com"} or len(parts) != 2:
        raise DirectSkillSourceError("Unsupported GitHub repository URL.")
    org, repo = parts
    ref = repo_ref or "main"
    return f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{repo_path}"


def _find_git_root(path: Path) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root) if root else None
