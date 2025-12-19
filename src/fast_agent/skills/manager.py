from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fast_agent.config import Settings, get_settings
from fast_agent.constants import DEFAULT_SKILLS_PATHS
from fast_agent.core.logging.logger import get_logger
from fast_agent.skills.registry import SkillManifest, SkillRegistry

logger = get_logger(__name__)

DEFAULT_MARKETPLACE_URL = (
    "https://github.com/huggingface/skills/blob/main/.claude-plugin/marketplace.json"
)


@dataclass(frozen=True)
class MarketplaceSkill:
    name: str
    description: str | None
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None = None

    @property
    def repo_subdir(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == "skill.md":
            return str(path.parent)
        return str(path)

    @property
    def install_dir_name(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == "skill.md":
            return path.parent.name or self.name
        return path.name or self.name


class MarketplaceEntryModel(BaseModel):
    name: str | None = None
    description: str | None = None
    repo_url: str | None = Field(default=None, alias="repo")
    repo_ref: str | None = None
    repo_path: str | None = None
    source_url: str | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_entry(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        repo_url = _first_str(data, "repo", "repository", "git", "repo_url")
        repo_ref = _first_str(data, "ref", "branch", "tag", "revision", "commit")
        repo_path = _first_str(data, "path", "skill_path", "directory", "dir", "location")
        source_url = _first_str(data, "url", "skill_url", "source", "skill_source")

        parsed = _parse_github_url(repo_url) if repo_url else None
        if parsed and not repo_path:
            repo_url, repo_ref, repo_path = parsed
        elif parsed:
            repo_url = parsed[0]
            repo_ref = repo_ref or parsed[1]

        if source_url and (not repo_url or not repo_path):
            parsed_skill = _parse_github_url(source_url)
            if parsed_skill:
                repo_url, repo_ref, repo_path = parsed_skill

        name = _first_str(data, "name", "id", "slug", "title")
        description = _first_str(data, "description", "summary")
        if not name and repo_path:
            guessed = PurePosixPath(repo_path).parent.name
            name = guessed or repo_path

        return {
            "name": name,
            "description": description,
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "repo_path": repo_path,
            "source_url": source_url,
        }

    @classmethod
    def from_entry(cls, entry: dict[str, Any], *, source_url: str | None = None) -> "MarketplaceEntryModel":
        model = cls.model_validate(entry)
        if source_url and not model.source_url:
            return model.model_copy(update={"source_url": source_url})
        return model


class MarketplacePayloadModel(BaseModel):
    entries: list[MarketplaceEntryModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any, info: Any) -> Any:
        entries = _extract_marketplace_entries(data)
        source_url = None
        if info is not None and getattr(info, "context", None):
            source_url = info.context.get("source_url")
        if source_url:
            for entry in entries:
                if isinstance(entry, dict) and "source_url" not in entry:
                    entry["source_url"] = source_url
        return {"entries": entries}


def get_manager_directory(
    settings: Settings | None = None, *, cwd: Path | None = None
) -> Path:
    """Resolve the local skills directory the manager operates on."""
    base = cwd or Path.cwd()
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)

    directory = None
    if skills_settings and getattr(skills_settings, "directories", None):
        if skills_settings.directories:
            directory = skills_settings.directories[0]
    if not directory:
        directory = DEFAULT_SKILLS_PATHS[0]

    path = Path(directory).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def get_marketplace_url(settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)
    url = None
    if skills_settings is not None:
        url = getattr(skills_settings, "marketplace_url", None)
    return _normalize_marketplace_url(url or DEFAULT_MARKETPLACE_URL)


def list_local_skills(directory: Path) -> list[SkillManifest]:
    return SkillRegistry.load_directory(directory)


async def fetch_marketplace_skills(url: str) -> list[MarketplaceSkill]:
    normalized = _normalize_marketplace_url(url)
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(normalized)
        response.raise_for_status()
        data = response.json()
    return _parse_marketplace_payload(data, source_url=normalized)


async def install_marketplace_skill(
    skill: MarketplaceSkill,
    *,
    destination_root: Path,
) -> Path:
    return await asyncio.to_thread(_install_marketplace_skill_sync, skill, destination_root)


def remove_local_skill(skill_dir: Path, *, destination_root: Path) -> None:
    skill_dir = skill_dir.resolve()
    destination_root = destination_root.resolve()
    if destination_root not in skill_dir.parents:
        raise ValueError("Skill path is outside of the managed skills directory.")
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
    shutil.rmtree(skill_dir)


def select_skill_by_name_or_index(
    entries: Iterable[MarketplaceSkill], selector: str
) -> MarketplaceSkill | None:
    selector = selector.strip()
    if not selector:
        return None
    if selector.isdigit():
        index = int(selector)
        entries_list = list(entries)
        if 1 <= index <= len(entries_list):
            return entries_list[index - 1]
        return None
    selector_lower = selector.lower()
    for entry in entries:
        if entry.name.lower() == selector_lower:
            return entry
    return None


def select_manifest_by_name_or_index(
    manifests: Iterable[SkillManifest], selector: str
) -> SkillManifest | None:
    selector = selector.strip()
    if not selector:
        return None
    manifests_list = list(manifests)
    if selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(manifests_list):
            return manifests_list[index - 1]
        return None
    selector_lower = selector.lower()
    for manifest in manifests_list:
        if manifest.name.lower() == selector_lower:
            return manifest
    return None


def reload_skill_manifests(
    *,
    base_dir: Path | None = None,
    override_directories: list[Path] | None = None,
) -> tuple[SkillRegistry, list[SkillManifest]]:
    registry = SkillRegistry(
        base_dir=base_dir or Path.cwd(),
        directories=override_directories,
    )
    manifests = registry.load_manifests()
    return registry, manifests


def _normalize_marketplace_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            org, repo, _, ref = parts[:4]
            file_path = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{file_path}"
    return url


def _parse_marketplace_payload(
    payload: Any, *, source_url: str | None = None
) -> list[MarketplaceSkill]:
    try:
        model = MarketplacePayloadModel.model_validate(
            payload, context={"source_url": source_url}
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to parse marketplace payload",
            data={"error": str(exc)},
        )
        return []

    skills: list[MarketplaceSkill] = []
    for entry in model.entries:
        try:
            skill = _skill_from_entry_model(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse marketplace entry",
                data={"error": str(exc), "entry": _safe_json(entry.model_dump())},
            )
            continue
        if skill:
            skills.append(skill)
    return skills


def _extract_marketplace_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        for key in ("skills", "items", "entries", "marketplace"):
            value = payload.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            return [value for value in payload.values() if isinstance(value, dict)]
    raise ValueError("Unsupported marketplace payload format.")


def _skill_from_entry_model(model: MarketplaceEntryModel) -> MarketplaceSkill | None:
    if not model.repo_url or not model.repo_path:
        return None

    repo_path = _normalize_repo_path(model.repo_path)
    if not repo_path:
        return None

    return MarketplaceSkill(
        name=model.name or repo_path,
        description=model.description,
        repo_url=model.repo_url,
        repo_ref=model.repo_ref,
        repo_path=repo_path,
        source_url=model.source_url,
    )


def _normalize_repo_path(path: str) -> str | None:
    if not path:
        return None
    normalized = path.strip().lstrip("/")
    return normalized or None


def _parse_github_url(url: str | None) -> tuple[str, str | None, str] | None:
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


def _first_str(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _install_marketplace_skill_sync(skill: MarketplaceSkill, destination_root: Path) -> Path:
    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    install_dir = destination_root / skill.install_dir_name
    if install_dir.exists():
        raise FileExistsError(f"Skill already exists: {install_dir}")

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
        if skill.repo_ref:
            clone_args.extend(["--branch", skill.repo_ref])
        clone_args.extend([skill.repo_url, str(tmp_path)])

        _run_git(clone_args)
        _run_git(["git", "-C", str(tmp_path), "sparse-checkout", "set", skill.repo_subdir])
        _run_git(["git", "-C", str(tmp_path), "checkout"])

        source_dir = tmp_path / skill.repo_subdir
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Skill path not found in repository: {skill.repo_subdir}"
            )

        if (source_dir / "SKILL.md").exists():
            shutil.copytree(source_dir, install_dir)
        elif source_dir.name.lower() == "skill.md" and source_dir.is_file():
            install_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir, install_dir / "SKILL.md")
        else:
            raise FileNotFoundError("SKILL.md not found in the selected repository path.")

    return install_dir


def _run_git(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git command failed: {' '.join(args)}\n{stderr}")
