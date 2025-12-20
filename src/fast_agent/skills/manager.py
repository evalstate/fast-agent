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

DEFAULT_SKILL_REGISTRIES = [
    "https://github.com/huggingface/skills",
    "https://github.com/anthropics/skills",
]

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
    bundle_name: str | None = None
    bundle_description: str | None = None

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
    bundle_name: str | None = None
    bundle_description: str | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_entry(cls, data: Any, info: Any) -> Any:
        if not isinstance(data, dict):
            return data

        context = getattr(info, "context", None) or {}
        default_repo_url = context.get("repo_url")
        default_repo_ref = context.get("repo_ref")

        repo_url = _first_str(data, "repo", "repository", "git", "repo_url")
        repo_ref = _first_str(data, "ref", "branch", "tag", "revision", "commit")
        repo_path = _first_str(
            data,
            "path",
            "skill_path",
            "directory",
            "dir",
            "location",
            "repo_path",
        )
        source_value = _first_str(data, "url", "skill_url", "source", "skill_source")
        source_url = source_value if _is_probable_url(source_value) else None

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
        elif source_value and not _is_probable_url(source_value) and not repo_path:
            repo_path = _normalize_source_path(source_value, data)

        name = _first_str(data, "name", "id", "slug", "title")
        description = _first_str(data, "description", "summary")
        bundle_name = _first_str(data, "bundle_name")
        bundle_description = _first_str(data, "bundle_description")
        if not name and repo_path:
            guessed = PurePosixPath(repo_path).parent.name
            name = guessed or repo_path

        repo_url = repo_url or default_repo_url
        repo_ref = repo_ref or default_repo_ref

        return {
            "name": name,
            "description": description,
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "repo_path": repo_path,
            "source_url": source_url,
            "bundle_name": bundle_name,
            "bundle_description": bundle_description,
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
        context = getattr(info, "context", None) or {}
        source_url = context.get("source_url")
        repo_url = context.get("repo_url")
        repo_ref = context.get("repo_ref")
        for entry in entries:
            if isinstance(entry, dict):
                if source_url and "source_url" not in entry:
                    entry["source_url"] = source_url
                if repo_url and "repo_url" not in entry and "repo" not in entry:
                    entry["repo_url"] = repo_url
                if repo_ref and "repo_ref" not in entry and "ref" not in entry:
                    entry["repo_ref"] = repo_ref
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
        # Check active registry first (set by /skills registry command)
        url = getattr(skills_settings, "marketplace_url", None)
        # Fall back to first configured registry
        if not url:
            urls = getattr(skills_settings, "marketplace_urls", None)
            if urls:
                url = urls[0]
    return _normalize_marketplace_url(url or DEFAULT_MARKETPLACE_URL)


def format_marketplace_display_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            org, repo = parts[:2]
            return f"https://github.com/{org}/{repo}"
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            org, repo = parts[:2]
            return f"https://github.com/{org}/{repo}"
    return url


def resolve_skill_directories(
    settings: Settings | None = None, *, cwd: Path | None = None
) -> list[Path]:
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)
    override_dirs: list[Path] | None = None
    if skills_settings and getattr(skills_settings, "directories", None):
        override_dirs = [Path(entry).expanduser() for entry in skills_settings.directories]
    manager_dir = get_manager_directory(resolved_settings, cwd=cwd)
    if override_dirs is None:
        return [manager_dir]
    if manager_dir not in override_dirs:
        override_dirs.append(manager_dir)
    return override_dirs


def list_local_skills(directory: Path) -> list[SkillManifest]:
    return SkillRegistry.load_directory(directory)


async def fetch_marketplace_skills(url: str) -> list[MarketplaceSkill]:
    skills, _ = await fetch_marketplace_skills_with_source(url)
    return skills


async def fetch_marketplace_skills_with_source(
    url: str,
) -> tuple[list[MarketplaceSkill], str]:
    candidates = _candidate_marketplace_urls(url)
    last_error: Exception | None = None
    for candidate in candidates:
        normalized = _normalize_marketplace_url(candidate)
        local_payload = _load_local_marketplace_payload(normalized)
        if local_payload is not None:
            return _parse_marketplace_payload(local_payload, source_url=normalized), normalized
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(normalized)
                response.raise_for_status()
                data = response.json()
            return _parse_marketplace_payload(data, source_url=normalized), normalized
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return [], _normalize_marketplace_url(url)


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


def _candidate_marketplace_urls(url: str) -> list[str]:
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
                suffix = ".claude-plugin/marketplace.json"
                if base_path:
                    suffix = f"{base_path.rstrip('/')}/{suffix}"
                return [
                    f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{suffix}"
                ]
            if len(parts) == 2:
                return [
                    f"https://raw.githubusercontent.com/{org}/{repo}/main/.claude-plugin/marketplace.json",
                    f"https://raw.githubusercontent.com/{org}/{repo}/master/.claude-plugin/marketplace.json",
                ]
    return [normalized]


def candidate_marketplace_urls(url: str) -> list[str]:
    return _candidate_marketplace_urls(url)


def _parse_marketplace_payload(
    payload: Any, *, source_url: str | None = None
) -> list[MarketplaceSkill]:
    repo_url = None
    repo_ref = None
    if source_url:
        parsed = _parse_github_url(source_url)
        if parsed:
            repo_url, repo_ref, _ = parsed
    try:
        model = MarketplacePayloadModel.model_validate(
            payload,
            context={
                "source_url": source_url,
                "repo_url": repo_url,
                "repo_ref": repo_ref,
            },
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
        if isinstance(payload.get("plugins"), list):
            plugin_root = None
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                plugin_root = metadata.get("pluginRoot") or metadata.get("plugin_root")
            entries: list[dict[str, Any]] = []
            for entry in payload.get("plugins", []):
                if isinstance(entry, dict):
                    entries.extend(_expand_plugin_entry(entry, plugin_root))
            return entries
        for key in ("skills", "items", "entries", "marketplace", "plugins"):
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
        bundle_name=model.bundle_name,
        bundle_description=model.bundle_description,
    )


def _normalize_repo_path(path: str) -> str | None:
    if not path:
        return None
    raw = path.strip()
    if not raw:
        return None
    raw = raw.replace("\\", "/")
    posix_path = PurePosixPath(raw)
    if posix_path.is_absolute():
        return None
    if ".." in posix_path.parts:
        return None
    normalized = str(posix_path).lstrip("/")
    if normalized in {"", "."}:
        return None
    return normalized


def _expand_plugin_entry(entry: dict[str, Any], plugin_root: str | None) -> list[dict[str, Any]]:
    source = entry.get("source")
    repo_url, repo_ref, repo_path = _parse_plugin_source(source, plugin_root)
    skills = entry.get("skills")
    bundle_name = entry.get("name")
    bundle_description = entry.get("description")
    base_entry = dict(entry)
    base_entry.pop("skills", None)
    if repo_url and not base_entry.get("repo_url"):
        base_entry["repo_url"] = repo_url
    if repo_ref and not base_entry.get("repo_ref"):
        base_entry["repo_ref"] = repo_ref
    if repo_path and not base_entry.get("repo_path"):
        base_entry["repo_path"] = repo_path

    if isinstance(skills, list) and skills:
        expanded: list[dict[str, Any]] = []
        for skill in skills:
            if not isinstance(skill, str) or not skill.strip():
                continue
            skill_name = PurePosixPath(skill).name or skill.strip()
            combined_path = _join_relative_paths(repo_path, skill)
            skill_entry = dict(base_entry)
            skill_entry["name"] = skill_name
            skill_entry["description"] = None
            skill_entry["bundle_name"] = bundle_name
            skill_entry["bundle_description"] = bundle_description
            skill_entry["repo_path"] = combined_path
            expanded.append(skill_entry)
        if expanded:
            return expanded
    return [base_entry]


def _parse_plugin_source(
    source: Any, plugin_root: str | None
) -> tuple[str | None, str | None, str | None]:
    repo_url = None
    repo_ref = None
    repo_path = None
    plugin_root_applied = False

    if isinstance(source, str) and source.strip():
        if _is_probable_url(source):
            repo_url = source.strip()
        else:
            repo_path = _join_relative_paths(plugin_root, source)
            plugin_root_applied = True
    elif isinstance(source, dict):
        source_kind = source.get("source")
        if source_kind == "github":
            repo = _first_str(source, "repo")
            if repo:
                repo_url = f"https://github.com/{repo}"
            repo_ref = _first_str(source, "ref", "branch", "tag", "revision", "commit")
            repo_path = _first_str(source, "path", "directory", "dir", "location")
        elif source_kind in {"url", "git"}:
            repo_url = _first_str(source, "url", "repo", "repository")
            repo_ref = _first_str(source, "ref", "branch", "tag", "revision", "commit")
            repo_path = _first_str(source, "path", "directory", "dir", "location")
        else:
            repo_url = _first_str(source, "url", "repo", "repository")
            repo_ref = _first_str(source, "ref", "branch", "tag", "revision", "commit")
            repo_path = _first_str(source, "path", "directory", "dir", "location")

    if repo_path and plugin_root and not plugin_root_applied and not _is_probable_url(repo_path):
        repo_path = _join_relative_paths(plugin_root, repo_path)

    return repo_url, repo_ref, repo_path


def _join_relative_paths(base: str | None, leaf: str | None) -> str | None:
    base_clean = _clean_relative_path(base)
    leaf_clean = _clean_relative_path(leaf)
    if not base_clean:
        return leaf_clean
    if not leaf_clean:
        return base_clean
    return str(PurePosixPath(base_clean) / PurePosixPath(leaf_clean))


def _clean_relative_path(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip().replace("\\", "/")
    cleaned = cleaned.lstrip("./").strip("/")
    if cleaned in {"", "."}:
        return None
    return cleaned


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


def _is_probable_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def _normalize_source_path(source: str, entry: dict[str, Any]) -> str | None:
    if not source:
        return None
    source_path = source.strip().lstrip("./")
    if not source_path:
        return None

    name = _first_str(entry, "name", "id", "slug", "title")
    if "/skills/" in source_path:
        return source_path
    if source_path.endswith("/skills"):
        if name:
            return f"{source_path}/{name}"
        return source_path
    if name:
        return f"{source_path}/skills/{name}"
    return source_path


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

    local_repo = _resolve_local_repo(skill.repo_url)
    if local_repo is not None:
        source_dir = _resolve_repo_subdir(local_repo, skill.repo_subdir)
        source_dir = _resolve_skill_source_dir(source_dir, skill.name)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Skill path not found in repository: {skill.repo_subdir}"
            )
        _copy_skill_source(source_dir, install_dir)
        return install_dir

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

        source_dir = _resolve_repo_subdir(tmp_path, skill.repo_subdir)
        source_dir = _resolve_skill_source_dir(source_dir, skill.name)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Skill path not found in repository: {skill.repo_subdir}"
            )

        _copy_skill_source(source_dir, install_dir)

    return install_dir


def _run_git(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git command failed: {' '.join(args)}\n{stderr}")


def _load_local_marketplace_payload(url: str) -> Any | None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        return _read_json_file(path)
    if parsed.scheme in {"http", "https"}:
        return None
    candidate = Path(url).expanduser()
    if candidate.exists():
        return _read_json_file(candidate)
    return None


def _read_json_file(path: Path) -> Any:
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def _resolve_local_repo(repo_url: str) -> Path | None:
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


def _resolve_repo_subdir(repo_root: Path, repo_subdir: str) -> Path:
    repo_root = repo_root.resolve()
    source_dir = (repo_root / Path(repo_subdir)).resolve()
    try:
        source_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Skill path escapes repository root.") from exc
    return source_dir


def _copy_skill_source(source_dir: Path, install_dir: Path) -> None:
    if (source_dir / "SKILL.md").exists():
        shutil.copytree(source_dir, install_dir)
    elif source_dir.name.lower() == "skill.md" and source_dir.is_file():
        install_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir, install_dir / "SKILL.md")
    else:
        raise FileNotFoundError("SKILL.md not found in the selected repository path.")


def _resolve_skill_source_dir(source_dir: Path, skill_name: str | None) -> Path:
    if (source_dir / "SKILL.md").exists():
        return source_dir
    if source_dir.is_file() and source_dir.name.lower() == "skill.md":
        return source_dir

    skills_dir = source_dir / "skills"
    if skill_name:
        named_dir = skills_dir / skill_name
        if (named_dir / "SKILL.md").exists():
            return named_dir

    if skills_dir.is_dir():
        candidates = [
            entry
            for entry in skills_dir.iterdir()
            if entry.is_dir() and (entry / "SKILL.md").exists()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise FileNotFoundError(
                "Multiple skills found; specify plugins[].skills to select one."
            )

    return source_dir
