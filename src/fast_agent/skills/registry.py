from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Sequence

import frontmatter

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SkillManifest:
    """Represents a single skill description loaded from SKILL.md."""

    name: str
    description: str
    body: str
    path: Path
    relative_path: Path | None = None


class SkillRegistry:
    """Simple registry that resolves a single skills directory and parses manifests."""

    DEFAULT_CANDIDATES = (Path(".fast-agent/skills"), Path(".claude/skills"))

    def __init__(
        self, *, base_dir: Path | None = None, override_directory: Path | None = None
    ) -> None:
        self._base_dir = base_dir or Path.cwd()
        self._directory: Path | None = None
        self._override_failed: bool = False
        self._errors: List[dict[str, str]] = []
        if override_directory:
            resolved = self._resolve_directory(override_directory)
            if resolved and resolved.exists() and resolved.is_dir():
                self._directory = resolved
            else:
                logger.warning(
                    "Skills directory override not found",
                    data={"directory": str(resolved)},
                )
                self._override_failed = True
        if self._directory is None and not self._override_failed:
            self._directory = self._find_default_directory()

    @property
    def directory(self) -> Path | None:
        return self._directory

    @property
    def override_failed(self) -> bool:
        return self._override_failed

    def load_manifests(self) -> List[SkillManifest]:
        self._errors = []
        if not self._directory:
            return []
        return self._load_directory(self._directory, self._errors)

    def load_manifests_with_errors(self) -> tuple[List[SkillManifest], List[dict[str, str]]]:
        manifests = self.load_manifests()
        return manifests, list(self._errors)

    @property
    def errors(self) -> List[dict[str, str]]:
        return list(self._errors)

    def _find_default_directory(self) -> Path | None:
        for candidate in self.DEFAULT_CANDIDATES:
            resolved = self._resolve_directory(candidate)
            if resolved and resolved.exists() and resolved.is_dir():
                return resolved
        return None

    def _resolve_directory(self, directory: Path) -> Path:
        if directory.is_absolute():
            return directory
        return (self._base_dir / directory).resolve()

    @classmethod
    def load_directory(cls, directory: Path) -> List[SkillManifest]:
        if not directory.exists() or not directory.is_dir():
            logger.debug(
                "Skills directory not found",
                data={"directory": str(directory)},
            )
            return []
        return cls._load_directory(directory)

    @classmethod
    def load_directory_with_errors(
        cls, directory: Path
    ) -> tuple[List[SkillManifest], List[dict[str, str]]]:
        errors: List[dict[str, str]] = []
        manifests = cls._load_directory(directory, errors)
        return manifests, errors

    @classmethod
    def _load_directory(
        cls,
        directory: Path,
        errors: List[dict[str, str]] | None = None,
    ) -> List[SkillManifest]:
        manifests: List[SkillManifest] = []
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / "SKILL.md"
            if not manifest_path.exists():
                continue
            manifest, error = cls._parse_manifest(manifest_path)
            if manifest:
                # Compute relative path from skills directory (not cwd)
                # Old behavior: try both cwd and directory
                # relative_path: Path | None = None
                # for base in (cwd, directory):
                #     try:
                #         relative_path = manifest_path.relative_to(base)
                #         break
                #     except ValueError:
                #         continue

                # New behavior: always relative to skills directory
                try:
                    relative_path = manifest_path.relative_to(directory)
                except ValueError:
                    relative_path = None

                manifest = replace(manifest, relative_path=relative_path)
                manifests.append(manifest)
            elif errors is not None:
                errors.append(
                    {
                        "path": str(manifest_path),
                        "error": error or "Failed to parse skill manifest",
                    }
                )
        return manifests

    @classmethod
    def _parse_manifest(cls, manifest_path: Path) -> tuple[SkillManifest | None, str | None]:
        try:
            post = frontmatter.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse skill manifest",
                data={"path": str(manifest_path), "error": str(exc)},
            )
            return None, str(exc)

        metadata = post.metadata or {}
        name = metadata.get("name")
        description = metadata.get("description")

        if not isinstance(name, str) or not name.strip():
            logger.warning("Skill manifest missing name", data={"path": str(manifest_path)})
            return None, "Missing 'name' field"
        if not isinstance(description, str) or not description.strip():
            logger.warning("Skill manifest missing description", data={"path": str(manifest_path)})
            return None, "Missing 'description' field"

        body_text = (post.content or "").strip()

        return SkillManifest(
            name=name.strip(),
            description=description.strip(),
            body=body_text,
            path=manifest_path,
        ), None


def format_skills_for_prompt(manifests: Sequence[SkillManifest]) -> str:
    """
    Format a collection of skill manifests into an XML-style block suitable for system prompts.
    """
    if not manifests:
        return ""

    preamble = (
        "Skills provide specialized capabilities and domain knowledge. Use a Skill if it seems in any way "
        "relevant to the Users task, intent or would increase effectiveness. \n"
        "The 'execute' tool gives you direct shell access to the current working directory (agent workspace) "
        "and outputted files are visible to the User.\n"
        "To use a Skill you must first read the SKILL.md file (use 'execute' tool).\n "
        "Only use skills listed in <available_skills> below.\n\n"
    )
    formatted_parts: List[str] = []

    for manifest in manifests:
        description = (manifest.description or "").strip()
        relative_path = manifest.relative_path
        path_attr = f' path="{relative_path}"' if relative_path is not None else ""
        if relative_path is None and manifest.path:
            path_attr = f' path="{manifest.path}"'

        block_lines: List[str] = [f'<agent-skill name="{manifest.name}"{path_attr}>']
        if description:
            block_lines.append(f"{description}")
        block_lines.append("</agent-skill>")
        formatted_parts.append("\n".join(block_lines))

    return "".join(
        (f"{preamble}<available_skills>\n", "\n".join(formatted_parts), "\n</available_skills>")
    )
