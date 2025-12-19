from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import frontmatter

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)

# Agent Skills name validation pattern per https://agentskills.io/specification.md
# Lowercase letters, numbers, hyphens only; max 64 chars; no start/end hyphens; no consecutive hyphens
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_MAX_SKILL_NAME_LENGTH = 64


def _validate_skill_name(name: str, manifest_path: Path) -> list[str]:
    """
    Validate a skill name against the Agent Skills specification.

    Returns a list of warning messages (empty if valid).
    Per spec: lowercase letters, numbers, hyphens only; max 64 chars;
    no start/end hyphens; no consecutive hyphens; must match parent directory.
    """
    warnings: list[str] = []
    parent_dir_name = manifest_path.parent.name

    if len(name) > _MAX_SKILL_NAME_LENGTH:
        warnings.append(
            f"Skill name '{name}' exceeds max length of {_MAX_SKILL_NAME_LENGTH} characters"
        )

    if not _SKILL_NAME_PATTERN.match(name):
        warnings.append(
            f"Skill name '{name}' should contain only lowercase letters, numbers, "
            "and hyphens, and must not start or end with a hyphen"
        )

    if "--" in name:
        warnings.append(f"Skill name '{name}' should not contain consecutive hyphens")

    if name != parent_dir_name:
        warnings.append(
            f"Skill name '{name}' does not match parent directory '{parent_dir_name}'"
        )

    return warnings


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
        self._original_override_directory: Path | None = None  # Store original before resolution
        self._override_failed: bool = False
        self._errors: list[dict[str, str]] = []
        if override_directory:
            self._original_override_directory = override_directory
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

    def load_manifests(self) -> list[SkillManifest]:
        self._errors = []
        if not self._directory:
            return []
        manifests = self._load_directory(self._directory, self._errors)

        # Recompute relative paths to be from base_dir (workspace root) instead of skills directory
        adjusted_manifests: list[SkillManifest] = []
        for manifest in manifests:
            try:
                relative_path = manifest.path.relative_to(self._base_dir)
                adjusted_manifest = replace(manifest, relative_path=relative_path)
                adjusted_manifests.append(adjusted_manifest)
            except ValueError:
                # Path is outside workspace - compute relative to skills directory
                # and prepend the original override path (e.g., ../skills/my-skill/SKILL.md)
                if self._original_override_directory is not None:
                    try:
                        skill_relative = manifest.path.relative_to(self._directory)
                        relative_path = self._original_override_directory / skill_relative
                        adjusted_manifest = replace(manifest, relative_path=relative_path)
                    except ValueError:
                        # Fallback to absolute path if we can't compute relative
                        adjusted_manifest = replace(manifest, relative_path=None)
                else:
                    adjusted_manifest = replace(manifest, relative_path=None)
                adjusted_manifests.append(adjusted_manifest)

        return adjusted_manifests

    def load_manifests_with_errors(self) -> tuple[list[SkillManifest], list[dict[str, str]]]:
        manifests = self.load_manifests()
        return manifests, list(self._errors)

    @property
    def errors(self) -> list[dict[str, str]]:
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
    def load_directory(cls, directory: Path) -> list[SkillManifest]:
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
    ) -> tuple[list[SkillManifest], list[dict[str, str]]]:
        errors: list[dict[str, str]] = []
        manifests = cls._load_directory(directory, errors)
        return manifests, errors

    @classmethod
    def _load_directory(
        cls,
        directory: Path,
        errors: list[dict[str, str]] | None = None,
    ) -> list[SkillManifest]:
        manifests: list[SkillManifest] = []
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

        name = name.strip()

        # Validate name against Agent Skills specification (warnings only, not errors)
        name_warnings = _validate_skill_name(name, manifest_path)
        for warning in name_warnings:
            logger.warning(warning, data={"path": str(manifest_path)})

        body_text = (post.content or "").strip()

        return SkillManifest(
            name=name,
            description=description.strip(),
            body=body_text,
            path=manifest_path,
        ), None


def format_skills_for_prompt(
    manifests: Sequence[SkillManifest],
    *,
    has_read_tool: bool = False,
) -> str:
    """
    Format a collection of skill manifests into an XML-style block suitable for system prompts.

    Args:
        manifests: Collection of skill manifests to format
        has_read_tool: If True, indicates a dedicated read_text_file tool is available
                      (e.g., in ACP context). If False, assumes shell execute is available.

    Per the Agent Skills standard (https://agentskills.io/integrate-skills.md),
    filesystem-based agents should use absolute paths to SKILL.md files.
    """
    if not manifests:
        return ""

    # Context-aware preamble based on available tools
    if has_read_tool:
        read_instruction = "To use a Skill you must first read the SKILL.md file using the 'read_text_file' tool."
    else:
        read_instruction = (
            "To use a Skill you must first read the SKILL.md file "
            "(use 'execute' tool with cat or similar)."
        )

    preamble = (
        "Skills provide specialized capabilities and domain knowledge. Use a Skill if it seems in any way "
        "relevant to the Users task, intent or would increase your effectiveness.\n"
        f"{read_instruction}\n"
        "Paths in Skill documentation are relative to that Skill's directory, NOT the workspace root.\n"
        "For example if the 'test' skill has scripts/example.py access it with <skill_folder>/scripts/example.py.\n"
        "Only use Skills listed in <available_skills> below.\n\n"
    )
    formatted_parts: list[str] = []

    for manifest in manifests:
        description = (manifest.description or "").strip()
        # Per Agent Skills standard: use absolute paths for filesystem-based agents
        # manifest.path is always the absolute path to SKILL.md
        skill_path = manifest.path
        path_attr = f' path="{skill_path}"' if skill_path else ""

        block_lines: list[str] = [f'<agent-skill name="{manifest.name}"{path_attr}>']
        if description:
            block_lines.append(f"{description}")
        block_lines.append("</agent-skill>")
        formatted_parts.append("\n".join(block_lines))

    return "".join(
        (f"{preamble}<available_skills>\n", "\n".join(formatted_parts), "\n</available_skills>")
    )
