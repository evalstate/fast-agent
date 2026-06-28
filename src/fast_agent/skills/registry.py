from __future__ import annotations

from dataclasses import dataclass
from html import escape as escape_xml_text
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

from fast_agent.core.logging.logger import get_logger
from fast_agent.paths import default_skill_paths
from fast_agent.skills.models import SKILL_MANIFEST_FILENAME
from fast_agent.tools.skill_reader import READ_SKILL_TOOL_NAME
from fast_agent.utils.text import strip_casefold, strip_str_to_none, strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

SKILL_RESOURCE_DIRECTORIES = ("scripts", "references", "assets")


@dataclass(frozen=True)
class SkillManifest:
    """Represents a single skill description loaded from SKILL.md."""

    name: str
    description: str
    body: str
    path: Path  # Absolute path to SKILL.md
    # Optional fields from the Agent Skills specification
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: list[str] | None = None


class SkillRegistry:
    """Simple registry that resolves skills directories and parses manifests."""

    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        directories: Sequence[Path | str] | None = None,
    ) -> None:
        self._base_dir = base_dir or Path.cwd()
        self._directories: list[Path] = []
        self._errors: list[dict[str, str]] = []
        self._warnings: list[str] = []
        self._missing_directories: list[Path] = []

        self._configure_directories(directories)

    @property
    def directories(self) -> list[Path]:
        return list(self._directories)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def load_manifests(self) -> list[SkillManifest]:
        """Load all skill manifests from the configured directories.

        Returns manifests with absolute paths per Agent Skills specification.
        """
        self._errors = []
        self._warnings = [
            f"Skills directory not found: {path}" for path in self._missing_directories
        ]
        if not self._directories:
            return []
        manifests_by_name: dict[str, SkillManifest] = {}
        for directory in self._directories:
            for manifest in self._load_directory(directory, self._errors):
                key = strip_casefold(manifest.name)
                if key in manifests_by_name:
                    prior = manifests_by_name[key]
                    warning = (
                        f"Duplicate skill '{manifest.name}' from {manifest.path} overrides "
                        f"{prior.path}"
                    )
                    self._warnings.append(warning)
                    logger.warning("Duplicate skill manifest", data={"warning": warning})
                manifests_by_name.pop(key, None)
                manifests_by_name[key] = manifest
        return list(manifests_by_name.values())

    def load_manifests_with_errors(self) -> tuple[list[SkillManifest], list[dict[str, str]]]:
        manifests = self.load_manifests()
        return manifests, list(self._errors)

    @property
    def errors(self) -> list[dict[str, str]]:
        return list(self._errors)

    def _resolve_directory(self, directory: Path) -> Path:
        if directory.is_absolute():
            return directory
        return (self._base_dir / directory).resolve()

    def _configure_directories(self, directories: Sequence[Path | str] | None) -> None:
        self._warnings = []
        self._missing_directories = []
        self._directories = []
        default_entries = {path.resolve() for path in default_skill_paths(cwd=self._base_dir)}
        if directories is None:
            entries = default_skill_paths(cwd=self._base_dir)
        else:
            entries = list(directories)

        for entry in entries:
            raw_path = Path(entry) if isinstance(entry, str) else entry
            resolved = self._resolve_directory(raw_path)
            if resolved.exists() and resolved.is_dir():
                self._directories.append(resolved)
            elif directories is not None:
                if resolved in default_entries:
                    logger.debug(
                        "Skills directory not found",
                        data={"directory": str(resolved), "optional": True},
                    )
                else:
                    self._missing_directories.append(resolved)
                    logger.warning(
                        "Skills directory not found",
                        data={"directory": str(resolved)},
                    )

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
        """Load manifests from a directory, using absolute paths."""
        manifests: list[SkillManifest] = []
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / SKILL_MANIFEST_FILENAME
            if not manifest_path.exists():
                continue
            manifest, error = cls._parse_manifest(manifest_path)
            if manifest:
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
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "Failed to read skill manifest",
                data={"path": str(manifest_path), "error": str(exc)},
            )
            return None, str(exc)
        return cls._parse_manifest_content(manifest_text, manifest_path)

    @classmethod
    def parse_manifest_text(
        cls,
        manifest_text: str,
        *,
        path: Path | None = None,
    ) -> tuple[SkillManifest | None, str | None]:
        manifest_path = path or Path("<in-memory>")
        return cls._parse_manifest_content(manifest_text, manifest_path)

    @classmethod
    def _parse_manifest_content(
        cls,
        manifest_text: str,
        manifest_path: Path,
    ) -> tuple[SkillManifest | None, str | None]:
        try:
            post = frontmatter.loads(manifest_text)
        except Exception as exc:
            logger.warning(
                "Failed to parse skill manifest",
                data={"path": str(manifest_path), "error": str(exc)},
            )
            return None, str(exc)

        metadata = post.metadata or {}
        name = metadata.get("name")
        description = metadata.get("description")

        normalized_name = strip_str_to_none(name)
        normalized_description = strip_str_to_none(description)

        if normalized_name is None:
            logger.warning("Skill manifest missing name", data={"path": str(manifest_path)})
            return None, "Missing 'name' field"
        if normalized_description is None:
            logger.warning("Skill manifest missing description", data={"path": str(manifest_path)})
            return None, "Missing 'description' field"

        body_text = strip_to_none(post.content) or ""

        return SkillManifest(
            name=normalized_name,
            description=normalized_description,
            body=body_text,
            path=manifest_path,
            license=_optional_str(metadata.get("license")),
            compatibility=_optional_str(metadata.get("compatibility")),
            metadata=_string_metadata(metadata.get("metadata")),
            allowed_tools=_allowed_tools(metadata.get("allowed-tools")),
        ), None


def format_skills_for_prompt(
    manifests: Sequence[SkillManifest],
    *,
    read_tool_name: str = READ_SKILL_TOOL_NAME,
    include_preamble: bool = True,
) -> str:
    """
    Format skill manifests into XML block per the Agent Skills specification.

    Uses the standard format from https://agentskills.io with absolute paths:
    <skill>
      <name>skill-name</name>
      <description>Brief capability summary</description>
      <location>/absolute/path/to/SKILL.md</location>
      <directory>/absolute/path/to/skill-name</directory>
    </skill>

    Args:
        manifests: Collection of skill manifests to format
        read_tool_name: Name of the tool used to read skill files (for preamble)
        include_preamble: Whether to include instructional preamble text
    """
    if not manifests:
        return ""

    formatted_parts: list[str] = []

    for manifest in manifests:
        skill_dir = manifest.path.parent
        lines: list[str] = ["<skill>"]
        lines.append(_xml_element("name", manifest.name))

        description = strip_to_none(manifest.description)
        if description is not None:
            lines.append(_xml_element("description", description))

        # Use absolute path per Agent Skills specification
        lines.append(_xml_element("location", str(manifest.path)))
        lines.append(_xml_element("directory", str(skill_dir)))

        for tag_name in SKILL_RESOURCE_DIRECTORIES:
            subdir = skill_dir / tag_name
            if subdir.is_dir():
                lines.append(_xml_element(tag_name, str(subdir)))

        lines.append("</skill>")
        formatted_parts.append("\n".join(lines))

    skills_xml = "<available_skills>\n" + "\n".join(formatted_parts) + "\n</available_skills>"

    if not include_preamble:
        return skills_xml

    preamble = (
        "Skills provide specialized capabilities and domain knowledge. Use a Skill if it seems "
        "relevant to the user's task, intent, or would increase your effectiveness.\n"
        f"To use a Skill, read its SKILL.md file from the specified location using the '{read_tool_name}' tool.\n"
        "Prefer that file-reading tool over shell commands when loading skill content or "
        "skill resources.\n"
        "The <location> value is the absolute path to the skill's SKILL.md file, and "
        "<directory> is the resolved absolute path to the skill's root directory.\n"
        f"When present, {_skill_resource_directory_tags()} provide resolved absolute paths "
        "for standard skill resource directories.\n"
        "When a skill references relative paths, resolve them against the skill's "
        "directory (the parent of SKILL.md) and use absolute paths in tool calls.\n"
        "Only use Skills listed in <available_skills> below.\n\n"
    )

    return preamble + skills_xml


def _xml_element(tag_name: str, value: str) -> str:
    return f"  <{tag_name}>{escape_xml_text(value)}</{tag_name}>"


def _skill_resource_directory_tags() -> str:
    quoted_tags = [f"<{tag_name}>" for tag_name in SKILL_RESOURCE_DIRECTORIES]
    return f"{', '.join(quoted_tags[:-1])}, and {quoted_tags[-1]}"


def _optional_str(value: object) -> str | None:
    return strip_str_to_none(value)


def _allowed_tools(value: object) -> list[str] | None:
    if not isinstance(value, str):
        return None
    tools = value.split()
    return tools or None


def _string_metadata(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): str(item) for key, item in value.items()}
