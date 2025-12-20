from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import frontmatter

from fast_agent.constants import DEFAULT_SKILLS_PATHS
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


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
                key = manifest.name.lower()
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
        entries = DEFAULT_SKILLS_PATHS if directories is None else list(directories)

        for entry in entries:
            raw_path = Path(entry) if isinstance(entry, str) else entry
            resolved = self._resolve_directory(raw_path)
            if resolved.exists() and resolved.is_dir():
                self._directories.append(resolved)
            elif directories is not None:
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
            manifest_path = entry / "SKILL.md"
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

        # Parse optional fields per Agent Skills specification
        license_field = metadata.get("license")
        compatibility = metadata.get("compatibility")
        custom_metadata = metadata.get("metadata")
        allowed_tools_raw = metadata.get("allowed-tools")

        # Parse allowed-tools as space-delimited list
        allowed_tools: list[str] | None = None
        if isinstance(allowed_tools_raw, str) and allowed_tools_raw.strip():
            allowed_tools = allowed_tools_raw.split()

        # Validate metadata is a dict if present
        if custom_metadata is not None and not isinstance(custom_metadata, dict):
            custom_metadata = None

        return SkillManifest(
            name=name.strip(),
            description=description.strip(),
            body=body_text,
            path=manifest_path,
            license=license_field.strip() if isinstance(license_field, str) else None,
            compatibility=compatibility.strip() if isinstance(compatibility, str) else None,
            metadata=custom_metadata,
            allowed_tools=allowed_tools,
        ), None


def format_skills_for_prompt(
    manifests: Sequence[SkillManifest],
    *,
    read_tool_name: str = "read_skill",
    include_preamble: bool = True,
) -> str:
    """
    Format skill manifests into XML block per the Agent Skills specification.

    Uses the standard format from https://agentskills.io with absolute paths:
    <skill>
      <name>skill-name</name>
      <description>Brief capability summary</description>
      <location>/absolute/path/to/SKILL.md</location>
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
        lines: list[str] = ["<skill>"]
        lines.append(f"  <name>{manifest.name}</name>")

        description = (manifest.description or "").strip()
        if description:
            lines.append(f"  <description>{description}</description>")

        # Use absolute path per Agent Skills specification
        lines.append(f"  <location>{manifest.path}</location>")

        lines.append("</skill>")
        formatted_parts.append("\n".join(lines))

    skills_xml = "<available_skills>\n" + "\n".join(formatted_parts) + "\n</available_skills>"

    if not include_preamble:
        return skills_xml

    preamble = (
        "Skills provide specialized capabilities and domain knowledge. Use a Skill if it seems "
        "relevant to the user's task, intent, or would increase your effectiveness.\n"
        f"To use a Skill, read its SKILL.md file from the specified location using the '{read_tool_name}' tool.\n"
        "Only use Skills listed in <available_skills> below.\n\n"
    )

    return preamble + skills_xml
