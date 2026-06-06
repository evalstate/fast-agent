"""
SkillReader - Read skill files for non-ACP contexts.

This provides a dedicated 'read_skill' tool for reading SKILL.md files and
associated resources when not running in an ACP context (where read_text_file
is provided by ACPFilesystemRuntime).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, TextContent, Tool

from fast_agent.tools.filesystem_tool_args import (
    coerce_required_string_argument,
    coerce_tool_arguments,
)
from fast_agent.tools.tool_sources import SKILL_TOOL_SOURCE, set_tool_source

if TYPE_CHECKING:
    from fast_agent.skills.registry import SkillManifest

READ_SKILL_TOOL_NAME = "read_skill"


class SkillReader:
    """Provides the read_skill tool for reading skill files in non-ACP contexts."""

    def __init__(
        self,
        skill_manifests: list[SkillManifest],
        logger,
    ) -> None:
        """
        Initialize the skill reader.

        Args:
            skill_manifests: List of available skill manifests (for path validation)
            logger: Logger instance for debugging
        """
        self._skill_manifests = skill_manifests
        self._logger = logger

        # Build set of allowed skill directories for security
        self._allowed_directories: set[Path] = set()
        for manifest in skill_manifests:
            if manifest.path:
                # Allow reading from the skill's directory and subdirectories
                self._allowed_directories.add(manifest.path.parent.resolve())

        self._tool = set_tool_source(
            Tool(
                name=READ_SKILL_TOOL_NAME,
                description=(
                    "Read a skill's SKILL.md file or associated resources. "
                    "Use this to load skill instructions before using the skill."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to read (from the <location> in available_skills)",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            SKILL_TOOL_SOURCE,
        )

    @property
    def tool(self) -> Tool:
        """Get the read_skill tool definition."""
        return self._tool

    @property
    def enabled(self) -> bool:
        """Whether the skill reader is enabled (has skills available)."""
        return bool(self._skill_manifests)

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if the path is within an allowed skill directory."""
        resolved = path.resolve()
        for allowed_dir in self._allowed_directories:
            try:
                resolved.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
        return False

    async def execute(self, arguments: dict[str, Any] | None = None) -> CallToolResult:
        """Read a skill file."""
        path, validation_error = self._validated_skill_path(arguments)
        if validation_error is not None:
            return validation_error

        try:
            content = path.read_text(encoding="utf-8")
            self._logger.debug(f"Read skill file: {path} ({len(content)} bytes)")
            return _text_result(content, is_error=False)
        except Exception as exc:
            self._logger.error(f"Failed to read skill file: {exc}")
            return _text_result(f"Error reading file: {exc}", is_error=True)

    def _validated_skill_path(
        self,
        arguments: dict[str, Any] | None,
    ) -> tuple[Path, None] | tuple[Path, CallToolResult]:
        try:
            payload = {} if arguments is None else coerce_tool_arguments(arguments)
            path_str = coerce_required_string_argument(
                payload.get("path"),
                "path",
                strip=True,
            )
        except ValueError as exc:
            return Path(), _text_result(
                str(exc),
                is_error=True,
            )

        path = Path(path_str)
        error_message = self._skill_path_error_message(path)
        if error_message is not None:
            return path, _text_result(error_message, is_error=True)
        return path, None

    def _skill_path_error_message(self, path: Path) -> str | None:
        if not path.is_absolute():
            return "Path must be absolute. Use the path from <location> in available_skills."
        if not self._is_path_allowed(path):
            return f"Access denied: {path} is not within an allowed skill directory."
        if not path.exists():
            return f"File not found: {path}"
        if not path.is_file():
            return f"Path is not a file: {path}"
        return None


def _text_result(text: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        isError=is_error,
        content=[TextContent(type="text", text=text)],
    )
