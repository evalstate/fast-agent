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

if TYPE_CHECKING:
    from fast_agent.skills.registry import SkillManifest


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

        self._tool = Tool(
            name="read_skill",
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
        )

    @property
    def tool(self) -> Tool:
        """Get the read_skill tool definition."""
        return self._tool

    @property
    def enabled(self) -> bool:
        """Whether the skill reader is enabled (has skills available)."""
        return len(self._skill_manifests) > 0

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
        path_str = (arguments or {}).get("path") if arguments else None
        if not isinstance(path_str, str) or not path_str.strip():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text="The read_skill tool requires a 'path' string argument.",
                    )
                ],
            )

        path = Path(path_str.strip())

        # Security: ensure path is absolute
        if not path.is_absolute():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text="Path must be absolute. Use the path from <location> in available_skills.",
                    )
                ],
            )

        # Security: ensure path is within an allowed skill directory
        if not self._is_path_allowed(path):
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Access denied: {path} is not within an allowed skill directory.",
                    )
                ],
            )

        # Check file exists
        if not path.exists():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"File not found: {path}",
                    )
                ],
            )

        if not path.is_file():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Path is not a file: {path}",
                    )
                ],
            )

        try:
            content = path.read_text(encoding="utf-8")
            self._logger.debug(f"Read skill file: {path} ({len(content)} bytes)")

            return CallToolResult(
                isError=False,
                content=[
                    TextContent(
                        type="text",
                        text=content,
                    )
                ],
            )
        except Exception as exc:
            self._logger.error(f"Failed to read skill file: {exc}")
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error reading file: {exc}",
                    )
                ],
            )
