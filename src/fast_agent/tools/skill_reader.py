"""
Skill Reader Tool - Provides file reading capabilities for Agent Skills.

This tool follows the Agent Skills standard (https://agentskills.io/integrate-skills.md)
for filesystem-based agents. It allows the agent to:
1. Read SKILL.md files to activate skills
2. Read resources within skill directories (scripts/, references/, assets/)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mcp.types import CallToolResult, TextContent, Tool

if TYPE_CHECKING:
    from fast_agent.skills import SkillManifest


class SkillReader:
    """Tool for reading skill files and resources."""

    def __init__(
        self,
        skill_manifests: list["SkillManifest"],
        *,
        max_file_size: int = 1024 * 1024,  # 1MB default limit
    ) -> None:
        """
        Initialize the skill reader.

        Args:
            skill_manifests: List of available skill manifests
            max_file_size: Maximum file size to read in bytes
        """
        self._skill_manifests = skill_manifests
        self._max_file_size = max_file_size

        # Build a map of skill name -> skill directory
        self._skill_dirs: dict[str, Path] = {}
        for manifest in skill_manifests:
            if manifest.path:
                self._skill_dirs[manifest.name] = manifest.path.parent

        self._tool = Tool(
            name="read_skill",
            description=(
                "Read a file from a skill directory. Use this to activate a skill by reading "
                "its SKILL.md file, or to access skill resources (scripts, references, assets). "
                "The path should be relative to the skill's directory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill (as listed in available_skills)",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the file relative to the skill directory. "
                            "Use 'SKILL.md' to read the skill instructions, or paths like "
                            "'scripts/example.py' or 'references/guide.md' for resources."
                        ),
                        "default": "SKILL.md",
                    },
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
        )

    @property
    def tool(self) -> Tool:
        """Get the MCP Tool definition."""
        return self._tool

    @property
    def enabled(self) -> bool:
        """Check if the skill reader is enabled (has skills available)."""
        return bool(self._skill_manifests)

    async def execute(self, arguments: dict | None = None) -> CallToolResult:
        """
        Execute the read_skill tool.

        Args:
            arguments: Tool arguments containing skill_name and optional path

        Returns:
            CallToolResult with file contents or error
        """
        args = arguments or {}
        skill_name = args.get("skill_name")
        file_path = args.get("path", "SKILL.md")

        if not skill_name:
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text="skill_name is required")],
            )

        if skill_name not in self._skill_dirs:
            available = ", ".join(sorted(self._skill_dirs.keys()))
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Unknown skill: {skill_name}. Available skills: {available}",
                    )
                ],
            )

        skill_dir = self._skill_dirs[skill_name]
        target_path = (skill_dir / file_path).resolve()

        # Security: ensure the resolved path is within the skill directory
        try:
            target_path.relative_to(skill_dir.resolve())
        except ValueError:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Access denied: path '{file_path}' is outside the skill directory",
                    )
                ],
            )

        if not target_path.exists():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"File not found: {file_path} in skill '{skill_name}'",
                    )
                ],
            )

        if not target_path.is_file():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Not a file: {file_path}",
                    )
                ],
            )

        # Check file size
        file_size = target_path.stat().st_size
        if file_size > self._max_file_size:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"File too large: {file_size} bytes (max {self._max_file_size})",
                    )
                ],
            )

        try:
            content = target_path.read_text(encoding="utf-8")
            return CallToolResult(
                isError=False,
                content=[TextContent(type="text", text=content)],
            )
        except UnicodeDecodeError:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Cannot read file: {file_path} is not a text file",
                    )
                ],
            )
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Error reading file: {e}")],
            )


def get_skill_reader_tool(skill_manifests: list["SkillManifest"]) -> Tool | None:
    """
    Get the read_skill tool if skills are available.

    Args:
        skill_manifests: List of available skill manifests

    Returns:
        Tool definition or None if no skills
    """
    if not skill_manifests:
        return None
    reader = SkillReader(skill_manifests)
    return reader.tool
