"""
Tool provider that exposes local skill manifests as tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Mapping

from mcp.types import CallToolResult, TextContent, Tool

from fast_agent.tools.providers.base import BaseToolProvider

if TYPE_CHECKING:
    from fast_agent.skills.registry import SkillManifest


@dataclass
class SkillRuntimeHandle:
    """Placeholder data for the skill execution runtime."""

    active: bool = False


class SkillToolProvider(BaseToolProvider):
    """Expose SkillManifest entries as callable tools for an agent."""

    name = "skills"

    def __init__(self, manifests: Iterable[SkillManifest] | None = None) -> None:
        self._manifests: Dict[str, SkillManifest] = {}
        self._manifest_order: List[str] = []
        self._runtime = SkillRuntimeHandle(active=False)
        self._execute_tool_name = "execute"
        self._execute_tool: Tool | None = None
        if manifests:
            self.set_manifests(manifests)

    @property
    def skill_count(self) -> int:
        return len(self._manifest_order)

    def set_manifests(self, manifests: Iterable[SkillManifest]) -> None:
        """Replace the current manifests with a new collection."""
        self._manifests.clear()
        self._manifest_order = []
        for manifest in manifests:
            self._manifests[manifest.name] = manifest
            self._manifest_order.append(manifest.name)

    def get_manifest(self, name: str) -> SkillManifest | None:
        """Return the manifest associated with a skill name, if available."""
        return self._manifests.get(name)

    def iter_manifests(self) -> Iterator[SkillManifest]:
        """Iterate over manifests in registration order."""
        for name in self._manifest_order:
            yield self._manifests[name]

    async def list_tools(self) -> List[Tool]:
        tools: List[Tool] = []
        for name in self._manifest_order:
            manifest = self._manifests[name]
            meta: Mapping[str, Any] = {
                "fast-agent/skillPath": str(manifest.path),
            }
            tools.append(
                Tool(
                    name=manifest.name,
                    description=manifest.description,
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    _meta=meta,
                )
            )

        if self._runtime.active:
            execute_tool = self._get_or_create_execute_tool()
            tools.append(execute_tool)

        return tools

    def can_handle_tool(self, tool_name: str) -> bool:
        if tool_name in self._manifests:
            return True
        if self._runtime.active and tool_name == self._execute_tool_name:
            return True
        return False

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any] | None = None
    ) -> CallToolResult:
        if tool_name in self._manifests:
            manifest = self._manifests[tool_name]
            response_text = manifest.body or manifest.description
            self._runtime.active = True
            return CallToolResult(
                isError=False,
                content=[
                    TextContent(type="text", text=response_text),
                ],
            )

        if tool_name == self._execute_tool_name and self._runtime.active:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text="Execute tool runtime is not yet implemented for agent skills.",
                    )
                ],
            )

        return CallToolResult(
            isError=True,
            content=[
                TextContent(type="text", text=f"Skill tool '{tool_name}' is not available."),
            ],
        )

    def summary_counts(self) -> Dict[str, int]:
        return {"skills": self.skill_count}

    def _get_or_create_execute_tool(self) -> Tool:
        if self._execute_tool is None:
            self._execute_tool = Tool(
                name=self._execute_tool_name,
                description="Execute shell commands in the agent skill runtime.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute inside the skill runtime shell.",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            )
        return self._execute_tool
