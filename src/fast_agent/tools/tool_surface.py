"""How an agent's tools reach the model.

``list_tools()`` returns the tool definitions sent to the model. With programmatic tool calling
(PTC) most tools are not in that list: they are Python functions described inside the code tool.
``ToolSurface`` records every tool the model can reach and how it reaches it, so tool listings
can show what is in context and why. Future exposure kinds (for example tools that are only
discoverable through search) extend ``ToolExposure``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp_types import ListToolsResult, Tool

CHARS_PER_TOKEN = 4
"""Rough characters-per-token ratio used for context estimates."""

LARGE_STUB_TOKENS = 500
"""Programmatic function stubs at or above this estimate are flagged as large."""


class ToolExposure(StrEnum):
    MODEL = "model"
    """The tool definition is sent to the model with each request."""

    PROGRAMMATIC = "programmatic"
    """Callable only from the PTC code tool; its stub is part of that tool's description."""


def estimate_tokens(text: str) -> int:
    """Return a rough token estimate for ``text``."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def tool_definition_tokens(tool: Tool) -> int:
    """Estimate the context used by a tool definition sent to the model."""
    schema = json.dumps(tool.input_schema, separators=(",", ":"), ensure_ascii=False)
    return estimate_tokens(f"{tool.name}{tool.description or ''}{schema}")


@dataclass(frozen=True, slots=True)
class SurfaceTool:
    tool: Tool
    exposure: ToolExposure
    context_tokens: int
    """Estimated tokens this tool adds to the model context."""
    direct_reason: str | None = None
    """Why a tool stays directly callable while PTC is enabled."""
    python_name: str | None = None
    signature: str | None = None
    stub: str | None = None

    @property
    def large(self) -> bool:
        return (
            self.exposure is ToolExposure.PROGRAMMATIC and self.context_tokens >= LARGE_STUB_TOKENS
        )

    def matches(self, name: str) -> bool:
        return name in (self.tool.name, self.python_name)


@dataclass(frozen=True, slots=True)
class ProgrammaticSurface:
    """Settings and model-facing instructions of the PTC code tool."""

    tool_name: str
    tool_format: Literal["json", "freeform"]
    instructions: str
    type_check: bool
    max_duration_seconds: float
    max_memory_mb: int


@dataclass(frozen=True, slots=True)
class ToolSurface:
    tools: tuple[SurfaceTool, ...]
    programmatic: ProgrammaticSurface | None = None

    @classmethod
    def direct(cls, tools: Sequence[Tool]) -> ToolSurface:
        """Build the surface of an agent whose tools are all sent to the model."""
        return cls(
            tools=tuple(
                SurfaceTool(
                    tool=tool,
                    exposure=ToolExposure.MODEL,
                    context_tokens=tool_definition_tokens(tool),
                )
                for tool in tools
            )
        )

    def exposed(self, exposure: ToolExposure) -> list[SurfaceTool]:
        return [tool for tool in self.tools if tool.exposure is exposure]

    def find(self, name: str) -> SurfaceTool | None:
        """Find a tool by its tool name or, for programmatic tools, its Python name."""
        return next((tool for tool in self.tools if tool.matches(name)), None)


@runtime_checkable
class ToolSurfaceProvider(Protocol):
    """Optional capability for agents whose callable tools differ from their model tools."""

    async def tool_surface(self) -> ToolSurface: ...


class ToolLister(Protocol):
    async def list_tools(self) -> ListToolsResult: ...


async def resolve_tool_surface(agent: ToolLister) -> ToolSurface:
    """Return the agent's tool surface, treating plain agents as all-direct."""
    if isinstance(agent, ToolSurfaceProvider):
        return await agent.tool_surface()
    return ToolSurface.direct((await agent.list_tools()).tools)
