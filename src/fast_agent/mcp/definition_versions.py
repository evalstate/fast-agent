"""Advisory MCP definition versions (Hugging Face prototype of the Definition Versions SEP).

Servers advertise digests of their tool list and instructions; clients echo the versions
they hold on ``tools/call`` and the server rejects stale calls before executing them.
Keys are provisional: keep them here so adopting the final SEP names is a one-line change.
"""

from collections.abc import Mapping
from typing import Any, Literal

from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

DEFINITION_VERSIONS_META = "huggingface.co/definition-versions"
KNOWN_DEFINITION_VERSIONS_META = "huggingface.co/known-definition-versions"
DEFINITION_VERSION_MISMATCH = -32987

DefinitionTarget = Literal["tools", "instructions"]


class DefinitionVersions(BaseModel):
    tools: str | None = None
    instructions: str | None = None

    def known(self, *, instructions: bool) -> dict[str, str]:
        """Versions to echo on tools/call; instructions only when we render them."""
        known = {"tools": self.tools} if self.tools else {}
        if instructions and self.instructions:
            known["instructions"] = self.instructions
        return known

    def validates(self, *, instructions: bool) -> bool:
        """Whether every definition we cache is covered by a server digest."""
        return bool(self.tools) and (not instructions or bool(self.instructions))


def parse_definition_versions(meta: Mapping[str, Any] | None) -> DefinitionVersions:
    """Read advertised versions from result ``_meta`` (external data: ignore bad shapes)."""
    value = (meta or {}).get(DEFINITION_VERSIONS_META)
    if not isinstance(value, Mapping):
        return DefinitionVersions()
    tools = value.get("tools")
    instructions = value.get("instructions")
    return DefinitionVersions(
        tools=tools if isinstance(tools, str) else None,
        instructions=instructions if isinstance(instructions, str) else None,
    )


def stale_definitions(error: BaseException) -> frozenset[DefinitionTarget] | None:
    """Targets a definition-version mismatch names, or None for any other error."""
    if not isinstance(error, MCPError) or error.code != DEFINITION_VERSION_MISMATCH:
        return None
    data = error.data
    named = data.get("stale") if isinstance(data, Mapping) else None
    stale: set[DefinitionTarget] = set()
    for target in named if isinstance(named, list) else []:
        if target == "tools":
            stale.add("tools")
        elif target == "instructions":
            stale.add("instructions")
    # An unspecific mismatch still means our view is stale: refresh everything.
    return frozenset(stale) or frozenset({"tools", "instructions"})
