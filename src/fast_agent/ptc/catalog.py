"""The PTC function catalogue: stable names, sources and definition visibility.

``ProgrammaticToolRuntime`` rebuilds a ``PtcCatalog`` whenever the agent's tools are listed.
Each sandbox run captures one catalogue, so a tool list refresh (``/mcp connect``, a
``tools/list_changed`` notification) never changes the functions a running program sees.
``PtcNameLedger`` keeps every Python name assigned during a conversation bound to the same tool,
so a newly connected tool can neither take an existing name nor inherit a removed tool's name.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

from fast_agent.mcp.common import get_resource_name, get_server_name, is_namespaced_name
from fast_agent.ptc.vocabulary import is_builtin_tool

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from mcp_types import Tool

    from fast_agent.ptc.search import SearchDocument
    from fast_agent.ptc.stubs import PtcFunction

PTC_SEARCH_EXTRA_THRESHOLD: Final = 3
"""Tool search starts deferring definitions when more extra functions than this are present."""
BUILTIN_SOURCE: Final = "builtin"
LOCAL_SOURCE: Final = "local"
CARD_SOURCE: Final = "card"
AGENTS_SOURCE: Final = "agents"
TOOL_SOURCES_TAG: Final = "fastagent:tool-sources"
TOOL_SOURCES_FINGERPRINT_META: Final = "fast_agent_tool_sources_fingerprint"
"""Text block metadata recording which source roster a user message carried."""
_MAX_ROSTER_ENTRIES = 40
_ROSTER_HEADER = (
    "Searchable tool sources in `exec_sandbox` (discover signatures with "
    '`await search_tools(query="...", server="...")`):'
)


class ToolIdentity(NamedTuple):
    """Source-qualified identity of a tool, independent of its Python name."""

    kind: Literal["local", "mcp"]
    server: str
    name: str


class ToolRoute(NamedTuple):
    """Where a function aliased away from its tool name dispatches: the tool and its origin."""

    name: str
    local: bool


def tool_identity(tool_name: str) -> ToolIdentity:
    if is_namespaced_name(tool_name):
        return ToolIdentity("mcp", get_server_name(tool_name), get_resource_name(tool_name))
    return ToolIdentity("local", "", tool_name)


def function_source(tool: Tool, overrides: Mapping[str, str]) -> str:
    """The source group a function is listed and searched under."""
    if is_builtin_tool(tool):
        return BUILTIN_SOURCE
    override = overrides.get(tool.name)
    if override is not None:
        return override
    if is_namespaced_name(tool.name):
        return get_server_name(tool.name)
    return LOCAL_SOURCE


class PtcNameLedger:
    """Python names assigned to tools during one conversation.

    A name stays bound to its tool's identity after the tool disappears, so the name is never
    reassigned: old code (or old definitions in history) cannot reach a different tool.
    """

    def __init__(self) -> None:
        self._names: dict[ToolIdentity, str] = {}
        self._identities: dict[str, ToolIdentity] = {}

    def allocate(
        self,
        identity: ToolIdentity,
        candidate: str,
        taken: Collection[str],
        reserved: Collection[str] = (),
    ) -> str:
        """Return ``identity``'s name, assigning ``candidate`` or a free variant on first use.

        ``taken`` holds names already used in the catalogue being built; ``reserved`` names
        are sandbox globals a function must not shadow (they can change between builds).
        """
        assigned = self._names.get(identity)
        if assigned is not None and assigned not in taken and assigned not in reserved:
            return assigned
        name = candidate
        suffix = 2
        while (
            name in taken
            or name in reserved
            or (name in self._identities and self._identities[name] != identity)
        ):
            name = f"{candidate}_{suffix}"
            suffix += 1
        self._names[identity] = name
        self._identities[name] = identity
        return name

    def retired(self, current: Collection[str]) -> frozenset[str]:
        """Names assigned earlier in the conversation that are not in ``current``."""
        return frozenset(name for name in self._identities if name not in current)

    def clear(self) -> None:
        self._names.clear()
        self._identities.clear()


@dataclass(frozen=True, slots=True)
class SourceEntry:
    name: str
    functions: int
    note: str | None = None
    """Why a configured source has no searchable functions, e.g. not connected."""


@dataclass(frozen=True, slots=True)
class SourceRoster:
    """Sources of searchable functions, announced to the model without listing each function."""

    entries: tuple[SourceEntry, ...] = ()

    def find(self, name: str) -> SourceEntry | None:
        folded = name.strip().casefold()
        return next((entry for entry in self.entries if entry.name.casefold() == folded), None)

    def render(self) -> str:
        lines = [
            f"<{TOOL_SOURCES_TAG}>",
            _ROSTER_HEADER,
        ]
        lines.extend(_entry_line(entry) for entry in self.entries[:_MAX_ROSTER_ENTRIES])
        omitted = len(self.entries) - _MAX_ROSTER_ENTRIES
        if omitted > 0:
            lines.append(f"- ... {omitted} more (list them with `await search_tools()`)")
        lines.append(f"</{TOOL_SOURCES_TAG}>")
        return "\n".join(lines)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.render().encode()).hexdigest()[:16]


def _entry_line(entry: SourceEntry) -> str:
    noun = "function" if entry.functions == 1 else "functions"
    line = f"- {entry.name}: {entry.functions} {noun}"
    return f"{line} ({entry.note})" if entry.note else line


def build_roster(
    sources: Mapping[str, str], deferred: Collection[str], unavailable: Mapping[str, str]
) -> SourceRoster:
    """Count deferred functions per source; ``unavailable`` maps empty sources to a note."""
    counts: dict[str, int] = {}
    for python_name in deferred:
        source = sources[python_name]
        counts[source] = counts.get(source, 0) + 1
    entries = [SourceEntry(name, count) for name, count in counts.items()]
    entries.extend(
        SourceEntry(name, 0, note) for name, note in unavailable.items() if name not in counts
    )
    return SourceRoster(tuple(sorted(entries, key=lambda entry: entry.name.casefold())))


@dataclass(frozen=True, slots=True)
class PtcCatalog:
    """An immutable snapshot of the sandbox's functions and their definition visibility."""

    functions: Mapping[str, PtcFunction] = field(default_factory=dict)
    """Every callable function by Python name, upfront or deferred."""
    tools: Mapping[str, Tool] = field(default_factory=dict)
    """The tool behind each function, by tool name."""
    sources: Mapping[str, str] = field(default_factory=dict)
    """Source group by Python name."""
    deferred: frozenset[str] = frozenset()
    """Python names whose definitions are not in the code tool's description."""
    search_active: bool = False
    roster: SourceRoster = field(default_factory=SourceRoster)
    shared_functions: tuple[str, ...] = ()
    """Functions whose tools the model can also call directly."""
    shell_function: PtcFunction | None = None
    type_check_stubs: str = ""
    retired: frozenset[str] = frozenset()
    """Names assigned earlier in this conversation whose tools are gone."""
    routes: Mapping[str, ToolRoute] = field(default_factory=dict)
    """Origin-qualified destination of each external (non-built-in) function's tool, by listed
    name; a run dispatches its nested calls through the routes of the catalogue it captured."""
    search_index: dict[str, SearchDocument] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )
    """``search_tools`` documents by Python name, built on this catalogue's first search."""

    @property
    def upfront(self) -> list[PtcFunction]:
        return [f for name, f in self.functions.items() if name not in self.deferred]

    @property
    def deferred_functions(self) -> list[PtcFunction]:
        return [f for name, f in self.functions.items() if name in self.deferred]
