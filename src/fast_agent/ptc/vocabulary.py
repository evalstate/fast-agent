"""PTC vocabulary versioning and temporary profile-name migration.

Remove the legacy hints and ``function_names`` switch after the matched rollout
(two or three releases). No compatibility functions are registered in the sandbox.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from fast_agent.tools.tool_sources import SHELL_TOOL_SOURCE, SKILL_TOOL_SOURCE, tool_source

if TYPE_CHECKING:
    from mcp_types import Tool

PtcFunctionNames = Literal["canonical", "profile"]
PTC_SHELL_FUNCTION: Final = "shell"
PTC_VOCABULARY_VERSION: Final = 1
PTC_VOCABULARY_META: Final = "fast-agent-ptc-vocabulary"
PTC_BUILTIN_ORDER: Final = (
    PTC_SHELL_FUNCTION,
    "process",
    "write_text_file",
    "edit_file",
    "attach_media",
    "get_resource",
    "subagent",
)
"""Order of built-in functions in the code tool's stubs; other tools follow alphabetically."""
PTC_BUILTIN_FUNCTIONS: Final = frozenset(PTC_BUILTIN_ORDER)

# ptc-vocab-v0: remove after matched rollout; deliberately not aliases in the stub/type-check surface.
LEGACY_FUNCTIONS: Final = {
    "bash": "shell",
    "Shell": "shell",
    "exec": "shell",
    "exec_tool": "shell",
    "execute": "shell",
    "poll_process": "process",
    "terminate_process": "process",
    "apply_patch": "edit_file",
}


@runtime_checkable
class PtcBuiltinNameReserver(Protocol):
    """Optional reservation capability used when installing harness-owned tools."""

    def reserve_ptc_builtin_name(self, name: str) -> bool: ...


BUILTIN_TOOL_OWNERS: Final = frozenset({"harness", "subagent", "elicitation", "image_gen"})
"""``meta["fast_agent"]["builtin"]`` values of tools the fast-agent runtime installs."""


def is_builtin_tool(tool: Tool) -> bool:
    """Runtime ownership, not a name guess: user tools cannot claim built-in names."""
    return tool_source(tool) in {
        SHELL_TOOL_SOURCE,
        SKILL_TOOL_SOURCE,
    } or is_builtin_function_metadata(tool.meta)


def is_builtin_function_metadata(meta: Mapping[str, object] | None) -> bool:
    metadata = (meta or {}).get("fast_agent")
    return isinstance(metadata, Mapping) and metadata.get("builtin") in BUILTIN_TOOL_OWNERS


def function_order(tool: Tool) -> tuple[int, str]:
    """Sort key: built-ins in ``PTC_BUILTIN_ORDER``, then other built-ins, then the rest by name."""
    if not is_builtin_tool(tool):
        return len(PTC_BUILTIN_ORDER) + 1, tool.name
    if tool.name in PTC_BUILTIN_ORDER:
        return PTC_BUILTIN_ORDER.index(tool.name), ""
    return len(PTC_BUILTIN_ORDER), tool.name


_UNDEFINED_NAME = re.compile(
    r"(?m)^[^\n]+:\d+:\d+: error\[unresolved-reference\] Name `([A-Za-z_]\w*)` used when not defined$"
)
_NAME_ERROR = re.compile(r"NameError: name '([A-Za-z_]\w*)' is not defined")


def undefined_names(detail: str, *, ran: bool = False) -> set[str]:
    """Names Monty's type checker reported as undefined.

    With ``ran``, ``detail`` is a failed run's traceback (no type check ran): its final
    NameError, if any, names the undefined name.
    """
    if not ran:
        return set(_UNDEFINED_NAME.findall(detail))
    match = _NAME_ERROR.fullmatch(detail.rstrip().rpartition("\n")[2])
    return {match[1]} if match else set()


def legacy_name_hint(undefined: Collection[str], available: Mapping[str, object]) -> str:
    """Point ``undefined`` names a renamed function used to have at its current name."""
    hints = [
        f"`{old}` is `{new}` here."
        + (" Use action='wait'." if old == "poll_process" else "")
        + (" Use action='stop'." if old == "terminate_process" else "")
        for old, new in LEGACY_FUNCTIONS.items()
        if old not in available and new in available and old in undefined
    ]
    return "\nHint: " + " ".join(hints) if hints else ""
