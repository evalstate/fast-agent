"""PTC vocabulary versioning and temporary profile-name migration.

Remove the legacy hints and ``function_names`` switch after the matched rollout
(two or three releases). No compatibility functions are registered in the sandbox.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from fast_agent.tools.tool_sources import SHELL_TOOL_SOURCE, tool_source

if TYPE_CHECKING:
    from mcp_types import Tool

PtcFunctionNames = Literal["canonical", "profile"]
PTC_SHELL_FUNCTION: Final = "shell"
PTC_VOCABULARY_VERSION: Final = 1
PTC_VOCABULARY_META: Final = "fast-agent-ptc-vocabulary"
PTC_BUILTIN_FUNCTIONS: Final = frozenset(
    {
        PTC_SHELL_FUNCTION,
        "process",
        "read_text_file",
        "write_text_file",
        "edit_file",
        "subagent",
        "attach_media",
        "get_resource",
    }
)

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


def is_builtin_tool(tool: Tool) -> bool:
    """Runtime ownership, not a name guess: user tools cannot claim built-in names."""
    metadata = (tool.meta or {}).get("fast_agent")
    return tool_source(tool) == SHELL_TOOL_SOURCE or (
        isinstance(metadata, Mapping) and metadata.get("builtin") in {"harness", "subagent"}
    )


def legacy_name_hint(detail: str, available: Mapping[str, object]) -> str:
    undefined = set(
        re.findall(
            r"(?m)^[^\n]+:\d+:\d+: error\[unresolved-reference\] Name `([A-Za-z_]\w*)` used when not defined$",
            detail,
        )
    )
    hints = [
        f"`{old}` is `{new}` here."
        + (" Use action='wait'." if old == "poll_process" else "")
        + (" Use action='stop'." if old == "terminate_process" else "")
        for old, new in LEGACY_FUNCTIONS.items()
        if old not in available and new in available and old in undefined
    ]
    return "\nHint: " + " ".join(hints) if hints else ""
