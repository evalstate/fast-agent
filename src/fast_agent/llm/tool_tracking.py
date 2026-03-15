from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ToolKind = Literal["tool", "server_tool", "web_search"]


def _is_generic_tool_name(value: str) -> bool:
    return not value or value == "tool"


@dataclass(slots=True)
class ToolCallState:
    tool_use_id: str
    name: str
    kind: ToolKind = "tool"
    index: int | None = None
    start_notified: bool = False


class ToolCallTracker:
    """Track open and completed tool calls for a single stream."""

    def __init__(self) -> None:
        self._open_by_id: dict[str, ToolCallState] = {}
        self._open_by_index: dict[int, ToolCallState] = {}
        self._completed_by_id: dict[str, ToolCallState] = {}
        self._completed_by_index: dict[int, ToolCallState] = {}

    def register(
        self,
        *,
        tool_use_id: str,
        name: str,
        index: int | None = None,
        kind: ToolKind = "tool",
    ) -> ToolCallState:
        state = self._open_by_id.get(tool_use_id)
        if state is None and index is not None:
            state = self._open_by_index.get(index)
            if state is not None and state.tool_use_id != tool_use_id:
                self._rekey_open_state(state, tool_use_id)

        if state is None:
            state = ToolCallState(tool_use_id=tool_use_id, name=name, kind=kind, index=index)
            self._open_by_id[tool_use_id] = state
        else:
            if not _is_generic_tool_name(name) or _is_generic_tool_name(state.name):
                state.name = name
            if state.kind == "tool" and kind != "tool":
                state.kind = kind

        if index is not None:
            self._attach_index(state, index)

        self._open_by_id[state.tool_use_id] = state
        return state

    def resolve_open(
        self,
        *,
        tool_use_id: str | None = None,
        index: int | None = None,
    ) -> ToolCallState | None:
        if tool_use_id is not None and tool_use_id in self._open_by_id:
            return self._open_by_id[tool_use_id]
        if index is not None and index in self._open_by_index:
            return self._open_by_index[index]
        return None

    def close(
        self,
        *,
        tool_use_id: str | None = None,
        index: int | None = None,
    ) -> ToolCallState | None:
        state = self.resolve_open(tool_use_id=tool_use_id, index=index)
        if state is None:
            return None

        self._open_by_id.pop(state.tool_use_id, None)
        if state.index is not None:
            self._open_by_index.pop(state.index, None)

        self._completed_by_id[state.tool_use_id] = state
        if state.index is not None:
            self._completed_by_index[state.index] = state
        return state

    def rekey_open(
        self,
        *,
        tool_use_id: str,
        new_tool_use_id: str,
    ) -> ToolCallState | None:
        state = self._open_by_id.get(tool_use_id)
        if state is None:
            return None
        self._rekey_open_state(state, new_tool_use_id)
        return state

    def is_completed(
        self,
        *,
        tool_use_id: str | None = None,
        index: int | None = None,
    ) -> bool:
        if tool_use_id is not None and tool_use_id in self._completed_by_id:
            return True
        if index is not None and index in self._completed_by_index:
            return True
        return False

    def incomplete(self) -> list[ToolCallState]:
        return list(self._open_by_id.values())

    def completed(self) -> list[ToolCallState]:
        return list(self._completed_by_id.values())

    def _attach_index(self, state: ToolCallState, index: int) -> None:
        if state.index is not None and state.index != index:
            self._open_by_index.pop(state.index, None)
        state.index = index
        self._open_by_index[index] = state

    def _rekey_open_state(self, state: ToolCallState, new_tool_use_id: str) -> None:
        if state.tool_use_id == new_tool_use_id:
            return
        self._open_by_id.pop(state.tool_use_id, None)
        state.tool_use_id = new_tool_use_id
        self._open_by_id[new_tool_use_id] = state
