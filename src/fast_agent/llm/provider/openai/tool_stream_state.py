from __future__ import annotations

from dataclasses import dataclass

from fast_agent.llm.tool_tracking import ToolCallState, ToolCallTracker, ToolKind


@dataclass(slots=True)
class OpenAIToolStreamEntry:
    state: ToolCallState
    item_id: str | None = None
    item_type: str | None = None
    stop_notified: bool = False

    @property
    def tool_name(self) -> str:
        return self.state.name

    @property
    def tool_use_id(self) -> str:
        return self.state.tool_use_id

    @property
    def index(self) -> int | None:
        return self.state.index

    @property
    def start_notified(self) -> bool:
        return self.state.start_notified

    @start_notified.setter
    def start_notified(self, value: bool) -> None:
        self.state.start_notified = value


class OpenAIToolStreamState:
    """Track OpenAI-family stream tool lifecycle plus item-id aliases."""

    def __init__(self) -> None:
        self._tracker = ToolCallTracker()
        self._entries: dict[str, OpenAIToolStreamEntry] = {}
        self._item_id_aliases: dict[str, str] = {}

    def register(
        self,
        *,
        tool_use_id: str,
        name: str,
        index: int | None = None,
        item_id: str | None = None,
        item_type: str | None = None,
        kind: ToolKind = "tool",
    ) -> OpenAIToolStreamEntry:
        previous_entry = self.resolve_open(
            tool_use_id=tool_use_id,
            index=index,
            item_id=item_id,
        )
        previous_tool_use_id = previous_entry.tool_use_id if previous_entry is not None else None
        if previous_tool_use_id and previous_tool_use_id != tool_use_id:
            self._tracker.rekey_open(
                tool_use_id=previous_tool_use_id,
                new_tool_use_id=tool_use_id,
            )

        state = self._tracker.register(
            tool_use_id=tool_use_id,
            name=name,
            index=index,
            kind=kind,
        )

        entry: OpenAIToolStreamEntry | None = None
        if previous_tool_use_id and previous_tool_use_id != state.tool_use_id:
            entry = self._entries.pop(previous_tool_use_id, None)
        if entry is None:
            entry = previous_entry or self._entries.get(state.tool_use_id)
        if entry is None:
            entry = OpenAIToolStreamEntry(state=state)
        else:
            entry.state = state

        if item_id:
            self._item_id_aliases[item_id] = state.tool_use_id
            entry.item_id = item_id
        if item_type:
            entry.item_type = item_type

        self._entries[state.tool_use_id] = entry
        return entry

    def resolve_open(
        self,
        *,
        tool_use_id: str | None = None,
        index: int | None = None,
        item_id: str | None = None,
    ) -> OpenAIToolStreamEntry | None:
        resolved_tool_use_id = self._resolve_tool_use_id(
            tool_use_id=tool_use_id,
            item_id=item_id,
        )
        state = self._tracker.resolve_open(tool_use_id=resolved_tool_use_id, index=index)
        if state is None and item_id:
            state = self._tracker.resolve_open(tool_use_id=item_id, index=index)
        if state is None:
            return None
        return self._entry_for_state(state)

    def close(
        self,
        *,
        tool_use_id: str | None = None,
        index: int | None = None,
        item_id: str | None = None,
    ) -> OpenAIToolStreamEntry | None:
        entry = self.resolve_open(tool_use_id=tool_use_id, index=index, item_id=item_id)
        if entry is None:
            return None
        state = self._tracker.close(tool_use_id=entry.tool_use_id, index=entry.index)
        if state is not None:
            entry.state = state
        return entry

    def is_completed(
        self,
        *,
        tool_use_id: str | None = None,
        index: int | None = None,
        item_id: str | None = None,
    ) -> bool:
        resolved_tool_use_id = self._resolve_tool_use_id(
            tool_use_id=tool_use_id,
            item_id=item_id,
        )
        if self._tracker.is_completed(tool_use_id=resolved_tool_use_id, index=index):
            return True
        if item_id and self._tracker.is_completed(tool_use_id=item_id, index=index):
            return True
        return False

    def incomplete(self) -> list[OpenAIToolStreamEntry]:
        return [self._entry_for_state(state) for state in self._tracker.incomplete()]

    def _entry_for_state(self, state: ToolCallState) -> OpenAIToolStreamEntry:
        entry = self._entries.get(state.tool_use_id)
        if entry is None:
            entry = OpenAIToolStreamEntry(state=state)
            self._entries[state.tool_use_id] = entry
        else:
            entry.state = state
        return entry

    def _resolve_tool_use_id(
        self,
        *,
        tool_use_id: str | None,
        item_id: str | None,
    ) -> str | None:
        if tool_use_id:
            return tool_use_id
        if item_id:
            return self._item_id_aliases.get(item_id, item_id)
        return None
