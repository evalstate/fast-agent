from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, cast

import pytest
from mcp.types import TextContent

from fast_agent.commands.context import AgentProvider, CommandContext, NonInteractiveCommandIOBase
from fast_agent.commands.handlers import history
from fast_agent.commands.handlers.shared import LoadedPromptMessagesResult
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandMessage


class _Agent:
    name = "main"
    usage_accumulator = None

    def __init__(self) -> None:
        self.message_history: list[PromptMessageExtended] = []

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts
        self.message_history.clear()

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None:
        self.message_history = list(messages or [])

    def pop_last_message(self) -> PromptMessageExtended | None:
        return self.message_history.pop() if self.message_history else None


class _Provider:
    def __init__(self) -> None:
        self.agent = _Agent()

    def _agent(self, name: str) -> _Agent:
        del name
        return self.agent


class _IO(NonInteractiveCommandIOBase):
    def __init__(self) -> None:
        self.history_turns: list[
            tuple[str, list[PromptMessageExtended], int | None, int | None]
        ] = []

    async def emit(self, message: "CommandMessage") -> None:
        del message

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        self.history_turns.append((agent_name, list(turn), turn_index, total_turns))


@pytest.mark.asyncio
async def test_history_load_formats_singular_message_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    message = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="hello")],
    )
    monkeypatch.setattr(
        history,
        "load_prompt_messages_result",
        lambda *_args, **_kwargs: LoadedPromptMessagesResult(messages=[message]),
    )

    outcome = await history.handle_history_load(
        CommandContext(
            agent_provider=cast("AgentProvider", provider),
            current_agent_name="main",
            io=_IO(),
        ),
        agent_name="main",
        filename="one.json",
    )

    assert provider.agent.message_history == [message]
    assert [item.plain_text() for item in outcome.messages] == ["Loaded 1 message from one.json"]


@pytest.mark.asyncio
async def test_history_load_returns_loader_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    monkeypatch.setattr(
        history,
        "load_prompt_messages_result",
        lambda *_args, **_kwargs: LoadedPromptMessagesResult(
            error="Error loading history: invalid JSON"
        ),
    )

    outcome = await history.handle_history_load(
        CommandContext(
            agent_provider=cast("AgentProvider", provider),
            current_agent_name="main",
            io=_IO(),
        ),
        agent_name="main",
        filename="bad.json",
    )

    assert provider.agent.message_history == []
    assert [item.plain_text() for item in outcome.messages] == [
        "Error loading history: invalid JSON"
    ]


def test_strip_web_tool_traces_removes_only_known_web_tool_result_types() -> None:
    result = history._strip_web_tool_traces_from_raw_assistant_channel(
        [
            TextContent(type="text", text=json.dumps({"type": "web_search_tool_result"})),
            TextContent(type="text", text=json.dumps({"type": "web_fetch_tool_result"})),
            TextContent(type="text", text=json.dumps({"type": "web_search_preview_tool_result"})),
        ]
    )

    assert result.removed_count == 2
    assert len(result.blocks) == 1
    retained_block = result.blocks[0]
    assert isinstance(retained_block, TextContent)
    assert "web_search_preview_tool_result" in retained_block.text


def test_strip_web_tool_traces_removes_only_known_server_tool_uses() -> None:
    result = history._strip_web_tool_traces_from_raw_assistant_channel(
        [
            TextContent(
                type="text",
                text=json.dumps({"type": "server_tool_use", "name": "web_search"}),
            ),
            TextContent(
                type="text",
                text=json.dumps({"type": "server_tool_use", "name": "web_search_preview"}),
            ),
        ]
    )

    assert result.removed_count == 1
    assert len(result.blocks) == 1
    retained_block = result.blocks[0]
    assert isinstance(retained_block, TextContent)
    assert "web_search_preview" in retained_block.text


def test_strip_web_tool_traces_retains_malformed_json_text() -> None:
    result = history._strip_web_tool_traces_from_raw_assistant_channel(
        [
            TextContent(type="text", text="{not-json"),
        ]
    )

    assert result.removed_count == 0
    assert len(result.blocks) == 1
    retained_block = result.blocks[0]
    assert isinstance(retained_block, TextContent)
    assert retained_block.text == "{not-json"


def _message(role: Literal["user", "assistant"], text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role=role,
        content=[TextContent(type="text", text=text)],
    )


def test_trim_history_for_rewind_keeps_previous_assistant_boundary() -> None:
    first_turn = [_message("user", "first"), _message("assistant", "first reply")]
    second_turn = [_message("user", "second"), _message("assistant", "second reply")]
    history_messages = [*first_turn, *second_turn]

    assert history._trim_history_for_rewind(
        history_messages,
        turn_start_index=2,
    ) == first_turn


def test_trim_history_for_rewind_uses_templates_without_previous_assistant() -> None:
    template_messages = [_message("assistant", "system template")]
    history_messages = [_message("user", "first")]

    assert history._trim_history_for_rewind(
        history_messages,
        turn_start_index=0,
        template_messages=template_messages,
    ) == template_messages


@pytest.mark.asyncio
async def test_history_review_reports_no_user_turns() -> None:
    provider = _Provider()
    ctx = CommandContext(
        agent_provider=cast("AgentProvider", provider),
        current_agent_name="main",
        io=_IO(),
    )

    outcome = await history.handle_history_review(ctx, agent_name="main", turn_index=1)

    assert [message.plain_text() for message in outcome.messages] == [
        "No user turns available to review."
    ]


@pytest.mark.asyncio
async def test_history_rewind_reports_no_user_turns() -> None:
    provider = _Provider()
    ctx = CommandContext(
        agent_provider=cast("AgentProvider", provider),
        current_agent_name="main",
        io=_IO(),
    )

    outcome = await history.handle_history_rewind(ctx, agent_name="main", turn_index=1)

    assert [message.plain_text() for message in outcome.messages] == [
        "No user turns available to rewind."
    ]


@pytest.mark.asyncio
async def test_history_review_and_rewind_report_missing_turn_usage() -> None:
    provider = _Provider()
    ctx = CommandContext(
        agent_provider=cast("AgentProvider", provider),
        current_agent_name="main",
        io=_IO(),
    )

    review = await history.handle_history_review(ctx, agent_name="main", turn_index=None)
    rewind = await history.handle_history_rewind(ctx, agent_name="main", turn_index=None)

    assert [message.plain_text() for message in review.messages] == [
        "Usage: /history detail <turn>",
        "Tip: press Tab after '/history detail ' to see turn options.",
    ]
    assert [message.plain_text() for message in rewind.messages] == [
        "Usage: /history rewind <turn>",
        "Tip: press Tab after '/history rewind ' to see turn options.",
    ]


@pytest.mark.asyncio
async def test_history_review_and_rewind_share_out_of_range_error() -> None:
    provider = _Provider()
    provider.agent.message_history = [_message("user", "hello"), _message("assistant", "hi")]
    ctx = CommandContext(
        agent_provider=cast("AgentProvider", provider),
        current_agent_name="main",
        io=_IO(),
    )

    review = await history.handle_history_review(ctx, agent_name="main", turn_index=2)
    rewind = await history.handle_history_rewind(ctx, agent_name="main", turn_index=2)

    assert [message.plain_text() for message in review.messages] == [
        "Turn index out of range."
    ]
    assert [message.plain_text() for message in rewind.messages] == [
        "Turn index out of range."
    ]


@pytest.mark.asyncio
async def test_history_rewind_rejects_whitespace_only_user_text() -> None:
    provider = _Provider()
    original_history = [_message("user", "   "), _message("assistant", "hi")]
    provider.agent.message_history = list(original_history)
    ctx = CommandContext(
        agent_provider=cast("AgentProvider", provider),
        current_agent_name="main",
        io=_IO(),
    )

    outcome = await history.handle_history_rewind(ctx, agent_name="main", turn_index=1)

    assert provider.agent.message_history == original_history
    assert outcome.buffer_prefill is None
    assert [message.plain_text() for message in outcome.messages] == [
        "Selected turn has no text content to rewind."
    ]


@pytest.mark.asyncio
async def test_history_review_displays_selected_turn_with_total_count() -> None:
    provider = _Provider()
    first_turn = [_message("user", "first"), _message("assistant", "first reply")]
    second_turn = [_message("user", "second"), _message("assistant", "second reply")]
    provider.agent.message_history = [*first_turn, *second_turn]
    io = _IO()
    ctx = CommandContext(
        agent_provider=cast("AgentProvider", provider),
        current_agent_name="main",
        io=io,
    )

    outcome = await history.handle_history_review(ctx, agent_name="main", turn_index=2)

    assert [message.plain_text() for message in outcome.messages] == [
        "History detail: turn 2"
    ]
    assert io.history_turns == [("main", second_turn, 2, 2)]
