"""Turn boundaries must follow user prompts, not user-role messages fast-agent adds."""

import pytest
from mcp_types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ImageContent,
    TextContent,
)

from fast_agent.commands.history_summaries import collect_user_turns
from fast_agent.constants import FAST_AGENT_TOOL_MEDIA_MESSAGE
from fast_agent.history.compaction import build_summary_message
from fast_agent.hooks.history_trimmer import _find_turn_start
from fast_agent.types import PromptMessageExtended, split_into_turns
from fast_agent.types.llm_stop_reason import LlmStopReason


def _user(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(role="user", content=[TextContent(type="text", text=text)])


def _reply(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        stop_reason=LlmStopReason.END_TURN,
    )


def _tool_call(call_id: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        stop_reason=LlmStopReason.TOOL_USE,
        tool_calls={
            call_id: CallToolRequest(params=CallToolRequestParams(name="tool", arguments={}))
        },
    )


def _tool_result(call_id: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="user",
        tool_results={call_id: CallToolResult(content=[TextContent(type="text", text="ok")])},
    )


def _tool_media() -> PromptMessageExtended:
    return PromptMessageExtended(
        role="user",
        content=[ImageContent(type="image", data="AA==", mime_type="image/png")],
        channels={FAST_AGENT_TOOL_MEDIA_MESSAGE: [TextContent(type="text", text="media")]},
    )


def _checkpoint() -> PromptMessageExtended:
    return build_summary_message(
        "summary",
        prompt_text=None,
        instructions=None,
        messages_compacted=2,
        tokens_before=None,
        context_window=None,
        model=None,
    )


def _prompts(history: list[PromptMessageExtended]) -> list[str]:
    return [turn.prompt.first_text() for turn in collect_user_turns(history)]


@pytest.mark.unit
def test_tool_media_continuation_stays_in_its_turn() -> None:
    history = [
        _user("q1"),
        _reply("a1"),
        _user("q2"),
        _tool_call("c"),
        _tool_result("c"),
        _tool_media(),
        _reply("a2"),
    ]

    assert _prompts(history) == ["q1", "q2"]
    assert _find_turn_start(history) == 2
    assert [turn[0].first_text() for turn in split_into_turns(history)] == ["q1", "q2"]


@pytest.mark.unit
def test_compaction_checkpoint_is_not_a_turn() -> None:
    assert _prompts([_checkpoint(), _user("q2"), _reply("a2")]) == ["q2"]
    assert _prompts([_checkpoint(), _tool_call("c"), _tool_result("c"), _reply("a")]) == []


@pytest.mark.unit
def test_prompt_after_interrupted_tool_call_starts_a_turn() -> None:
    history = [_user("q1"), _tool_call("c"), _tool_result("c"), _user("q2"), _reply("a2")]

    assert _prompts(history) == ["q1", "q2"]


@pytest.mark.unit
def test_prompts_before_a_reply_share_a_turn() -> None:
    history = [_user("part 1"), _user("part 2"), _reply("a")]

    turns = collect_user_turns(history)
    assert len(turns) == 1
    assert len(turns[0].messages) == 3


def test_hc_deliveries_continue_the_turn_a_person_started() -> None:
    history = [
        _user("delegate the build"),
        _reply("started t9"),
        _user("<hc>t9 exited 0</hc>"),
        _reply("build finished"),
        _user('<hc from="t7">any news?</hc>'),
        _reply("told t7"),
        _user("thanks, now deploy"),
        _reply("deployed"),
    ]

    turns = collect_user_turns(history)

    assert [turn.prompt.first_text() for turn in turns] == [
        "delegate the build",
        "thanks, now deploy",
    ]
    assert [len(turn.messages) for turn in turns] == [6, 2]
    # The model still sees four prompts; only review and rewind group them.
    assert len(split_into_turns(history)) == 4


def test_hc_delivery_opening_a_session_is_still_a_turn() -> None:
    history = [
        _user('<hc from="t1">your task</hc>'),
        _reply("on it"),
        _user("status?"),
        _reply("ok"),
    ]

    assert [turn.start_index for turn in collect_user_turns(history)] == [0, 2]
