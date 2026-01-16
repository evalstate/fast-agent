"""Unit tests for the history trimmer hook."""

import pytest
from mcp.types import TextContent

from fast_agent.hooks.history_trimmer import (
    _find_turn_start,
    _trim_turn_messages,
    trim_tool_loop_history,
)
from fast_agent.hooks.hook_context import HookContext
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


def _make_user_msg(text: str, has_tool_results: bool = False) -> PromptMessageExtended:
    """Create a user message, optionally with tool results."""
    msg = PromptMessageExtended(role="user", content=[TextContent(type="text", text=text)])
    if has_tool_results:
        # Simulate tool results by setting the attribute
        msg.tool_results = {"tool1": type("Result", (), {"content": []})()}
    return msg


def _make_assistant_msg(text: str, is_tool_call: bool = False) -> PromptMessageExtended:
    """Create an assistant message, optionally as a tool call."""
    msg = PromptMessageExtended(role="assistant", content=[TextContent(type="text", text=text)])
    msg.stop_reason = LlmStopReason.TOOL_USE if is_tool_call else LlmStopReason.END_TURN
    return msg


def _first_text(message: PromptMessageExtended) -> str:
    content = message.content[0]
    assert isinstance(content, TextContent)
    return content.text


@pytest.mark.unit
class TestTrimTurnMessages:
    """Tests for _trim_turn_messages function."""

    def test_multiple_tool_calls_trimmed(self):
        """Test that multiple tool calls are trimmed to just the last one."""
        # Input: user, 3 tool_calls with results, final response (8 messages)
        messages = [
            _make_user_msg("hello"),
            _make_assistant_msg("calling tool 1", True),
            _make_user_msg("result 1", True),
            _make_assistant_msg("calling tool 2", True),
            _make_user_msg("result 2", True),
            _make_assistant_msg("calling tool 3", True),
            _make_user_msg("result 3", True),
            _make_assistant_msg("final response"),
        ]

        trimmed = _trim_turn_messages(messages)

        # Should have 4 messages: user, last_tool_call, result, final
        assert len(trimmed) == 4
        assert _first_text(trimmed[0]) == "hello"
        assert _first_text(trimmed[1]) == "calling tool 3"
        assert _first_text(trimmed[2]) == "result 3"
        assert _first_text(trimmed[3]) == "final response"

    def test_single_tool_call_not_trimmed(self):
        """Test that a single tool call is not trimmed."""
        messages = [
            _make_user_msg("hello"),
            _make_assistant_msg("calling tool", True),
            _make_user_msg("result", True),
            _make_assistant_msg("final response"),
        ]

        trimmed = _trim_turn_messages(messages)

        # Should be unchanged - only 1 tool call
        assert len(trimmed) == 4
        assert _first_text(trimmed[0]) == "hello"
        assert _first_text(trimmed[1]) == "calling tool"
        assert _first_text(trimmed[2]) == "result"
        assert _first_text(trimmed[3]) == "final response"

    def test_no_tool_calls_not_trimmed(self):
        """Test that messages without tool calls are not trimmed."""
        messages = [
            _make_user_msg("hello"),
            _make_assistant_msg("direct response"),
        ]

        trimmed = _trim_turn_messages(messages)

        # Should be unchanged - no tool calls
        assert len(trimmed) == 2
        assert _first_text(trimmed[0]) == "hello"
        assert _first_text(trimmed[1]) == "direct response"

    def test_too_few_messages_not_trimmed(self):
        """Test that very short message lists are not trimmed."""
        messages = [
            _make_user_msg("hello"),
            _make_assistant_msg("hi"),
        ]

        trimmed = _trim_turn_messages(messages)

        assert len(trimmed) == 2

    def test_two_tool_calls_trimmed_to_last(self):
        """Test that exactly two tool calls are trimmed correctly."""
        messages = [
            _make_user_msg("hello"),
            _make_assistant_msg("calling tool 1", True),
            _make_user_msg("result 1", True),
            _make_assistant_msg("calling tool 2", True),
            _make_user_msg("result 2", True),
            _make_assistant_msg("final response"),
        ]

        trimmed = _trim_turn_messages(messages)

        assert len(trimmed) == 4
        assert _first_text(trimmed[0]) == "hello"
        assert _first_text(trimmed[1]) == "calling tool 2"
        assert _first_text(trimmed[2]) == "result 2"
        assert _first_text(trimmed[3]) == "final response"


@pytest.mark.unit
class TestFindTurnStart:
    """Tests for _find_turn_start function."""

    def test_single_turn(self):
        """Test finding turn start with a single turn."""
        history = [
            _make_user_msg("hello"),
            _make_assistant_msg("calling tool", True),
            _make_user_msg("result", True),
            _make_assistant_msg("done"),
        ]

        idx = _find_turn_start(history)
        assert idx == 0

    def test_multiple_turns(self):
        """Test finding turn start with multiple prior turns."""
        history = [
            # First turn
            _make_user_msg("first question"),
            _make_assistant_msg("first answer"),
            # Second turn (current)
            _make_user_msg("second question"),
            _make_assistant_msg("calling tool", True),
            _make_user_msg("result", True),
            _make_assistant_msg("second answer"),
        ]

        idx = _find_turn_start(history)
        assert idx == 2  # Index of "second question"

    def test_turn_start_with_prior_tool_results(self):
        """Test that tool result messages don't count as turn starts."""
        history = [
            _make_user_msg("question"),
            _make_assistant_msg("tool call 1", True),
            _make_user_msg("tool result 1", True),  # This is NOT a turn start
            _make_assistant_msg("tool call 2", True),
            _make_user_msg("tool result 2", True),  # This is NOT a turn start
            _make_assistant_msg("final"),
        ]

        idx = _find_turn_start(history)
        assert idx == 0  # Should be the original user message

    def test_empty_history(self):
        """Test with empty history."""
        idx = _find_turn_start([])
        assert idx == 0


@pytest.mark.unit
class TestTrimToolLoopHistory:
    """Tests for the full trim_tool_loop_history hook."""

    @pytest.mark.asyncio
    async def test_trims_history_when_turn_complete(self):
        """Test that history is trimmed when turn is complete."""
        # Create a mock agent with message history
        class MockAgent:
            name = "mock"

            def __init__(self):
                self._history = [
                    _make_user_msg("hello"),
                    _make_assistant_msg("tool 1", True),
                    _make_user_msg("result 1", True),
                    _make_assistant_msg("tool 2", True),
                    _make_user_msg("result 2", True),
                    _make_assistant_msg("done"),
                ]

            @property
            def message_history(self) -> list[PromptMessageExtended]:
                return self._history

            def load_message_history(
                self, messages: list[PromptMessageExtended] | None
            ) -> None:
                self._history = list(messages or [])

        # Create mock runner
        class MockRunner:
            iteration = 2

        agent = MockAgent()
        final_message = _make_assistant_msg("done")

        ctx = HookContext(
            runner=MockRunner(),
            agent=agent,
            message=final_message,
            hook_type="after_turn_complete",
        )

        await trim_tool_loop_history(ctx)

        # History should be trimmed
        assert len(agent.message_history) == 4
        assert _first_text(agent.message_history[0]) == "hello"
        assert _first_text(agent.message_history[1]) == "tool 2"
        assert _first_text(agent.message_history[2]) == "result 2"
        assert _first_text(agent.message_history[3]) == "done"

    @pytest.mark.asyncio
    async def test_does_not_trim_when_turn_not_complete(self):
        """Test that history is NOT trimmed during intermediate tool calls."""

        class MockAgent:
            name = "mock"

            def __init__(self):
                self._history = [
                    _make_user_msg("hello"),
                    _make_assistant_msg("tool 1", True),
                ]

            @property
            def message_history(self) -> list[PromptMessageExtended]:
                return self._history

            def load_message_history(
                self, messages: list[PromptMessageExtended] | None
            ) -> None:
                self._history = list(messages or [])

        class MockRunner:
            iteration = 1

        agent = MockAgent()
        # Simulate an intermediate tool_call message (turn not complete)
        tool_message = _make_assistant_msg("tool 1", True)

        ctx = HookContext(
            runner=MockRunner(),
            agent=agent,
            message=tool_message,
            hook_type="after_turn_complete",
        )

        original_len = len(agent.message_history)
        await trim_tool_loop_history(ctx)

        # History should NOT be modified
        assert len(agent.message_history) == original_len

    @pytest.mark.asyncio
    async def test_preserves_prior_turns(self):
        """Test that prior turns are preserved when trimming current turn."""

        class MockAgent:
            name = "mock"

            def __init__(self):
                self._history = [
                    # Prior turn
                    _make_user_msg("prior question"),
                    _make_assistant_msg("prior answer"),
                    # Current turn with multiple tool calls
                    _make_user_msg("current question"),
                    _make_assistant_msg("tool 1", True),
                    _make_user_msg("result 1", True),
                    _make_assistant_msg("tool 2", True),
                    _make_user_msg("result 2", True),
                    _make_assistant_msg("current answer"),
                ]

            @property
            def message_history(self) -> list[PromptMessageExtended]:
                return self._history

            def load_message_history(
                self, messages: list[PromptMessageExtended] | None
            ) -> None:
                self._history = list(messages or [])

        class MockRunner:
            iteration = 2

        agent = MockAgent()
        final_message = _make_assistant_msg("current answer")

        ctx = HookContext(
            runner=MockRunner(),
            agent=agent,
            message=final_message,
            hook_type="after_turn_complete",
        )

        await trim_tool_loop_history(ctx)

        # Should have: prior turn (2) + trimmed current turn (4) = 6 messages
        assert len(agent.message_history) == 6
        # Prior turn preserved
        assert _first_text(agent.message_history[0]) == "prior question"
        assert _first_text(agent.message_history[1]) == "prior answer"
        # Current turn trimmed
        assert _first_text(agent.message_history[2]) == "current question"
        assert _first_text(agent.message_history[3]) == "tool 2"
        assert _first_text(agent.message_history[4]) == "result 2"
        assert _first_text(agent.message_history[5]) == "current answer"
