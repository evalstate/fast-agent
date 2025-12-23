from fast_agent.config import Settings
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.ui.console_display import ConsoleDisplay, _StreamingMessageHandle


def _make_handle(streaming_mode: str = "markdown") -> _StreamingMessageHandle:
    settings = Settings()
    settings.logger.streaming = streaming_mode
    display = ConsoleDisplay(settings)
    return _StreamingMessageHandle(
        display=display,
        bottom_items=None,
        highlight_index=None,
        max_item_length=None,
        use_plain_text=streaming_mode == "plain",
        header_left="",
        header_right="",
        progress_display=None,
    )


def test_reasoning_stream_switches_back_to_markdown() -> None:
    handle = _make_handle("markdown")

    handle._handle_stream_chunk(StreamChunk("Intro"))
    assert handle._use_plain_text is False

    handle._handle_stream_chunk(StreamChunk("Thinking", is_reasoning=True))
    assert handle._use_plain_text is True
    assert handle._reasoning_active is True

    handle._handle_stream_chunk(StreamChunk("Answer"))
    assert handle._use_plain_text is False
    assert handle._reasoning_active is False

    text = "".join(handle._buffer)
    intro_idx = text.find("Intro")
    thinking_idx = text.find("Thinking")
    answer_idx = text.find("Answer")
    assert intro_idx != -1
    assert thinking_idx != -1
    assert answer_idx != -1
    assert "\n" in text[intro_idx + len("Intro") : thinking_idx]
    assert "\n" in text[thinking_idx + len("Thinking") : answer_idx]


def test_tool_mode_switches_back_to_markdown() -> None:
    handle = _make_handle("markdown")

    handle._handle_chunk("Intro")
    handle._begin_tool_mode()
    assert handle._use_plain_text is True

    handle._handle_chunk("Calling tool")
    handle._end_tool_mode()
    assert handle._use_plain_text is False

    handle._handle_chunk("Result")

    text = "".join(handle._buffer)
    intro_idx = text.find("Intro")
    tool_idx = text.find("Calling tool")
    result_idx = text.find("Result")
    assert intro_idx != -1
    assert tool_idx != -1
    assert result_idx != -1
    assert "\n" in text[intro_idx + len("Intro") : tool_idx]
    assert "\n" in text[tool_idx + len("Calling tool") : result_idx]
