from fast_agent.config import Settings
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.ui.console_display import ConsoleDisplay, _StreamingMessageHandle


def _make_handle() -> _StreamingMessageHandle:
    settings = Settings()
    settings.logger.streaming = "markdown"
    display = ConsoleDisplay(settings)
    return _StreamingMessageHandle(
        display=display,
        bottom_items=None,
        highlight_index=None,
        max_item_length=None,
        use_plain_text=False,
        header_left="",
        header_right="",
        progress_display=None,
    )


def test_table_rows_do_not_duplicate_when_streaming_in_parts() -> None:
    handle = _make_handle()

    chunks = ["| Mission | ", "Landing Date |", "\n"]
    for chunk in chunks:
        handle._handle_chunk(chunk)

    text = "".join(handle._buffer)
    assert text == "".join(chunks)


def test_table_rows_do_not_duplicate_when_reasoning_interrupts() -> None:
    handle = _make_handle()

    handle._handle_chunk("| Mission ")
    handle._handle_stream_chunk(StreamChunk("thinking", is_reasoning=True))
    handle._handle_stream_chunk(StreamChunk(" done", is_reasoning=False))
    handle._handle_chunk("Mission | | Landing Date |\n")

    text = "".join(handle._buffer)
    assert text.count("| Mission Mission |") == 0
    assert text.count("| Mission ") == 1


def test_table_pending_row_not_duplicated_after_reasoning() -> None:
    handle = _make_handle()

    handle._handle_stream_chunk(StreamChunk("thinking", is_reasoning=True))
    handle._handle_stream_chunk(StreamChunk(" |", is_reasoning=False))
    assert handle._pending_table_row == " |"

    handle._handle_stream_chunk(StreamChunk(" Fact |\n", is_reasoning=False))
    text = "".join(handle._buffer)
    assert text.endswith(" | Fact |\n")
