from fast_agent.config import Settings
from fast_agent.ui.console_display import ConsoleDisplay, _StreamingMessageHandle


def _make_handle() -> _StreamingMessageHandle:
    settings = Settings()
    settings.logger.streaming = "markdown"
    display = ConsoleDisplay(settings)
    handle = _StreamingMessageHandle(
        display=display,
        bottom_items=None,
        highlight_index=None,
        max_item_length=None,
        use_plain_text=False,
        header_left="",
        header_right="",
        progress_display=None,
    )
    handle._async_mode = False
    handle._queue = None
    handle._live = None
    handle._active = True
    def _capture(chunk: str) -> None:
        if chunk:
            handle._buffer.append(chunk)
    handle.update = _capture  # type: ignore[method-assign]
    return handle


def test_tool_stream_delta_bootstraps_mode() -> None:
    handle = _make_handle()

    handle.handle_tool_event("delta", {"tool_name": "search", "chunk": "{\"q\":1}"})

    text = "".join(handle._buffer)
    assert "Calling search" in text
    assert "{\"q\":1}" in text
    assert handle._tool_active is True

    handle.handle_tool_event("stop", {"tool_name": "search"})
    assert handle._tool_active is False
