from fast_agent.ui.stream_segments import StreamSegmentAssembler


def _make_assembler() -> StreamSegmentAssembler:
    return StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")


def test_tool_stream_delta_bootstraps_mode() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "delta", {"tool_name": "search", "tool_use_id": "tool-1", "chunk": "{\"q\":1}"}
    )

    text = "".join(segment.text for segment in assembler.segments)
    assert "-> search" in text
    assert "{\"q\":1}" in text

    assembler.handle_tool_event("stop", {"tool_name": "search", "tool_use_id": "tool-1"})
    text = "".join(segment.text for segment in assembler.segments)
    assert "\"q\": 1" in text


def test_tool_stream_status_updates_visible_text() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "start",
        {
            "tool_name": "web_search",
            "tool_use_id": "ws-1",
            "chunk": "starting search...",
        },
    )
    assembler.handle_tool_event(
        "status",
        {
            "tool_name": "web_search",
            "tool_use_id": "ws-1",
            "chunk": "searching...",
            "status": "searching",
        },
    )
    assembler.handle_tool_event(
        "status",
        {
            "tool_name": "web_search",
            "tool_use_id": "ws-1",
            "chunk": "search complete",
            "status": "completed",
        },
    )
    assembler.handle_tool_event("stop", {"tool_name": "web_search", "tool_use_id": "ws-1"})

    text = "".join(segment.text for segment in assembler.segments)
    assert "Searching the web" in text
    assert "search complete" in text
    assert "starting search..." not in text


def test_tool_stream_status_uses_fallback_chunk_when_missing() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "status",
        {
            "tool_name": "web_search_call",
            "tool_use_id": "ws-2",
            "status": "searching",
        },
    )
    assembler.handle_tool_event("stop", {"tool_name": "web_search_call", "tool_use_id": "ws-2"})

    text = "".join(segment.text for segment in assembler.segments)
    assert "Searching the web" in text
    assert "searching..." in text
