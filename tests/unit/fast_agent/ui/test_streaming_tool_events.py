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
