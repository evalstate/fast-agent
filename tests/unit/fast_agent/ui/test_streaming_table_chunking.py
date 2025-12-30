from fast_agent.llm.stream_types import StreamChunk
from fast_agent.ui.stream_segments import StreamSegmentAssembler


def test_table_rows_do_not_duplicate_when_streaming_in_parts() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")
    chunks = ["| Mission | ", "Landing Date |", "\n"]
    for chunk in chunks:
        assembler.handle_text(chunk)

    text = "".join(segment.text for segment in assembler.segments)
    assert text == "".join(chunks)


def test_table_rows_do_not_duplicate_when_reasoning_interrupts() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_text("| Mission ")
    assembler.handle_stream_chunk(StreamChunk("thinking", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk(" done", is_reasoning=False))
    assembler.handle_text("Mission | | Landing Date |\n")

    text = "".join(segment.text for segment in assembler.segments)
    assert text.count("| Mission Mission |") == 0
    assert text.count("| Mission ") == 1


def test_table_pending_row_not_duplicated_after_reasoning() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("thinking", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk(" |", is_reasoning=False))
    assert assembler.pending_table_row == " |"

    assembler.handle_stream_chunk(StreamChunk(" Fact |\n", is_reasoning=False))
    text = "".join(segment.text for segment in assembler.segments)
    assert text.endswith(" | Fact |\n")
