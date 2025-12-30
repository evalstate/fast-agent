from fast_agent.llm.stream_types import StreamChunk
from fast_agent.ui.stream_segments import StreamSegmentAssembler


def test_reasoning_stream_switches_back_to_markdown() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Intro"))
    assembler.handle_stream_chunk(StreamChunk("Thinking", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer"))

    text = "".join(segment.text for segment in assembler.segments)
    intro_idx = text.find("Intro")
    thinking_idx = text.find("Thinking")
    answer_idx = text.find("Answer")
    assert intro_idx != -1
    assert thinking_idx != -1
    assert answer_idx != -1
    assert "\n" in text[intro_idx + len("Intro") : thinking_idx]
    assert "\n\n" in text[thinking_idx + len("Thinking") : answer_idx]


def test_reasoning_stream_handles_multiple_blocks() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Think1", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer1"))
    assembler.handle_stream_chunk(StreamChunk("Think2", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer2"))

    text = "".join(segment.text for segment in assembler.segments)
    assert "Think1" in text
    assert "Answer1" in text
    assert "Think2" in text
    assert "Answer2" in text
