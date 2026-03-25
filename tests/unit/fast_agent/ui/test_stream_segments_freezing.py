from fast_agent.ui.stream_segments import StreamSegmentAssembler


def test_markdown_freezes_completed_paragraph_before_mutable_tail() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_text("First paragraph\n\nSecond")

    segments = assembler.segments
    assert [segment.text for segment in segments] == ["First paragraph\n\n", "Second"]
    assert [segment.frozen for segment in segments] == [True, False]
    assert "".join(segment.text for segment in segments) == "First paragraph\n\nSecond"


def test_markdown_freezes_closed_code_block_before_mutable_tail() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_text("```python\nprint(1)\n```\nTrailing")

    segments = assembler.segments
    assert [segment.text for segment in segments] == ["```python\nprint(1)\n```\n", "Trailing"]
    assert [segment.frozen for segment in segments] == [True, False]


def test_markdown_does_not_freeze_unclosed_code_block() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_text("```python\nprint(1)\n")

    segments = assembler.segments
    assert len(segments) == 1
    assert segments[0].text == "```python\nprint(1)\n"
    assert segments[0].frozen is False


def test_markdown_does_not_freeze_list_item_before_indented_continuation() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_text("1. item\n\n")

    segments = assembler.segments
    assert len(segments) == 1
    assert segments[0].text == "1. item\n\n"
    assert segments[0].frozen is False

    assembler.handle_text("   continuation")

    segments = assembler.segments
    assert len(segments) == 1
    assert segments[0].text == "1. item\n\n   continuation"
    assert segments[0].frozen is False
