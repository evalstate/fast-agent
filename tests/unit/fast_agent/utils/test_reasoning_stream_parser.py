from __future__ import annotations

from fast_agent.utils.reasoning_stream_parser import ReasoningSegment, ReasoningStreamParser


def test_parser_handles_opening_think_tag_split_across_chunks() -> None:
    parser = ReasoningStreamParser()

    assert parser.feed("answer <thi") == [ReasoningSegment("answer ", is_thinking=False)]
    assert parser.feed("nk>thought") == [ReasoningSegment("thought", is_thinking=True)]
    assert parser.in_think


def test_parser_handles_closing_think_tag_split_across_chunks() -> None:
    parser = ReasoningStreamParser()

    assert parser.feed("<think>thought</thi") == [ReasoningSegment("thought", is_thinking=True)]
    assert parser.feed("nk>answer") == [ReasoningSegment("answer", is_thinking=False)]
    assert not parser.in_think


def test_parser_emits_held_partial_tag_when_it_does_not_complete() -> None:
    parser = ReasoningStreamParser()

    assert parser.feed("answer <") == [ReasoningSegment("answer ", is_thinking=False)]
    assert parser.feed("not a tag") == [ReasoningSegment("<not a tag", is_thinking=False)]


def test_parser_flushes_pending_text_with_current_reasoning_state() -> None:
    parser = ReasoningStreamParser()

    assert parser.feed("<think>partial thought <") == [
        ReasoningSegment("partial thought ", is_thinking=True)
    ]
    assert parser.flush() == [ReasoningSegment("<", is_thinking=True)]
    assert not parser.has_pending_text
