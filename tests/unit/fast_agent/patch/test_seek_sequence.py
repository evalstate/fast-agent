from __future__ import annotations

from fast_agent.patch.seek_sequence import seek_sequence


def test_seek_sequence_prefers_exact_match() -> None:
    lines = ["alpha", " beta ", "beta"]

    assert seek_sequence(lines, ["beta"], start=0, eof=False) == 2


def test_seek_sequence_matches_trimmed_trailing_whitespace() -> None:
    assert seek_sequence(["alpha", "beta  "], ["beta"], start=0, eof=False) == 1


def test_seek_sequence_matches_fully_trimmed_lines() -> None:
    assert seek_sequence(["alpha", "  beta"], ["beta"], start=0, eof=False) == 1


def test_seek_sequence_matches_normalized_unicode_punctuation() -> None:
    assert seek_sequence(["alpha", "a\u2014b"], ["a-b"], start=0, eof=False) == 1


def test_seek_sequence_eof_searches_only_final_possible_position() -> None:
    lines = ["target", "other", "target"]

    assert seek_sequence(lines, ["target"], start=0, eof=True) == 2
