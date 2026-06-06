"""
Fuzzy match helpers adapted from openai/codex apply_patch (Apache 2.0).
"""

from __future__ import annotations

from collections.abc import Callable

Matcher = Callable[[list[str], list[str], int], bool]

_MATCHERS: tuple[Matcher, ...] = (
    lambda lines, pattern, index: lines[index : index + len(pattern)] == pattern,
    lambda lines, pattern, index: _matches_trim_end(lines, pattern, index),
    lambda lines, pattern, index: _matches_trim(lines, pattern, index),
    lambda lines, pattern, index: _matches_normalized(lines, pattern, index),
)


def seek_sequence(
    lines: list[str],
    pattern: list[str],
    start: int,
    eof: bool,
) -> int | None:
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None

    search_range = _search_range(lines=lines, pattern=pattern, start=start, eof=eof)
    return _first_matching_index(lines, pattern, search_range)


def _search_range(
    *,
    lines: list[str],
    pattern: list[str],
    start: int,
    eof: bool,
) -> range:
    last_start = len(lines) - len(pattern)
    search_start = last_start if eof else start
    return range(search_start, last_start + 1)


def _first_matching_index(
    lines: list[str],
    pattern: list[str],
    search_range: range,
) -> int | None:
    for matcher in _MATCHERS:
        for index in search_range:
            if matcher(lines, pattern, index):
                return index
    return None


def _matches_trim_end(lines: list[str], pattern: list[str], index: int) -> bool:
    return all(lines[index + offset].rstrip() == pat.rstrip() for offset, pat in enumerate(pattern))


def _matches_trim(lines: list[str], pattern: list[str], index: int) -> bool:
    return all(lines[index + offset].strip() == pat.strip() for offset, pat in enumerate(pattern))


def _matches_normalized(lines: list[str], pattern: list[str], index: int) -> bool:
    for offset, pat in enumerate(pattern):
        if _normalise(lines[index + offset]) != _normalise(pat):
            return False
    return True


def _normalise(value: str) -> str:
    mapping = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00a0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
    }
    return "".join(mapping.get(ch, ch) for ch in value.strip())
