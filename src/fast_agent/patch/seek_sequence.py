from __future__ import annotations


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

    search_start = len(lines) - len(pattern) if eof and len(lines) >= len(pattern) else start
    last_start = len(lines) - len(pattern)

    for index in range(search_start, last_start + 1):
        if lines[index : index + len(pattern)] == pattern:
            return index

    for index in range(search_start, last_start + 1):
        if _matches_trim_end(lines, pattern, index):
            return index

    for index in range(search_start, last_start + 1):
        if _matches_trim(lines, pattern, index):
            return index

    for index in range(search_start, last_start + 1):
        if _matches_normalized(lines, pattern, index):
            return index

    return None


def _matches_trim_end(lines: list[str], pattern: list[str], index: int) -> bool:
    for offset, pat in enumerate(pattern):
        if lines[index + offset].rstrip() != pat.rstrip():
            return False
    return True


def _matches_trim(lines: list[str], pattern: list[str], index: int) -> bool:
    for offset, pat in enumerate(pattern):
        if lines[index + offset].strip() != pat.strip():
            return False
    return True


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
        "\u201A": "'",
        "\u201B": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u201F": '"',
        "\u00A0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200A": " ",
        "\u202F": " ",
        "\u205F": " ",
        "\u3000": " ",
    }
    return "".join(mapping.get(ch, ch) for ch in value.strip())
