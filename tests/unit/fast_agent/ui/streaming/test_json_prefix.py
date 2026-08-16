import json

import pytest

from fast_agent.ui.streaming.json_prefix import JsonPrefixFormatter, format_json_prefix


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("{}", "{}"),
        ("[]", "[]"),
        (
            '{"empty_array":[],"empty_object":{}}',
            '{\n  "empty_array": [],\n  "empty_object": {}\n}',
        ),
        (
            '{"key":"日本語","items":[1,true,null]}',
            '{\n  "key": "日本語",\n  "items": [\n    1,\n    true,\n    null\n  ]\n}',
        ),
        (
            '{"nested":{"items":[1,{"two":2}',
            '{\n  "nested": {\n    "items": [\n      1,\n      {\n        "two": 2\n      }',
        ),
        ('{"message":"line\\n\\u65e5\\', '{\n  "message": "line\\n\\u65e5\\'),
        ('{"message":"\\u6', '{\n  "message": "\\u6'),
        ('{"number":-12.3e+', '{\n  "number": -12.3e+'),
        ('{"number":-', '{\n  "number": -'),
        ('{"literal":tru', '{\n  "literal": tru'),
        ('{"key":', '{\n  "key": '),
    ],
)
def test_format_json_prefix_formats_complete_and_incomplete_prefixes(
    source: str, expected: str
) -> None:
    assert format_json_prefix(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "",
        "null",
        '{"key":}',
        '{"key" "value"}',
        '{"key":01}',
        '{"key":"\\x"}',
        '{"key":"\\u6x"}',
        '{"key":true false}',
        '{"key":1]}',
        "{}{}",
        '{"key":1,}',
        "[1.e]",
        "[1.e+]",
        "[1.e2]",
        "[1٢]",
    ],
)
def test_format_json_prefix_rejects_impossible_json(source: str) -> None:
    assert format_json_prefix(source) is None


def test_format_json_prefix_preserves_non_whitespace_tokens_without_synthesis() -> None:
    source = '{"é\\u0301":"日本語\\n","number":-12.30e+4,"nested":[true,null]}'
    formatted = format_json_prefix(source)

    assert formatted is not None
    assert "".join(character for character in formatted if not character.isspace()) == source
    assert json.loads(formatted) == json.loads(source)


def test_format_json_prefix_is_chunk_boundary_independent() -> None:
    source = '{"key":"a\\u65e5","number":-12.3e-4,"nested":[1,{"value":false}]}'
    expected = format_json_prefix(source)
    formatter = JsonPrefixFormatter()

    for end in range(1, len(source) + 1):
        assert format_json_prefix(source[:end]) is not None
        assert formatter.append(source[end - 1]) == format_json_prefix(source[:end])
    assert format_json_prefix(source) == expected
    assert formatter.formatted == expected
