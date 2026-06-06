import pytest

from fast_agent.tools.filesystem_tool_args import (
    is_permission_error,
    parse_read_text_file_arguments,
    parse_write_text_file_arguments,
    permission_denied_message,
)


def test_parse_read_text_file_arguments_normalizes_path_and_bounds() -> None:
    parsed = parse_read_text_file_arguments({"path": "  sample.txt  ", "line": 2, "limit": 3})

    assert parsed.payload == {"path": "  sample.txt  ", "line": 2, "limit": 3}
    assert parsed.path == "sample.txt"
    assert parsed.line == 2
    assert parsed.limit == 3


@pytest.mark.parametrize(
    "arguments",
    [
        None,
        {"path": "   "},
        {"path": "sample.txt", "line": True},
        {"path": "sample.txt", "limit": 0},
    ],
)
def test_parse_read_text_file_arguments_rejects_invalid_values(arguments) -> None:
    with pytest.raises(ValueError, match="Error:"):
        parse_read_text_file_arguments(arguments)


def test_parse_write_text_file_arguments_allows_empty_content() -> None:
    parsed = parse_write_text_file_arguments({"path": "  empty.txt  ", "content": ""})

    assert parsed.payload == {"path": "  empty.txt  ", "content": ""}
    assert parsed.path == "empty.txt"
    assert parsed.content == ""


@pytest.mark.parametrize(
    "arguments",
    [
        None,
        {"path": "   ", "content": "x"},
        {"path": "sample.txt"},
        {"path": "sample.txt", "content": 123},
    ],
)
def test_parse_write_text_file_arguments_rejects_invalid_values(arguments) -> None:
    with pytest.raises(ValueError, match="Error:"):
        parse_write_text_file_arguments(arguments)


def test_permission_error_helpers_normalize_filesystem_denials() -> None:
    assert is_permission_error(PermissionError("blocked"))
    assert not is_permission_error(FileNotFoundError("missing"))
    assert permission_denied_message("secret.txt") == "Permission denied for file: secret.txt."
