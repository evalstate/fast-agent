from __future__ import annotations

import pytest

from fast_agent.commands.option_parsing import (
    ValueOption,
    is_long_option_token,
    matches_option_token,
    read_option_token_value,
    read_value_option,
)


def test_value_option_display_name_prefers_error_name() -> None:
    assert ValueOption("message", ("--message", "-m"), error_name="--message").display_name == (
        "--message"
    )
    assert ValueOption("temp_dir", ("--temp-dir",)).display_name == "--temp-dir"


def test_read_value_option_returns_named_match_metadata() -> None:
    parsed = read_value_option(
        ["-m", "hello"],
        0,
        (ValueOption("message", ("--message", "-m"), error_name="--message"),),
    )

    assert parsed.matched is True
    assert parsed.name == "message"
    assert parsed.require_name() == "message"
    assert parsed.value == "hello"
    assert parsed.require_value() == "hello"
    assert parsed.display_name == "--message"
    assert parsed.error is None
    assert parsed.next_index == 2


def test_read_value_option_reports_named_missing_value() -> None:
    parsed = read_value_option(
        ["-m"],
        0,
        (ValueOption("message", ("--message", "-m"), error_name="--message"),),
    )

    assert parsed.matched is True
    assert parsed.name == "message"
    assert parsed.error == "Missing value for --message"
    assert parsed.next_index == 0


def test_read_value_option_reports_unmatched_token() -> None:
    parsed = read_value_option(
        ["--other"],
        0,
        (ValueOption("message", ("--message", "-m"), error_name="--message"),),
    )

    assert parsed.matched is False
    assert parsed.name is None
    assert parsed.value is None
    assert parsed.error is None
    assert parsed.next_index == 0


def test_read_value_option_requires_matched_name_and_value() -> None:
    missing = read_value_option(
        ["-m"],
        0,
        (ValueOption("message", ("--message", "-m"), error_name="--message"),),
    )
    unmatched = read_value_option(
        ["--other"],
        0,
        (ValueOption("message", ("--message", "-m"), error_name="--message"),),
    )

    with pytest.raises(ValueError, match="option value was not parsed"):
        missing.require_value()
    with pytest.raises(ValueError, match="option name was not parsed"):
        unmatched.require_name()


def test_read_option_token_value_consumes_split_value() -> None:
    parsed = read_option_token_value(["--message", "hello"], 0, ("--message", "-m"))

    assert parsed.matched is True
    assert parsed.value == "hello"
    assert parsed.require_value() == "hello"
    assert parsed.error is None
    assert parsed.next_index == 2


def test_read_option_token_value_consumes_equals_value() -> None:
    parsed = read_option_token_value(["--message=hello"], 0, ("--message", "-m"))

    assert parsed.matched is True
    assert parsed.value == "hello"
    assert parsed.error is None
    assert parsed.next_index == 1


def test_read_option_token_value_strips_split_value() -> None:
    parsed = read_option_token_value(["--message", "  hello  "], 0, ("--message",))

    assert parsed.matched is True
    assert parsed.value == "hello"
    assert parsed.error is None
    assert parsed.next_index == 2


def test_read_option_token_value_strips_equals_value() -> None:
    parsed = read_option_token_value(["--message=  hello  "], 0, ("--message",))

    assert parsed.matched is True
    assert parsed.value == "hello"
    assert parsed.error is None
    assert parsed.next_index == 1


def test_read_option_token_value_uses_error_name_for_alias() -> None:
    parsed = read_option_token_value(["-m"], 0, ("--message", "-m"), error_name="--message")

    assert parsed.matched is True
    assert parsed.error == "Missing value for --message"
    assert parsed.next_index == 0


def test_read_option_token_value_rejects_flag_like_split_value_by_default() -> None:
    parsed = read_option_token_value(["--message", "--no-push"], 0, ("--message",))

    assert parsed.matched is True
    assert parsed.error == "Missing value for --message"


def test_read_option_token_value_rejects_blank_split_value() -> None:
    parsed = read_option_token_value(["--message", "   "], 0, ("--message",))

    assert parsed.matched is True
    assert parsed.error == "Missing value for --message"


def test_read_option_token_value_rejects_blank_equals_value() -> None:
    parsed = read_option_token_value(["--message=   "], 0, ("--message",))

    assert parsed.matched is True
    assert parsed.error == "Missing value for --message"


def test_read_option_token_value_allows_flag_like_split_value_when_requested() -> None:
    parsed = read_option_token_value(
        ["--title", "-draft"],
        0,
        ("--title",),
        allow_flag_like_value=True,
    )

    assert parsed.value == "-draft"
    assert parsed.next_index == 2


def test_read_option_token_value_allows_flag_like_equals_value() -> None:
    parsed = read_option_token_value(["--title=-draft"], 0, ("--title",))

    assert parsed.value == "-draft"
    assert parsed.next_index == 1


def test_read_option_token_value_reports_unmatched_token() -> None:
    parsed = read_option_token_value(["--other"], 0, ("--message",))

    assert parsed.matched is False
    assert parsed.value is None
    assert parsed.error is None
    assert parsed.next_index == 0


@pytest.mark.parametrize("index", [-1, 1])
def test_read_option_token_value_treats_invalid_index_as_unmatched(index: int) -> None:
    parsed = read_option_token_value(["--message"], index, ("--message",))

    assert parsed.matched is False
    assert parsed.value is None
    assert parsed.error is None
    assert parsed.next_index == index


def test_read_option_token_value_requires_parsed_value() -> None:
    parsed = read_option_token_value(["--message"], 0, ("--message",))

    with pytest.raises(ValueError, match="option value was not parsed"):
        parsed.require_value()


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("--all", True),
        ("--target=env", True),
        ("--=env", False),
        ("--", False),
        ("-", False),
        ("-m", False),
        ("value", False),
    ],
)
def test_is_long_option_token_matches_double_dash_options(token: str, expected: bool) -> None:
    assert is_long_option_token(token) is expected


@pytest.mark.parametrize(
    ("token", "option_names", "expected"),
    [
        ("--message", ("--message", "-m"), True),
        ("--message=hello", ("--message", "-m"), True),
        ("-m", ("--message", "-m"), True),
        ("-m=hello", ("--message", "-m"), False),
        ("--message-extra", ("--message",), False),
        ("--other=hello", ("--message",), False),
    ],
)
def test_matches_option_token_matches_split_and_equals_forms(
    token: str,
    option_names: tuple[str, ...],
    expected: bool,
) -> None:
    assert matches_option_token(token, option_names) is expected
