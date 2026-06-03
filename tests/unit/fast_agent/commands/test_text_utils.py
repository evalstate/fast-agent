from fast_agent.commands.handlers._text_utils import truncate_description


def test_truncate_description_returns_stripped_short_description() -> None:
    assert truncate_description("  short description  ", char_limit=50) == "short description"


def test_truncate_description_returns_empty_for_blank_description() -> None:
    assert truncate_description("   ", char_limit=50) == ""


def test_truncate_description_uses_word_boundary_near_limit() -> None:
    text = "first second third fourth fifth"

    assert truncate_description(text, char_limit=20) == "first second third..."


def test_truncate_description_handles_nonpositive_limit() -> None:
    assert truncate_description("description", char_limit=0) == ""
    assert truncate_description("description", char_limit=-5) == ""
