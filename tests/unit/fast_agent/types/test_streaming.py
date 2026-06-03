from fast_agent.types.streaming import (
    DEFAULT_STREAMING_MODE,
    STREAMING_MODE_HELP,
    STREAMING_MODES,
    normalize_streaming_mode,
)


def test_streaming_mode_help_matches_declared_modes() -> None:
    assert STREAMING_MODE_HELP == "|".join(STREAMING_MODES)


def test_normalize_streaming_mode_accepts_known_modes() -> None:
    assert normalize_streaming_mode(" PLAIN ") == "plain"
    assert normalize_streaming_mode("NoNe") == "none"


def test_normalize_streaming_mode_defaults_unknown_values() -> None:
    assert normalize_streaming_mode("sideways") == DEFAULT_STREAMING_MODE
    assert normalize_streaming_mode(None) == DEFAULT_STREAMING_MODE
