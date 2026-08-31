from fast_agent.types.streaming import (
    DEFAULT_STREAMING_MODE,
    normalize_streaming_mode,
)


def test_normalize_streaming_mode_accepts_known_modes() -> None:
    assert normalize_streaming_mode(" PLAIN ") == "plain"
    assert normalize_streaming_mode("NoNe") == "none"


def test_normalize_streaming_mode_defaults_unknown_values() -> None:
    assert normalize_streaming_mode("sideways") == DEFAULT_STREAMING_MODE
    assert normalize_streaming_mode(None) == DEFAULT_STREAMING_MODE
