from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
    TERMINAL_BYTES_PER_TOKEN,
)
from fast_agent.llm.terminal_output_limits import calculate_terminal_output_limit_for_model


def test_default_terminal_output_limit_falls_back_without_model() -> None:
    assert calculate_terminal_output_limit_for_model(None) == DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT


def test_default_terminal_output_limit_falls_back_for_unknown_model() -> None:
    assert (
        calculate_terminal_output_limit_for_model("definitely-not-a-model")
        == DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
    )


def test_default_terminal_output_limit_scaled_for_large_output_models() -> None:
    # gpt-4.1 is defined in ModelDatabase with max_output_tokens=32768.
    # With the current budgeting constants this reaches the conservative hard cap.
    limit = calculate_terminal_output_limit_for_model("gpt-4.1")
    assert limit == MAX_TERMINAL_OUTPUT_BYTE_LIMIT


def test_default_terminal_output_limit_targets_two_thirds_of_model_output() -> None:
    # openai/gpt-oss-120b has max_output_tokens=32766 in ModelDatabase.
    # Budgeting should retain roughly ~2/3 after headroom until it reaches the hard cap.
    limit = calculate_terminal_output_limit_for_model("openai/gpt-oss-120b")
    assert limit == MAX_TERMINAL_OUTPUT_BYTE_LIMIT


def test_default_terminal_output_limit_is_capped() -> None:
    # o3 is defined in ModelDatabase with max_output_tokens=100000.
    assert calculate_terminal_output_limit_for_model("o3") == MAX_TERMINAL_OUTPUT_BYTE_LIMIT


def test_default_terminal_output_limit_caps_at_about_10k_tokens() -> None:
    assert int(MAX_TERMINAL_OUTPUT_BYTE_LIMIT / TERMINAL_BYTES_PER_TOKEN) == 10_000
