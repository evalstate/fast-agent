from fast_agent.tools.output_truncation import truncate_text_output


def test_truncate_text_output_retains_head_and_tail() -> None:
    result = truncate_text_output(
        "abcdefghij",
        byte_limit=6,
        label="Tool result",
        guidance="Request less.",
    )

    assert result is not None
    assert result.total_bytes == 10
    assert result.retained_bytes == 6
    assert result.omitted_bytes == 4
    assert result.text.startswith("abc\n[Tool result truncated:")
    assert result.text.endswith("\nhij")
    assert "omitted 4 middle bytes. Request less." in result.text


def test_truncate_text_output_returns_none_within_budget() -> None:
    assert (
        truncate_text_output(
            "small",
            byte_limit=5,
            label="Tool result",
            guidance="Request less.",
        )
        is None
    )
