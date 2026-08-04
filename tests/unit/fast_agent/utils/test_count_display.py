from fast_agent.utils.count_display import (
    format_compact_count,
    format_count,
    format_count_breakdown,
    format_count_parts,
    plural_label,
)


def test_format_compact_count_keeps_counts_below_one_million_exact() -> None:
    assert format_compact_count(999_999) == "999,999"


def test_format_compact_count_uses_three_significant_digits() -> None:
    assert format_compact_count(1_000_000) == "1.00M"
    assert format_compact_count(1_234_567) == "1.23M"
    assert format_compact_count(12_345_678) == "12.3M"
    assert format_compact_count(123_456_789) == "123M"
    assert format_compact_count(999_999_999) == "1.00B"
    assert format_compact_count(1_250_000_000) == "1.25B"


def test_format_compact_count_supports_four_significant_digits() -> None:
    assert format_compact_count(21_524_724, significant_digits=4) == "21.52M"
    assert format_compact_count(1_234_567, significant_digits=4) == "1.235M"
    assert format_compact_count(51_651, significant_digits=4) == "51,651"


def test_plural_label_uses_singular_for_one() -> None:
    assert plural_label(1, "agent") == "agent"


def test_plural_label_uses_default_plural_for_other_counts() -> None:
    assert plural_label(0, "agent") == "agents"
    assert plural_label(2, "agent") == "agents"


def test_plural_label_uses_explicit_plural() -> None:
    assert plural_label(2, "child agent", "child agents") == "child agents"


def test_format_count_uses_grouped_count_and_plural_label() -> None:
    assert format_count(1_200, "hook") == "1,200 hooks"


def test_format_count_parts_returns_grouped_count_and_label() -> None:
    assert format_count_parts(1_200, "child", "children") == ("1,200", "children")


def test_format_count_breakdown_preserves_part_order() -> None:
    assert (
        format_count_breakdown("Messages", 5, user=2, assistant=3)
        == "Messages: 5 (user: 2, assistant: 3)"
    )


def test_format_count_breakdown_omits_empty_parentheses_without_parts() -> None:
    assert format_count_breakdown("Messages", 0) == "Messages: 0"


def test_format_count_breakdown_groups_large_counts() -> None:
    assert (
        format_count_breakdown("Messages", 12_345, user=1_200, assistant=11_145)
        == "Messages: 12,345 (user: 1,200, assistant: 11,145)"
    )
