"""The surface fixture refresher rewrites each row's digest in place."""

import pytest

from tests.unit.fast_agent.ptc.surface_fixtures import replace_digests

X, A, B = "0" * 64, "a" * 64, "b" * 64


def test_rows_are_replaced_independently() -> None:
    # Rows that shared a digest may diverge, and a new digest may equal another row's old one.
    text = f'[{{"sha256": "{X}"}}, {{"sha256": "{X}"}}, {{"sha256": "{A}"}}]'

    assert replace_digests(text, [A, X, B]) == (
        f'[{{"sha256": "{A}"}}, {{"sha256": "{X}"}}, {{"sha256": "{B}"}}]'
    )


def test_row_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="1 digest fields for 2 rows"):
        replace_digests(f'[{{"sha256": "{X}"}}]', [A, B])
