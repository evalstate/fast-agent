"""The surface fixture refresher rewrites digests as one simultaneous replacement."""

from tests.unit.fast_agent.ptc.surface_fixtures import replace_digests


def test_chained_digests_are_each_replaced_once() -> None:
    # Row one moves to row two's old digest while row two moves on: X -> A, A -> B.
    text = '[{"sha256": "X"}, {"sha256": "A"}]'

    assert replace_digests(text, {"X": "A", "A": "B"}) == '[{"sha256": "A"}, {"sha256": "B"}]'
