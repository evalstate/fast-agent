from dataclasses import dataclass
from pathlib import Path

from fast_agent.marketplace.selection import (
    select_one_by_name_or_index,
    select_updates_by_name_or_index,
)


@dataclass(frozen=True)
class Entry:
    name: str
    path: Path


def _names(entry: Entry) -> tuple[str, str]:
    return (entry.name, entry.path.name)


def test_select_one_by_name_or_index_uses_one_based_index() -> None:
    entries = [Entry("alpha", Path("a")), Entry("beta", Path("b"))]

    assert select_one_by_name_or_index(entries, "2", names=_names) == entries[1]


def test_select_one_by_name_or_index_matches_any_name_case_insensitively() -> None:
    entry = Entry("Alpha", Path("plugin-dir"))

    assert select_one_by_name_or_index([entry], "PLUGIN-DIR", names=_names) == entry


def test_select_one_by_name_or_index_normalizes_candidate_names() -> None:
    entry = Entry(" Alpha ", Path("plugin-dir"))

    assert select_one_by_name_or_index([entry], "alpha", names=_names) == entry


def test_select_one_by_name_or_index_rejects_ambiguous_name_match() -> None:
    entries = [Entry("alpha", Path("first")), Entry("alpha", Path("second"))]

    assert select_one_by_name_or_index(entries, "alpha", names=_names) is None
    assert select_one_by_name_or_index(entries, "2", names=_names) == entries[1]


def test_select_one_by_name_or_index_rejects_empty_and_out_of_range_selectors() -> None:
    entries = [Entry("alpha", Path("a"))]

    assert select_one_by_name_or_index(entries, " ", names=_names) is None
    assert select_one_by_name_or_index(entries, "2", names=_names) is None


def test_select_one_by_name_or_index_allows_numeric_names_when_index_is_out_of_range() -> None:
    entry = Entry("2", Path("numeric-name"))

    assert select_one_by_name_or_index([entry], "2", names=_names) == entry


def test_select_one_by_name_or_index_prefers_valid_index_over_numeric_name() -> None:
    entries = [Entry("2", Path("numeric-name")), Entry("beta", Path("b"))]

    assert select_one_by_name_or_index(entries, "2", names=_names) == entries[1]


def test_select_updates_by_name_or_index_supports_all_and_single_matches() -> None:
    updates = [Entry("alpha", Path("a")), Entry("beta", Path("b"))]

    assert select_updates_by_name_or_index(updates, " ALL ", names=_names) == updates
    assert select_updates_by_name_or_index(updates, "B", names=_names) == [updates[1]]
    assert select_updates_by_name_or_index(updates, "missing", names=_names) == []
