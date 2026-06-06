from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def test_pytest_test_module_basenames_are_unique() -> None:
    """Avoid pytest import-file-mismatch errors from duplicate top-level test names."""
    tests_root = Path(__file__).parents[1]
    paths_by_name: dict[str, list[Path]] = defaultdict(list)

    for path in tests_root.rglob("test_*.py"):
        paths_by_name[path.name].append(path.relative_to(tests_root))

    duplicates = {name: sorted(paths) for name, paths in paths_by_name.items() if len(paths) > 1}

    assert duplicates == {}
