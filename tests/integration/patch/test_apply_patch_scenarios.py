from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pytest


@dataclass(frozen=True)
class SnapshotEntry:
    kind: Literal["file", "dir"]
    content: bytes | None


@pytest.mark.integration
def test_apply_patch_scenarios() -> None:
    scenarios_dir = Path(__file__).resolve().parent.parent.parent / "fixtures" / "patch" / "scenarios"
    for scenario in sorted(scenarios_dir.iterdir()):
        if scenario.is_dir():
            _run_apply_patch_scenario(scenario)


def _run_apply_patch_scenario(scenario_dir: Path) -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_dir = scenario_dir / "input"
        if input_dir.is_dir():
            _copy_dir_recursive(input_dir, tmp_path)

        patch = (scenario_dir / "patch.txt").read_text(encoding="utf-8", newline="")

        subprocess.run(
            ["uv", "run", "python", "-m", "fast_agent.patch.cli", patch],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        expected_snapshot = _snapshot_dir(scenario_dir / "expected")
        actual_snapshot = _snapshot_dir(tmp_path)

        assert (
            actual_snapshot == expected_snapshot
        ), f"Scenario {scenario_dir} did not match expected final state"


def _snapshot_dir(root: Path) -> dict[Path, SnapshotEntry]:
    entries: dict[Path, SnapshotEntry] = {}
    if root.is_dir():
        _snapshot_dir_recursive(root, root, entries)
    return entries


def _snapshot_dir_recursive(base: Path, current: Path, entries: dict[Path, SnapshotEntry]) -> None:
    for entry in current.iterdir():
        rel = entry.relative_to(base)
        if entry.is_dir():
            entries[rel] = SnapshotEntry(kind="dir", content=None)
            _snapshot_dir_recursive(base, entry, entries)
        elif entry.is_file():
            entries[rel] = SnapshotEntry(kind="file", content=entry.read_bytes())


def _copy_dir_recursive(source: Path, destination: Path) -> None:
    for entry in source.iterdir():
        dest_path = destination / entry.name
        if entry.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            _copy_dir_recursive(entry, dest_path)
        elif entry.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, dest_path)
