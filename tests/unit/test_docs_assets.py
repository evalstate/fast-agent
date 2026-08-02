from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOCS_ASSETS_PATH = ROOT / "scripts" / "docs_assets.py"


def _load_docs_assets() -> Any:
    spec = importlib.util.spec_from_file_location("docs_assets", DOCS_ASSETS_PATH)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["docs_assets"] = module
    loader.exec_module(module)
    return module


docs_assets = _load_docs_assets()


def _cast_output(path: Path) -> str:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    return "".join(
        event[2]
        for event in events
        if isinstance(event, list)
        and len(event) >= 3
        and event[1] == "o"
        and isinstance(event[2], str)
    )


def test_asciinema_index_is_current() -> None:
    assert docs_assets.asciinema_index_problems() == []


def test_mcp_inspect_casts_show_modern_progress_and_legacy_health() -> None:
    modern = _cast_output(ROOT / "docs" / "docs" / "assets" / "tui" / "hf-image-generation.cast")
    legacy = _cast_output(ROOT / "docs" / "docs" / "assets" / "mcp" / "mcp-inspect-legacy.cast")

    assert "Connected MCP server 'hf'" in modern
    assert "$HF_TOKEN" in modern
    assert re.search(r"\bhf_[A-Za-z0-9]{20,}\b", modern) is None
    assert "gr1_z_image_turbo_generate - Progress: Step" in modern
    assert "[IMAGE 1:" in modern
    assert "2026-07-28 (modern)" in modern
    assert "LISTEN (SSE)" in modern
    assert "notif" in modern

    assert "Docs Legacy Remote" in legacy
    assert "2025-11-25 (forced legacy)" in legacy
    assert "docs-legacy-session" in legacy
    assert "health" in legacy
    assert "interval: 1s" in legacy
    assert "Warning" not in legacy
    assert "Traceback" not in legacy


def test_mcp_inspect_recorders_use_high_resolution_timelines() -> None:
    scenarios = docs_assets._scenarios()

    modern_script = docs_assets._record_script(scenarios["hf-image-generation"])
    legacy_script = docs_assets._record_script(scenarios["mcp-inspect-legacy"])

    for script in (modern_script, legacy_script):
        assert "steps: 60" in script
        assert "step_seconds: 1" in script


def test_asciinema_index_covers_all_committed_casts() -> None:
    index = json.loads(
        (ROOT / "docs" / "docs" / "assets" / "asciinema-index.json").read_text(encoding="utf-8")
    )
    indexed_paths = {entry["path"] for entry in index["casts"]}
    committed_paths = {
        str(path.relative_to(ROOT)) for path in (ROOT / "docs" / "docs" / "assets").rglob("*.cast")
    }

    assert indexed_paths == committed_paths


def test_asciinema_index_entries_have_record_commands_and_embeds() -> None:
    index = json.loads(
        (ROOT / "docs" / "docs" / "assets" / "asciinema-index.json").read_text(encoding="utf-8")
    )
    for entry in index["casts"]:
        assert entry["present"] is True
        assert entry["record_command"]
        assert entry["embedded"], entry["path"]
        assert entry["width"] > 0
        assert entry["height"] > 0
        assert entry["problems"] == []
