from __future__ import annotations

import importlib.util
import json
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


def test_asciinema_index_is_current() -> None:
    assert docs_assets.asciinema_index_problems() == []


def test_asciinema_index_covers_all_committed_casts() -> None:
    index = json.loads(
        (ROOT / "docs" / "docs" / "assets" / "asciinema-index.json").read_text(
            encoding="utf-8"
        )
    )
    indexed_paths = {entry["path"] for entry in index["casts"]}
    committed_paths = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "docs" / "docs" / "assets").rglob("*.cast")
    }

    assert indexed_paths == committed_paths


def test_asciinema_index_entries_have_record_commands_and_embeds() -> None:
    index = json.loads(
        (ROOT / "docs" / "docs" / "assets" / "asciinema-index.json").read_text(
            encoding="utf-8"
        )
    )
    for entry in index["casts"]:
        assert entry["present"] is True
        assert entry["record_command"]
        assert entry["embedded"], entry["path"]
        assert entry["width"] > 0
        assert entry["height"] > 0
        assert entry["problems"] == []
