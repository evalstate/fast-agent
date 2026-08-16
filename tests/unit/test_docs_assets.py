from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

from click.utils import strip_ansi

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
    assert "hf_whoami" in modern
    assert "gr1_z_image_turbo_generate - Progress: Step" in modern
    assert "[IMAGE 1:" in modern
    assert "2026-07-28 (modern)" in modern
    assert "LISTEN (SSE)" in modern
    assert "POST (JSON)" in modern
    assert "POST (SSE)" in modern
    assert "tools/call:2" in modern
    assert "notif" in modern

    modern_plain = strip_ansi(modern).replace("\x1b(B", "")
    stream_start = modern_plain.index("hf__gr1_z_image_turbo_generate")
    completed_call_start = modern_plain.index(
        "agent tool (MCP) hf gr1_z_image_turbo_generate",
        stream_start,
    )
    streamed_call = modern_plain[stream_start:completed_call_start]
    assert "\r\n{\r\n" in streamed_call
    assert '\r\n  "prompt": "' in streamed_call
    assert '\r\n  "resolution": "' in streamed_call
    assert re.search(
        r"agent tool \(MCP\) hf hf_whoami · id: [^\r\n]+\r?\n\{\}",
        modern_plain,
    )
    assert re.search(
        r"agent tool \(MCP\) hf gr1_z_image_turbo_generate · id: [^\r\n]+"
        r"\r?\n\{\r?\n  \"prompt\":",
        modern_plain,
    )
    assert "\r\n…\r\n" in modern_plain

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

    assert "wait_for_prompt" in legacy_script
    assert """wait_for_pane "$SESSION" 'Docs Legacy Remote'""" in legacy_script


def test_mcp_tool_schema_recorder_uses_named_local_server() -> None:
    scenario = docs_assets._scenarios()["mcp-tool-schema"]
    script = docs_assets._record_script(scenario)

    assert "/mcp connect http://localhost:3000/mcp --name hf" in script
    assert "/tool hf__hf_whoami" in script
    assert """wait_for_pane "$SESSION" "Connected MCP server 'hf'""" in script
    assert """wait_for_pane "$SESSION" 'Structured output schema'""" in script
    assert "search-backward 'Input schema'" in script
    assert "search-forward 'Structured output schema'" in script
    assert 'send-keys -X -t "$SESSION" page-down' in script
    assert "search-backward 'Structured output schema'" in script


def test_mcp_tool_schema_cast_shows_declared_output_schema() -> None:
    path = ROOT / "docs" / "docs" / "assets" / "tui" / "mcp-tool-schema.cast"
    output = _cast_output(path)
    plain = strip_ansi(output).replace("\x1b(B", "")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    duration = events[-1][0]
    assert isinstance(duration, int | float)
    final_output = "".join(
        event[2]
        for event in events
        if isinstance(event, list)
        and len(event) >= 3
        and event[1] == "o"
        and isinstance(event[0], int | float)
        and event[0] >= duration - 3.1
        and isinstance(event[2], str)
    )
    final_plain = strip_ansi(final_output).replace("\x1b(B", "")

    assert "Connected MCP server 'hf'" in plain
    assert "/tool hf__hf_whoami" in plain
    assert "Tool schema: hf__hf_whoami" in plain
    assert "Input schema" in plain
    assert "Structured output schema" in plain
    assert "Supplied by the MCP tool declaration." in plain
    assert "Traceback" not in plain
    assert "Tool not found" not in plain
    assert duration >= 15
    assert "Structured output schema" in final_plain
    assert "Supplied by the MCP tool declaration." in final_plain


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
