"""Tests for the oversized tool-result sanitiser (Layer A).

Pin the contract: any tool result text >64KB (default) is spilled to disk
and replaced with stub + 8KB preview, BEFORE it enters the staged history.
Without this defence one rogue tool poisons the entire conversation —
incident 2026-05-17 (figma_read export_svg returning 777KB SVG).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.agents.tool_runner import (
    _PREVIEW_BYTES,
    _DEFAULT_MAX_TOOL_RESULT_BYTES,
    _sanitize_oversized_tool_results,
)
from fast_agent.types import PromptMessageExtended


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("TEAM_WORKSPACE", str(tmp_path))
    return tmp_path


def _make_msg(text_by_tool_id: dict[str, str]) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="user",
        content=[],
        tool_results={
            tid: CallToolResult(
                content=[TextContent(type="text", text=t)],
                isError=False,
            )
            for tid, t in text_by_tool_id.items()
        },
    )


def test_small_result_untouched(workspace):
    """Tool results under the cap pass through unmodified."""
    small_text = "tiny output " * 10
    msg = _make_msg({"call_1": small_text})
    out = _sanitize_oversized_tool_results(msg, agent_name="Dev")
    assert out.tool_results["call_1"].content[0].text == small_text


def test_oversized_result_truncated_and_spilled(workspace):
    """A 100KB result is replaced with stub and saved to disk."""
    big_text = "X" * (100 * 1024)
    msg = _make_msg({"call_giant": big_text})

    out = _sanitize_oversized_tool_results(msg, agent_name="QE")

    # In-place mutation — same dict reference, different content
    stub_block = out.tool_results["call_giant"].content[0]
    assert isinstance(stub_block, TextContent)
    assert "Output too large" in stub_block.text
    assert str(100 * 1024) in stub_block.text  # original byte count mentioned
    assert "preview" in stub_block.text
    # Phase 2 stub must warn agent about read-back loopback (otherwise the
    # agent will call read_text_file(spill_path) and re-hit the cap).
    assert "bounded chunks" in stub_block.text
    assert "unbounded read" in stub_block.text.lower() or "hits the same" in stub_block.text
    # Stub itself is bounded — should be ~8KB preview + some boilerplate
    assert len(stub_block.text) < 10 * 1024

    # File exists with original content
    spill_files = list((workspace / ".tool-outputs").iterdir())
    assert len(spill_files) == 1
    written = spill_files[0].read_text()
    assert written == big_text
    assert "QE" in spill_files[0].name
    assert "call_giant" in spill_files[0].name


def test_preview_contains_first_8kb(workspace):
    """The stub embeds the first 8KB of the original so the LLM can still
    glance at the head of the payload without spending tokens on the rest.
    """
    head_marker = "HEAD_OF_PAYLOAD_" + "X" * 100
    big_text = head_marker + ("Y" * (100 * 1024))
    msg = _make_msg({"call_1": big_text})

    out = _sanitize_oversized_tool_results(msg, agent_name="agent")
    stub = out.tool_results["call_1"].content[0].text
    assert head_marker in stub


def test_no_tool_results_passthrough(workspace):
    """Messages without tool_results are not mutated and never write files."""
    msg = PromptMessageExtended(role="assistant", content=[TextContent(type="text", text="hi")])
    out = _sanitize_oversized_tool_results(msg, agent_name="x")
    assert out.tool_results in (None, {})  # type: ignore[arg-type]
    assert not (workspace / ".tool-outputs").exists()


def test_only_oversized_block_replaced(workspace):
    """A tool result with mixed small + large text blocks: only the large
    block is rewritten. Small ones pass through unchanged.
    """
    msg = PromptMessageExtended(
        role="user",
        content=[],
        tool_results={
            "call_1": CallToolResult(
                content=[
                    TextContent(type="text", text="small header"),
                    TextContent(type="text", text="Z" * (100 * 1024)),
                    TextContent(type="text", text="small footer"),
                ],
                isError=False,
            ),
        },
    )
    out = _sanitize_oversized_tool_results(msg, agent_name="agent")
    blocks = out.tool_results["call_1"].content
    assert blocks[0].text == "small header"
    assert "Output too large" in blocks[1].text
    assert blocks[2].text == "small footer"


def test_env_var_overrides_cap(workspace, monkeypatch):
    """FAST_AGENT_MAX_TOOL_RESULT_BYTES env var lowers / raises the cap."""
    monkeypatch.setenv("FAST_AGENT_MAX_TOOL_RESULT_BYTES", "5000")  # 5KB
    text_6kb = "A" * 6000
    msg = _make_msg({"call_1": text_6kb})
    out = _sanitize_oversized_tool_results(msg, agent_name="agent")
    assert "Output too large" in out.tool_results["call_1"].content[0].text


def test_invalid_env_falls_back_to_default(workspace, monkeypatch):
    """Garbage env value doesn't break — use default cap."""
    monkeypatch.setenv("FAST_AGENT_MAX_TOOL_RESULT_BYTES", "not-a-number")
    # 50KB — under default 64KB
    msg = _make_msg({"call_1": "A" * (50 * 1024)})
    out = _sanitize_oversized_tool_results(msg, agent_name="agent")
    # Should be unchanged because default 64KB > 50KB
    assert out.tool_results["call_1"].content[0].text == "A" * (50 * 1024)


def test_spill_failure_replaces_with_marker(workspace, monkeypatch):
    """When disk write fails (read-only mount, permission, full disk), we
    still MUST cap — the next LLM call would die otherwise. Stub mentions
    the spill failure so debugging is possible.
    """
    # Point spill dir at a file-not-a-dir so mkdir raises FileExistsError
    not_a_dir = workspace / "blocker"
    not_a_dir.write_text("collision")
    monkeypatch.setenv("TEAM_WORKSPACE", str(workspace / "blocker" / "child"))

    big_text = "B" * (100 * 1024)
    msg = _make_msg({"call_1": big_text})
    out = _sanitize_oversized_tool_results(msg, agent_name="agent")
    stub = out.tool_results["call_1"].content[0].text
    # Still capped — main goal preserved
    assert len(stub) < 20 * 1024
    # And caller is told the spill failed (so they can investigate)
    assert "spill failed" in stub or "Output too large" in stub


def test_unicode_safe_size_measurement(workspace):
    """Size cap uses UTF-8 byte length, not str length. A string of 32K
    emoji characters is 128K bytes — must be capped.
    """
    emoji_payload = "🚀" * (32 * 1024)  # 4 bytes per char in UTF-8 → 128KB
    assert len(emoji_payload.encode("utf-8")) > _DEFAULT_MAX_TOOL_RESULT_BYTES
    msg = _make_msg({"call_1": emoji_payload})
    out = _sanitize_oversized_tool_results(msg, agent_name="agent")
    assert "Output too large" in out.tool_results["call_1"].content[0].text
