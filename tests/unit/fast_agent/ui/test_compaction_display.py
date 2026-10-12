"""Compaction result rendering shared by /compact and auto-compaction."""

from __future__ import annotations

import json

import pytest
from mcp_types import TextContent

from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL
from fast_agent.history.compaction import CompactionResult
from fast_agent.llm.compaction_support import CompactionStrategy
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.compaction_display import compaction_summary_lines
from fast_agent.ui.history_display_rows import build_history_rows


def _result(strategy: CompactionStrategy, checkpoint_bytes: int | None = None) -> CompactionResult:
    return CompactionResult(
        agent_name="agent",
        summary_text="checkpoint",
        messages_before=4,
        messages_after=1,
        tokens_before=4_344,
        tokens_after_estimate=120,
        context_window=272_000,
        archive_file=None,
        strategy=strategy,
        checkpoint_bytes=checkpoint_bytes,
    )


def test_summary_result_shows_the_estimate() -> None:
    lines = [line.plain for line in compaction_summary_lines(_result(CompactionStrategy.SUMMARY))]

    assert "~120 tokens" in lines[1]
    assert not any("provider checkpoint" in line for line in lines)


def test_provider_checkpoint_result_shows_no_estimate_and_its_size() -> None:
    lines = [
        line.plain
        for line in compaction_summary_lines(
            _result(CompactionStrategy.OPENAI_COMPACTION, checkpoint_bytes=6_144)
        )
    ]

    # An opaque checkpoint's token cost is unknown until the provider reports usage.
    assert "→ — tokens" in lines[1]
    assert "──.─%" in lines[0] and " est" not in lines[0]
    assert "provider checkpoint (openai_compaction, 6.0 KB encrypted)" in lines[2]


def _checkpoint_row(metadata: object) -> str:
    message = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="[COMPACTED HISTORY] checkpoint")],
        channels={
            FAST_AGENT_COMPACTION_CHANNEL: [TextContent(type="text", text=json.dumps(metadata))]
        },
    )
    [row] = build_history_rows([message])
    return row.preview


def test_history_names_provider_checkpoint_size_and_archive() -> None:
    preview = _checkpoint_row(
        {
            "strategy": "openai_compaction",
            "checkpoint_bytes": 6_144,
            "archive_file": "compacted_20261012_agent.json",
        }
    )

    assert preview == (
        "openai_compaction checkpoint · 6.0 KB encrypted · archive compacted_20261012_agent.json"
    )


def test_history_previews_summary_text() -> None:
    assert _checkpoint_row({"strategy": "summary"}) == "checkpoint"


def test_history_preserves_unknown_native_strategy() -> None:
    assert _checkpoint_row({"strategy": "future_provider_checkpoint"}) == (
        "future_provider_checkpoint checkpoint · no archive"
    )


@pytest.mark.parametrize("metadata", [None, [], {}, {"strategy": 123}])
def test_history_falls_back_to_visible_text_for_unusable_metadata(metadata: object) -> None:
    assert _checkpoint_row(metadata) == "checkpoint"


def test_history_ignores_untyped_optional_checkpoint_fields() -> None:
    assert (
        _checkpoint_row(
            {
                "strategy": "future_provider_checkpoint",
                "checkpoint_bytes": "large",
                "archive_file": 123,
            }
        )
        == "future_provider_checkpoint checkpoint · no archive"
    )
