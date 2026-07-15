from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    TextContent,
)

from fast_agent.constants import FAST_AGENT_USAGE
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import save_json
from fast_agent.session import (
    SessionAgentSnapshot,
    SessionContinuationSnapshot,
    SessionSnapshot,
    SessionTraceExporter,
)
from fast_agent.session.session_manager import SessionManager
from fast_agent.session.token_accounting_validation import (
    ArtifactValidationError,
    validate_artifacts,
)
from fast_agent.session.trace_export_models import ExportRequest

if TYPE_CHECKING:
    from pathlib import Path


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
    cumulative_input: int,
    cumulative_output: int,
    cumulative_cached: int,
    cumulative_reasoning: int,
) -> TextContent:
    return TextContent(
        type="text",
        text=json.dumps(
            {
                "schema": "fast-agent.usage/v2",
                "provider_attempts": [
                    {
                        "provider": "openai",
                        "usage_schema": "openai-chat",
                        "model": "gpt-5.6",
                        "prompt": {
                            "total": input_tokens,
                            "cache_read": cached_tokens,
                            "cache_write": 0,
                        },
                        "completion": {
                            "total": output_tokens,
                            "reasoning": reasoning_tokens,
                        },
                        "tool_calls": 0,
                        "raw_usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "prompt_tokens_details": {
                                "cached_tokens": cached_tokens,
                                "cache_write_tokens": 0,
                            },
                            "completion_tokens_details": {
                                "reasoning_tokens": reasoning_tokens,
                            },
                        },
                    },
                ],
            }
        ),
    )


def _write_session(tmp_path: Path) -> tuple[SessionManager, Path]:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session_id = "2607151200-accounting"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    call_id = "call-1"
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="run the tool")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="running")],
            tool_calls={
                call_id: CallToolRequest(
                    params=CallToolRequestParams(
                        name="execute",
                        arguments={"command": "printf ok"},
                    )
                )
            },
            channels={
                FAST_AGENT_USAGE: [
                    _usage(
                        input_tokens=100,
                        output_tokens=20,
                        cached_tokens=10,
                        reasoning_tokens=5,
                        cumulative_input=100,
                        cumulative_output=20,
                        cumulative_cached=10,
                        cumulative_reasoning=5,
                    )
                ]
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                call_id: CallToolResult(
                    content=[TextContent(type="text", text="ok")],
                    isError=False,
                )
            },
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            channels={
                FAST_AGENT_USAGE: [
                    _usage(
                        input_tokens=150,
                        output_tokens=10,
                        cached_tokens=20,
                        reasoning_tokens=0,
                        cumulative_input=250,
                        cumulative_output=30,
                        cumulative_cached=30,
                        cumulative_reasoning=5,
                    )
                ]
            },
        ),
    ]
    save_json(messages, str(session_dir / "history_agent.json"))
    snapshot = SessionSnapshot(
        session_id=session_id,
        created_at=datetime(2026, 7, 15, 12, 0),
        last_activity=datetime(2026, 7, 15, 12, 1),
        continuation=SessionContinuationSnapshot(
            active_agent="agent",
            agents={
                "agent": SessionAgentSnapshot(
                    history_file="history_agent.json",
                    model="gpt-5.6",
                    model_spec="codexresponses.gpt-5.6?reasoning=low",
                    provider="openai",
                )
            },
        ),
    )
    (session_dir / "session.json").write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return manager, session_dir


def _export(
    manager: SessionManager,
    session_dir: Path,
    path: Path,
    export_format: str,
) -> None:
    SessionTraceExporter(session_manager=manager).export(
        ExportRequest(
            target=session_dir,
            agent_name="agent",
            output_path=path,
            format=export_format,
        )
    )


def test_validate_token_accounting_artifacts(tmp_path: Path) -> None:
    manager, session_dir = _write_session(tmp_path)
    codex_path = tmp_path / "trace.jsonl"
    atif_path = tmp_path / "trace.json"
    _export(manager, session_dir, codex_path, "codex")
    _export(manager, session_dir, atif_path, "atif")

    report = validate_artifacts(
        session_dir=session_dir,
        codex_path=codex_path,
        atif_path=atif_path,
        require_cache=True,
        require_tool=True,
    )

    assert report.session.usage_records == 2
    assert report.codex.token_records == 2
    assert report.codex.tool_calls == 1
    assert report.atif.metric_steps == 2
    assert report.atif.cached_tokens == 30


def test_validate_token_accounting_rejects_incorrect_codex_total(tmp_path: Path) -> None:
    manager, session_dir = _write_session(tmp_path)
    codex_path = tmp_path / "trace.jsonl"
    atif_path = tmp_path / "trace.json"
    _export(manager, session_dir, codex_path, "codex")
    _export(manager, session_dir, atif_path, "atif")

    records = [json.loads(line) for line in codex_path.read_text(encoding="utf-8").splitlines()]
    token_record = next(
        record
        for record in records
        if record["type"] == "event_msg" and record["payload"]["type"] == "token_count"
    )
    token_record["payload"]["info"]["last_token_usage"]["total_tokens"] = 999
    codex_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="total_tokens"):
        validate_artifacts(
            session_dir=session_dir,
            codex_path=codex_path,
            atif_path=atif_path,
        )
