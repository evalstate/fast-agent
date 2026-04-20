from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from mcp.types import (
    AnyUrl,
    AudioContent,
    BlobResourceContents,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from fast_agent.constants import FAST_AGENT_USAGE
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import save_json
from fast_agent.session import (
    SessionAgentSnapshot,
    SessionContinuationSnapshot,
    SessionRequestSettingsSnapshot,
    SessionSnapshot,
    SessionTraceExporter,
)
from fast_agent.session.session_manager import SessionManager
from fast_agent.session.trace_export_errors import (
    SessionExportAmbiguousAgentError,
    SessionExportReadError,
    SessionExportWriteError,
)
from fast_agent.session.trace_export_models import DatasetUploadResult, ExportRequest
from fast_agent.types import LlmStopReason

if TYPE_CHECKING:
    from pathlib import Path


def _write_session_snapshot(
    session_dir: Path,
    *,
    session_id: str,
    agents: dict[str, SessionAgentSnapshot],
    active_agent: str | None,
) -> None:
    snapshot = SessionSnapshot(
        session_id=session_id,
        created_at=datetime(2026, 4, 20, 13, 3, 0),
        last_activity=datetime(2026, 4, 20, 13, 8, 0),
        continuation=SessionContinuationSnapshot(
            active_agent=active_agent,
            agents=agents,
        ),
    )
    (session_dir / "session.json").write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def _write_history(path: Path, *, assistant_text: str) -> None:
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=assistant_text)],
            channels={
                "reasoning": [TextContent(type="text", text="thinking")],
            },
            stop_reason=LlmStopReason.END_TURN,
        ),
    ]
    save_json(messages, str(path))


def _build_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )


def test_session_trace_exporter_writes_codex_trace(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    _write_history(session_dir / "history_dev.json", assistant_text="done")
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={
            "dev": SessionAgentSnapshot(
                history_file="history_dev.json",
                resolved_prompt="You are dev.",
                model="gpt-5.4",
                provider="codexresponses",
                request_settings=SessionRequestSettingsSnapshot(use_history=True),
            )
        },
    )

    exporter = SessionTraceExporter(session_manager=manager)
    result = exporter.export(
        ExportRequest(
            target=session_dir / "session.json",
            agent_name=None,
            output_path=tmp_path / "trace.jsonl",
        )
    )

    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert result.record_count == 9
    assert records[0]["type"] == "session_meta"
    assert records[0]["payload"]["id"] == session_id
    assert records[0]["payload"]["base_instructions"]["text"] == "You are dev."
    assert "harness" not in records[0]
    assert records[1]["type"] == "response_item"
    assert records[1]["payload"] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "You are dev."}],
    }
    assert records[2]["type"] == "event_msg"
    assert records[2]["payload"]["type"] == "turn_started"
    assert records[3]["type"] == "event_msg"
    assert records[3]["payload"]["type"] == "user_message"
    assert records[3]["payload"]["message"] == "hello"
    assert records[4]["type"] == "turn_context"
    assert records[4]["payload"]["turn_id"] == "turn-1"
    assert "developer_instructions" not in records[4]["payload"]
    assert "approval_policy" not in records[4]["payload"]
    assert "sandbox_policy" not in records[4]["payload"]
    assert records[5]["type"] == "response_item"
    assert records[5]["payload"]["type"] == "message"
    assert records[5]["payload"]["role"] == "user"
    assert records[5]["payload"]["content"] == [{"type": "input_text", "text": "hello"}]
    assert records[6]["type"] == "response_item"
    assert records[6]["payload"] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "thinking"}],
    }
    assert "content" not in records[6]["payload"]
    assert "encrypted_content" not in records[6]["payload"]
    assert records[7]["type"] == "response_item"
    assert records[7]["payload"]["type"] == "message"
    assert records[7]["payload"]["role"] == "assistant"
    assert records[7]["payload"]["content"] == [{"type": "output_text", "text": "done"}]
    assert records[7]["payload"]["end_turn"] is True
    assert records[8]["type"] == "event_msg"
    assert records[8]["payload"]["type"] == "turn_complete"
    assert records[8]["payload"]["last_agent_message"] == "done"


def test_session_trace_exporter_writes_native_codex_tool_items(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Using tools")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="execute",
                        arguments={"command": "pwd"},
                    ),
                )
            },
            channels={
                "fast-agent-timing": [
                    TextContent(
                        type="text",
                        text='{"duration_ms": 12.5, "ttft_ms": 3.0}',
                    )
                ],
                "openai-reasoning-encrypted": [
                    TextContent(
                        type="text",
                        text='{"type":"reasoning","encrypted_content":"abc","id":"r1"}',
                    )
                ],
            },
            stop_reason=LlmStopReason.TOOL_USE,
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="process exit code was 0")],
                    isError=False,
                )
            },
        ),
    ]
    save_json(messages, str(session_dir / "history_dev.json"))
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    exporter = SessionTraceExporter(session_manager=manager)
    exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[1]["type"] == "event_msg"
    assert records[1]["payload"]["type"] == "turn_started"
    assert records[2]["type"] == "turn_context"
    assert records[3]["type"] == "response_item"
    assert records[3]["payload"] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Using tools"}],
    }
    assert records[4]["type"] == "response_item"
    assert records[4]["payload"] == {
        "type": "function_call",
        "name": "execute",
        "arguments": '{"command":"pwd"}',
        "call_id": "call_1",
    }
    assert records[5]["type"] == "response_item"
    assert records[5]["payload"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "process exit code was 0",
    }
    assert records[6]["type"] == "event_msg"
    assert records[6]["payload"]["type"] == "turn_complete"
    assert records[6]["payload"]["last_agent_message"] == "Using tools"


def test_session_trace_exporter_uses_usage_metadata_for_model_and_token_count(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            channels={
                FAST_AGENT_USAGE: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "turn": {
                                    "provider": "codexresponses",
                                    "model": "gpt-5.3-codex",
                                    "input_tokens": 120,
                                    "output_tokens": 30,
                                    "total_tokens": 150,
                                    "reasoning_tokens": 7,
                                    "display_input_tokens": 120,
                                    "cache_usage": {
                                        "cache_read_tokens": 0,
                                        "cache_write_tokens": 0,
                                        "cache_hit_tokens": 12,
                                    },
                                },
                                "summary": {
                                    "context_window_size": 400000,
                                },
                            }
                        ),
                    )
                ]
            },
            stop_reason=LlmStopReason.END_TURN,
        ),
    ]
    save_json(messages, str(session_dir / "history_dev.json"))
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    exporter = SessionTraceExporter(session_manager=manager)
    exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[0]["payload"]["model_provider"] == "codexresponses"
    assert records[1]["type"] == "event_msg"
    assert records[1]["payload"]["type"] == "turn_started"
    assert records[1]["payload"]["model_context_window"] == 400000
    assert records[3]["type"] == "turn_context"
    assert records[3]["payload"]["model"] == "gpt-5.3-codex"
    assert records[6]["type"] == "event_msg"
    assert records[6]["payload"] == {
        "type": "token_count",
        "info": {
            "last_token_usage": {
                "input_tokens": 120,
                "cached_input_tokens": 12,
                "output_tokens": 30,
                "reasoning_output_tokens": 7,
                "total_tokens": 150,
            },
            "model_context_window": 400000,
        },
    }


def test_session_trace_exporter_preserves_user_attachment_content(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="summarize these"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=AnyUrl("file:///tmp/example.py"),
                        mimeType="text/x-python",
                        text="print('hello')",
                    ),
                ),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri=AnyUrl("file:///tmp/report.pdf"),
                        mimeType="application/pdf",
                        blob="cGRm",
                    ),
                ),
                ResourceLink(
                    type="resource_link",
                    uri=AnyUrl("https://example.com/audio.mp3"),
                    mimeType="audio/mpeg",
                    name="audio.mp3",
                ),
                AudioContent(type="audio", data="d2F2", mimeType="audio/wav"),
            ],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            stop_reason=LlmStopReason.END_TURN,
        ),
    ]
    save_json(messages, str(session_dir / "history_dev.json"))
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    exporter = SessionTraceExporter(session_manager=manager)
    exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[4]["type"] == "response_item"
    assert records[4]["payload"] == {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "summarize these"},
            {
                "type": "input_text",
                "text": (
                    '<fastagent:file title="example.py" mimetype="text/x-python">\n'
                    "print('hello')\n"
                    "</fastagent:file>"
                ),
            },
            {
                "type": "input_text",
                "text": "Attached file: report.pdf (application/pdf)",
            },
            {
                "type": "input_text",
                "text": "Attached resource: audio.mp3 (audio/mpeg) — https://example.com/audio.mp3",
            },
            {
                "type": "input_text",
                "text": "Attached audio (audio/wav)",
            },
        ],
    }


def test_session_trace_exporter_preserves_non_text_tool_outputs(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Using tools")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="download",
                        arguments={"target": "report"},
                    ),
                )
            },
            stop_reason=LlmStopReason.TOOL_USE,
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "call_1": CallToolResult(
                    content=[
                        TextContent(type="text", text="Fetched resource"),
                        EmbeddedResource(
                            type="resource",
                            resource=BlobResourceContents(
                                uri=AnyUrl("file:///tmp/report.pdf"),
                                mimeType="application/pdf",
                                blob="cGRm",
                            ),
                        ),
                        ResourceLink(
                            type="resource_link",
                            uri=AnyUrl("https://example.com/audio.mp3"),
                            mimeType="audio/mpeg",
                            name="audio.mp3",
                        ),
                        AudioContent(type="audio", data="d2F2", mimeType="audio/wav"),
                    ],
                    isError=False,
                )
            },
        ),
    ]
    save_json(messages, str(session_dir / "history_dev.json"))
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    exporter = SessionTraceExporter(session_manager=manager)
    exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[5]["type"] == "response_item"
    assert records[5]["payload"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": "Fetched resource"},
            {
                "type": "input_file",
                "file_data": "cGRm",
                "filename": "report.pdf",
            },
            {
                "type": "input_file",
                "file_url": "https://example.com/audio.mp3",
            },
            {
                "type": "input_file",
                "file_data": "d2F2",
            },
        ],
    }


def test_session_trace_exporter_preserves_tool_output_item_order(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Using tools")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="download",
                        arguments={"target": "report"},
                    ),
                )
            },
            stop_reason=LlmStopReason.TOOL_USE,
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "call_1": CallToolResult(
                    content=[
                        TextContent(type="text", text="Fetched report"),
                        EmbeddedResource(
                            type="resource",
                            resource=BlobResourceContents(
                                uri=AnyUrl("file:///tmp/report-a.pdf"),
                                mimeType="application/pdf",
                                blob="YQ==",
                            ),
                        ),
                        TextContent(type="text", text="Fetched audio"),
                        ResourceLink(
                            type="resource_link",
                            uri=AnyUrl("https://example.com/audio.mp3"),
                            mimeType="audio/mpeg",
                            name="audio.mp3",
                        ),
                    ],
                    isError=False,
                )
            },
        ),
    ]
    save_json(messages, str(session_dir / "history_dev.json"))
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    exporter = SessionTraceExporter(session_manager=manager)
    exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[5]["type"] == "response_item"
    assert records[5]["payload"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": "Fetched report"},
            {
                "type": "input_file",
                "file_data": "YQ==",
                "filename": "report-a.pdf",
            },
            {"type": "input_text", "text": "Fetched audio"},
            {
                "type": "input_file",
                "file_url": "https://example.com/audio.mp3",
            },
        ],
    }


def test_session_trace_exporter_preserves_user_content_alongside_tool_outputs(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Using tools")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="execute",
                        arguments={"command": "pwd"},
                    ),
                )
            },
            stop_reason=LlmStopReason.TOOL_USE,
        ),
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Please also inspect the parent directory")],
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="process exit code was 0")],
                    isError=False,
                )
            },
        ),
    ]
    save_json(messages, str(session_dir / "history_dev.json"))
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    exporter = SessionTraceExporter(session_manager=manager)
    exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[5]["type"] == "response_item"
    assert records[5]["payload"] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Please also inspect the parent directory"}],
    }
    assert records[6]["type"] == "response_item"
    assert records[6]["payload"] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "process exit code was 0",
    }


def test_session_trace_exporter_preserves_assistant_attachment_content(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="show me the results")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[
                TextContent(type="text", text="Here they are"),
                ImageContent(type="image", data="aW1hZ2U=", mimeType="image/png"),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri=AnyUrl("file:///tmp/report.pdf"),
                        mimeType="application/pdf",
                        blob="cGRm",
                    ),
                ),
                ResourceLink(
                    type="resource_link",
                    uri=AnyUrl("https://example.com/audio.mp3"),
                    mimeType="audio/mpeg",
                    name="audio.mp3",
                ),
                AudioContent(type="audio", data="d2F2", mimeType="audio/wav"),
            ],
            stop_reason=LlmStopReason.END_TURN,
        ),
    ]
    save_json(messages, str(session_dir / "history_dev.json"))
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    exporter = SessionTraceExporter(session_manager=manager)
    exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[5]["type"] == "response_item"
    assert records[5]["payload"] == {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "Here they are"},
            {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
            {
                "type": "output_text",
                "text": "Attached file: report.pdf (application/pdf)",
            },
            {
                "type": "output_text",
                "text": "Attached resource: audio.mp3 (audio/mpeg) — https://example.com/audio.mp3",
            },
            {
                "type": "output_text",
                "text": "Attached audio (audio/wav)",
            },
        ],
        "end_turn": True,
    }


def test_session_trace_exporter_requires_agent_when_multiple_histories_exist(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    _write_history(session_dir / "history_dev.json", assistant_text="dev")
    _write_history(session_dir / "history_other.json", assistant_text="other")
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={
            "dev": SessionAgentSnapshot(history_file="history_dev.json"),
            "other": SessionAgentSnapshot(history_file="history_other.json"),
        },
    )

    exporter = SessionTraceExporter(session_manager=manager)

    with pytest.raises(SessionExportAmbiguousAgentError):
        exporter.export(
            ExportRequest(
                target=session_id,
                agent_name=None,
                output_path=tmp_path / "trace.jsonl",
            )
        )


def test_session_trace_exporter_wraps_invalid_snapshot_as_export_error(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text('{"schema_version": 2}', encoding="utf-8")

    exporter = SessionTraceExporter(session_manager=manager)

    with pytest.raises(SessionExportReadError, match="Failed to load session snapshot"):
        exporter.export(
            ExportRequest(
                target=session_dir,
                agent_name=None,
                output_path=tmp_path / "trace.jsonl",
            )
        )


def test_session_trace_exporter_wraps_invalid_history_as_export_error(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "history_dev.json").write_text("{", encoding="utf-8")
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    exporter = SessionTraceExporter(session_manager=manager)

    with pytest.raises(SessionExportReadError, match="Failed to load session history"):
        exporter.export(
            ExportRequest(
                target=session_dir,
                agent_name="dev",
                output_path=tmp_path / "trace.jsonl",
            )
        )


def test_session_trace_exporter_wraps_output_write_errors(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    _write_history(session_dir / "history_dev.json", assistant_text="done")
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )
    output_dir = tmp_path / "trace-dir"
    output_dir.mkdir()

    exporter = SessionTraceExporter(session_manager=manager)

    with pytest.raises(SessionExportWriteError, match="Failed to write trace export"):
        exporter.export(
            ExportRequest(
                target=session_dir,
                agent_name="dev",
                output_path=output_dir,
            )
        )


def test_session_trace_exporter_uploads_trace_to_hugging_face_dataset(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    _write_history(session_dir / "history_dev.json", assistant_text="done")
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    class _Uploader:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Path, str | None]] = []

        def upload(
            self,
            *,
            dataset_repo: str,
            trace_path: Path,
            dataset_path: str | None = None,
        ) -> DatasetUploadResult:
            self.calls.append((dataset_repo, trace_path, dataset_path))
            return DatasetUploadResult(
                repo_id=dataset_repo,
                path_in_repo=dataset_path or trace_path.name,
                commit_url="https://huggingface.co/datasets/owner/traces/commit/main",
                file_url=f"https://huggingface.co/datasets/{dataset_repo}/blob/main/{dataset_path or trace_path.name}",
            )

    uploader = _Uploader()
    exporter = SessionTraceExporter(
        session_manager=manager,
        dataset_uploader=uploader,
    )

    result = exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
            hf_dataset="owner/dataset",
            hf_dataset_path="exports/trace.jsonl",
        )
    )

    assert uploader.calls == [("owner/dataset", tmp_path / "trace.jsonl", "exports/trace.jsonl")]
    assert result.upload is not None
    assert result.upload.repo_id == "owner/dataset"
    assert result.upload.path_in_repo == "exports/trace.jsonl"
