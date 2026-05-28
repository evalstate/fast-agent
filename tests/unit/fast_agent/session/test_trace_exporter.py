from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

from fast_agent.constants import (
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    FAST_AGENT_TIMING,
    FAST_AGENT_USAGE,
    OPENAI_ASSISTANT_MESSAGE_ITEMS,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import save_json
from fast_agent.privacy.sanitizer import PrivacyFilterModelInfo, RedactionSpan, SanitizedText
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
    SessionExportPrivacyFilterError,
    SessionExportReadError,
    SessionExportWriteError,
)
from fast_agent.session.trace_export_models import DatasetUploadResult, ExportRequest
from fast_agent.types import COMMENTARY_PHASE, FINAL_ANSWER_PHASE, LlmStopReason


def _write_session_snapshot(
    session_dir: Path,
    *,
    session_id: str,
    agents: dict[str, SessionAgentSnapshot],
    active_agent: str | None,
    cwd: str | None = None,
) -> None:
    snapshot = SessionSnapshot(
        session_id=session_id,
        created_at=datetime(2026, 4, 20, 13, 3, 0),
        last_activity=datetime(2026, 4, 20, 13, 8, 0),
        continuation=SessionContinuationSnapshot(
            active_agent=active_agent,
            cwd=cwd,
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


def _write_history_with_timestamps(
    path: Path,
    *,
    messages: list[PromptMessageExtended],
    timestamps: list[datetime | None],
) -> None:
    payload_messages: list[dict[str, object]] = []
    for message, timestamp in zip(messages, timestamps, strict=True):
        payload = message.model_dump(mode="json", exclude_none=True)
        if timestamp is not None:
            payload["timestamp"] = timestamp.astimezone(timezone.utc).isoformat()
        payload_messages.append(payload)
    path.write_text(
        json.dumps({"messages": payload_messages}, indent=2),
        encoding="utf-8",
    )


def _build_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )


class _FakePrivacySanitizer:
    @property
    def model_info(self) -> PrivacyFilterModelInfo:
        return PrivacyFilterModelInfo(backend="fake", repo_id="test/privacy")

    def sanitize_text(self, text: str) -> SanitizedText:
        spans: list[RedactionSpan] = []
        start = 0
        while True:
            index = text.find("Alice", start)
            if index < 0:
                break
            spans.append(RedactionSpan(label="private_person", start=index, end=index + 5))
            start = index + 5
        return SanitizedText(
            text=text.replace("Alice", "<PRIVATE_PERSON>"),
            spans=tuple(spans),
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

    progress: list[str] = []
    exporter = SessionTraceExporter(session_manager=manager, progress_callback=progress.append)
    result = exporter.export(
        ExportRequest(
            target=session_dir / "session.json",
            agent_name=None,
            output_path=tmp_path / "trace.jsonl",
        )
    )

    assert progress == [
        "Export: preparing codex trace for agent 'dev' from 2 message(s): "
        "1 user, 1 assistant, 0 tool call(s), 0 tool result(s)."
    ]
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert result.record_count == 9
    assert records[0]["type"] == "session_meta"
    assert records[0]["timestamp"] == "2026-04-20T13:03:00.000Z"
    assert records[0]["payload"]["id"] == session_id
    assert records[0]["payload"]["model_spec"] == "gpt-5.4"
    assert records[0]["payload"]["base_instructions"]["text"] == "You are dev."
    assert "harness" not in records[0]
    assert records[1]["type"] == "response_item"
    assert "timestamp" not in records[1]
    assert records[1]["payload"] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "You are dev."}],
    }
    assert records[2]["type"] == "event_msg"
    assert "timestamp" not in records[2]
    assert records[2]["payload"]["type"] == "task_started"
    assert "started_at" not in records[2]["payload"]
    assert records[3]["type"] == "event_msg"
    assert "timestamp" not in records[3]
    assert records[3]["payload"]["type"] == "user_message"
    assert records[3]["payload"]["message"] == "hello"
    assert records[4]["type"] == "turn_context"
    assert "timestamp" not in records[4]
    assert records[4]["payload"]["turn_id"] == "turn-1"
    assert "current_date" not in records[4]["payload"]
    assert records[4]["payload"]["model_spec"] == "gpt-5.4"
    assert "developer_instructions" not in records[4]["payload"]
    assert "approval_policy" not in records[4]["payload"]
    assert "sandbox_policy" not in records[4]["payload"]
    assert records[5]["type"] == "response_item"
    assert "timestamp" not in records[5]
    assert records[5]["payload"]["type"] == "message"
    assert records[5]["payload"]["role"] == "user"
    assert records[5]["payload"]["content"] == [{"type": "input_text", "text": "hello"}]
    assert records[6]["type"] == "response_item"
    assert "timestamp" not in records[6]
    assert records[6]["payload"] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "thinking"}],
    }
    assert "content" not in records[6]["payload"]
    assert "encrypted_content" not in records[6]["payload"]
    assert records[7]["type"] == "response_item"
    assert "timestamp" not in records[7]
    assert records[7]["payload"]["type"] == "message"
    assert records[7]["payload"]["role"] == "assistant"
    assert records[7]["payload"]["content"] == [{"type": "output_text", "text": "done"}]
    assert records[7]["payload"]["end_turn"] is True
    assert "phase" not in records[7]["payload"]
    assert records[8]["type"] == "event_msg"
    assert "timestamp" not in records[8]
    assert records[8]["payload"]["type"] == "task_complete"
    assert records[8]["payload"]["last_agent_message"] == "done"


def test_session_trace_exporter_context_window_falls_back_to_model(
    tmp_path: Path,
) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-overlay"
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
                model="gpt-5.4",
                model_spec="custom-overlay",
                provider="codexresponses",
            )
        },
    )

    exporter = SessionTraceExporter(session_manager=manager)
    exporter.export(
        ExportRequest(
            target=session_dir / "session.json",
            agent_name=None,
            output_path=tmp_path / "trace.jsonl",
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[0]["payload"]["model_spec"] == "custom-overlay"
    assert records[1]["payload"]["type"] == "task_started"
    assert records[1]["payload"]["model_context_window"] == 272000
    assert records[3]["payload"]["model"] == "gpt-5.4"
    assert records[3]["payload"]["model_spec"] == "custom-overlay"


def test_session_trace_exporter_preserves_assistant_commentary_phase(tmp_path: Path) -> None:
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
            content=[TextContent(type="text", text="Planning next action.")],
            phase=COMMENTARY_PHASE,
            stop_reason=LlmStopReason.END_TURN,
        ),
    ]
    save_json(messages, str(session_dir / "history_dev.json"))
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={
            "dev": SessionAgentSnapshot(
                history_file="history_dev.json",
                request_settings=SessionRequestSettingsSnapshot(service_tier="flex"),
            )
        },
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
    commentary_messages = [
        record["payload"]
        for record in records
        if record["type"] == "response_item"
        and record["payload"].get("type") == "message"
        and record["payload"].get("role") == "assistant"
    ]

    assert commentary_messages == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Planning next action."}],
            "end_turn": True,
            "phase": "commentary",
        }
    ]


def test_session_trace_exporter_preserves_raw_openai_assistant_message_items(
    tmp_path: Path,
) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-raw-items"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Final answer.")],
            channels={
                OPENAI_ASSISTANT_MESSAGE_ITEMS: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "type": "message",
                                "role": "assistant",
                                "phase": "commentary",
                                "content": [
                                    {"type": "output_text", "text": "Checking the repo."}
                                ],
                            }
                        ),
                    ),
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "type": "message",
                                "role": "assistant",
                                "phase": "final_answer",
                                "content": [{"type": "output_text", "text": "Final answer."}],
                            }
                        ),
                    ),
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
    assistant_messages = [
        record["payload"]
        for record in records
        if record["type"] == "response_item"
        and record["payload"].get("type") == "message"
        and record["payload"].get("role") == "assistant"
    ]

    assert assistant_messages == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Checking the repo."}],
            "phase": "commentary",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Final answer."}],
            "phase": "final_answer",
        },
    ]


def test_session_trace_exporter_exports_server_web_search_as_codex_call(
    tmp_path: Path,
) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-web-search"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="look this up")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="I found a result.")],
            channels={
                ANTHROPIC_SERVER_TOOLS_CHANNEL: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "type": "server_tool_use",
                                "id": "ws_1",
                                "name": "web_search",
                                "status": "completed",
                                "input": {"query": "fast-agent trace viewer"},
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
    web_search_calls = [
        record["payload"]
        for record in records
        if record["type"] == "response_item"
        and record["payload"].get("type") == "web_search_call"
    ]

    assert web_search_calls == [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "fast-agent trace viewer"},
        }
    ]


def test_session_trace_exporter_uses_workspace_dir_for_relative_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = _build_manager(workspace)
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
                request_settings=SessionRequestSettingsSnapshot(service_tier="flex"),
            )
        },
    )
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    exporter = SessionTraceExporter(session_manager=manager)
    result = exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=Path("trace.jsonl"),
        )
    )

    assert result.output_path == workspace / "trace.jsonl"
    assert result.output_path.is_file()
    assert not (other_cwd / "trace.jsonl").exists()


def test_session_trace_exporter_uses_workspace_dir_for_default_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = _build_manager(workspace)
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
                request_settings=SessionRequestSettingsSnapshot(service_tier="flex"),
            )
        },
    )
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    exporter = SessionTraceExporter(session_manager=manager)
    result = exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=None,
        )
    )

    expected_path = workspace / f"{session_id}__dev__codex.jsonl"
    assert result.output_path == expected_path
    assert result.output_path.is_file()
    assert not (other_cwd / expected_path.name).exists()


def test_session_trace_exporter_uses_message_timestamps_for_turn_date(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    turn_started_at = datetime(2026, 4, 22, 9, 15, 0, tzinfo=timezone.utc)
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            stop_reason=LlmStopReason.END_TURN,
        ),
    ]
    _write_history_with_timestamps(
        session_dir / "history_dev.json",
        messages=messages,
        timestamps=[turn_started_at, turn_started_at + timedelta(seconds=5)],
    )
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

    assert records[0]["payload"]["timestamp"] == "2026-04-20T13:03:00.000Z"
    assert records[1]["payload"]["type"] == "task_started"
    assert records[1]["payload"]["started_at"] == int(turn_started_at.timestamp())
    assert records[1]["timestamp"] == "2026-04-22T09:15:00.000Z"
    assert records[2]["payload"]["type"] == "user_message"
    assert records[2]["timestamp"] == "2026-04-22T09:15:00.000Z"
    assert records[3]["type"] == "turn_context"
    assert "timestamp" not in records[3]
    assert records[3]["payload"]["current_date"] == "2026-04-22"
    assert records[4]["payload"]["role"] == "user"
    assert records[4]["timestamp"] == "2026-04-22T09:15:00.000Z"
    assert records[5]["payload"]["role"] == "assistant"
    assert records[5]["timestamp"] == "2026-04-22T09:15:05.000Z"
    assert records[6]["payload"]["type"] == "task_complete"
    assert records[6]["payload"]["completed_at"] == int(
        (turn_started_at + timedelta(seconds=5)).timestamp()
    )
    assert records[6]["payload"]["duration_ms"] == 5000
    assert records[6]["timestamp"] == "2026-04-22T09:15:05.000Z"


def test_session_trace_exporter_adds_turn_timing_from_fast_agent_channels(
    tmp_path: Path,
) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-timing"
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
                FAST_AGENT_TIMING: [
                    TextContent(
                        type="text",
                        text='{"duration_ms": 1234.56, "ttft_ms": 321.4}',
                    )
                ]
            },
            stop_reason=LlmStopReason.END_TURN,
        ),
    ]
    _write_history_with_timestamps(
        session_dir / "history_dev.json",
        messages=messages,
        timestamps=[None, None],
    )
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

    turn_complete = records[-1]["payload"]
    assert turn_complete["type"] == "task_complete"
    assert turn_complete["duration_ms"] == 1235
    assert turn_complete["time_to_first_token_ms"] == 321


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
        agents={
            "dev": SessionAgentSnapshot(
                history_file="history_dev.json",
                request_settings=SessionRequestSettingsSnapshot(service_tier="flex"),
            )
        },
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
    assert records[1]["payload"]["type"] == "task_started"
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
        "status": "success",
    }
    assert records[6]["type"] == "event_msg"
    assert records[6]["payload"]["type"] == "task_complete"
    assert records[6]["payload"]["last_agent_message"] == "Using tools"


def test_session_trace_exporter_applies_privacy_sanitizer_to_codex_text(
    tmp_path: Path,
) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello Alice")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Using tools for Alice")],
            tool_calls={
                "call_Alice": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="lookup_Alice",
                        arguments={"query": "Alice"},
                    ),
                )
            },
            channels={
                "reasoning": [TextContent(type="text", text="Thinking about Alice")],
            },
            stop_reason=LlmStopReason.TOOL_USE,
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "call_Alice": CallToolResult(
                    content=[TextContent(type="text", text="Alice result")],
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
        cwd="/home/Alice/work",
        agents={
            "dev": SessionAgentSnapshot(
                history_file="history_dev.json",
                resolved_prompt="Help Alice safely.",
            )
        },
    )

    exporter = SessionTraceExporter(
        session_manager=manager,
        privacy_sanitizer=_FakePrivacySanitizer(),
    )
    result = exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=None,
            privacy_filter=True,
        )
    )

    assert result.output_path == tmp_path / f"{session_id}__dev__codex-privacy.jsonl"
    assert result.redaction is not None
    assert result.redaction.total == 9
    assert result.redaction.by_label == {"private_person": 9}

    records = [
        json.loads(line)
        for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]
    session_meta = records[0]["payload"]
    assert session_meta["cwd"] == "/home/Alice/work"
    assert session_meta["base_instructions"]["text"] == "Help <PRIVATE_PERSON> safely."
    assert session_meta["privacy_filter"]["applied"] is True
    assert session_meta["privacy_filter"]["backend"] == "fake"
    redactions = session_meta["privacy_filter"]["redactions"]
    assert redactions["total"] == 9
    assert redactions["by_label"] == {"private_person": 9}
    assert redactions["elapsed_seconds"] >= 0

    payloads = [record["payload"] for record in records]
    developer = next(
        payload for payload in payloads if payload.get("role") == "developer"
    )
    assert developer["content"][0]["text"] == "Help <PRIVATE_PERSON> safely."
    user_event = next(
        payload for payload in payloads if payload.get("type") == "user_message"
    )
    assert user_event["message"] == "hello <PRIVATE_PERSON>"
    function_call = next(
        payload for payload in payloads if payload.get("type") == "function_call"
    )
    assert function_call["name"] == "lookup_Alice"
    assert function_call["call_id"] == "call_Alice"
    assert function_call["arguments"] == '{"query":"<PRIVATE_PERSON>"}'
    tool_output = next(
        payload for payload in payloads if payload.get("type") == "function_call_output"
    )
    assert tool_output["output"] == "<PRIVATE_PERSON> result"
    turn_complete = next(
        payload for payload in reversed(payloads) if payload.get("type") == "task_complete"
    )
    assert turn_complete["last_agent_message"] == "Using tools for <PRIVATE_PERSON>"


def test_session_trace_exporter_requires_privacy_sanitizer_for_privacy_filter(
    tmp_path: Path,
) -> None:
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

    exporter = SessionTraceExporter(session_manager=manager)
    with pytest.raises(SessionExportPrivacyFilterError):
        exporter.export(
            ExportRequest(
                target=session_dir,
                agent_name="dev",
                output_path=tmp_path / "trace.jsonl",
                privacy_filter=True,
            )
        )
    assert not (tmp_path / "trace.jsonl").exists()


def test_session_trace_exporter_reports_privacy_filter_progress(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    _write_history(session_dir / "history_dev.json", assistant_text="done for Alice")
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={
            "dev": SessionAgentSnapshot(
                history_file="history_dev.json",
                resolved_prompt="Help Alice safely.",
            )
        },
    )
    progress: list[str] = []
    exporter = SessionTraceExporter(
        session_manager=manager,
        privacy_sanitizer=_FakePrivacySanitizer(),
        progress_callback=progress.append,
    )

    exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
            privacy_filter=True,
        )
    )

    assert any(message.startswith("Privacy filter: sanitizing ") for message in progress)
    assert any(message.startswith("Privacy filter: overall ") for message in progress)
    assert any("100%" in message for message in progress)


def test_session_trace_exporter_uploads_sanitized_trace_to_hf_dataset(
    tmp_path: Path,
) -> None:
    manager = _build_manager(tmp_path)
    session_id = "2604201303-x5MNlH"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    _write_history(session_dir / "history_dev.json", assistant_text="done for Alice")
    _write_session_snapshot(
        session_dir,
        session_id=session_id,
        active_agent="dev",
        agents={"dev": SessionAgentSnapshot(history_file="history_dev.json")},
    )

    class _Uploader:
        def __init__(self) -> None:
            self.uploaded_text: str | None = None

        def upload(
            self,
            *,
            dataset_repo: str,
            trace_path: Path,
            dataset_path: str | None = None,
        ) -> DatasetUploadResult:
            del dataset_path
            self.uploaded_text = trace_path.read_text(encoding="utf-8")
            return DatasetUploadResult(
                repo_id=dataset_repo,
                path_in_repo=trace_path.name,
                commit_url="https://huggingface.co/datasets/owner/traces/commit/main",
                file_url=f"https://huggingface.co/datasets/{dataset_repo}/blob/main/{trace_path.name}",
            )

    uploader = _Uploader()
    exporter = SessionTraceExporter(
        session_manager=manager,
        dataset_uploader=uploader,
        privacy_sanitizer=_FakePrivacySanitizer(),
    )

    result = exporter.export(
        ExportRequest(
            target=session_dir,
            agent_name="dev",
            output_path=tmp_path / "trace.jsonl",
            hf_dataset="owner/dataset",
            privacy_filter=True,
        )
    )

    assert result.upload is not None
    assert result.redaction is not None
    assert uploader.uploaded_text is not None
    assert "done for <PRIVATE_PERSON>" in uploader.uploaded_text
    assert "done for Alice" not in uploader.uploaded_text


def test_session_trace_exporter_marks_tool_errors_in_codex_output(tmp_path: Path) -> None:
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
                        arguments={"command": "false"},
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
                    content=[TextContent(type="text", text="process exit code was 1")],
                    isError=True,
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
        "output": "process exit code was 1",
        "status": "error",
    }


def test_session_trace_exporter_preserves_explicit_assistant_phase(tmp_path: Path) -> None:
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
            phase=FINAL_ANSWER_PHASE,
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

    assert records[5]["payload"]["type"] == "message"
    assert records[5]["payload"]["role"] == "assistant"
    assert records[5]["payload"]["phase"] == FINAL_ANSWER_PHASE


def test_session_trace_exporter_serializes_zero_argument_tool_calls_as_empty_object(
    tmp_path: Path,
) -> None:
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
                    params=CallToolRequestParams(name="tool_function"),
                )
            },
            stop_reason=LlmStopReason.TOOL_USE,
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
        "type": "function_call",
        "name": "tool_function",
        "arguments": "{}",
        "call_id": "call_1",
    }


def test_session_trace_exporter_uses_usage_metadata_for_model_and_token_count(
    tmp_path: Path,
) -> None:
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
        agents={
            "dev": SessionAgentSnapshot(
                history_file="history_dev.json",
                request_settings=SessionRequestSettingsSnapshot(service_tier="flex"),
            )
        },
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
    assert records[0]["payload"]["model_spec"] == "gpt-5.3-codex?service_tier=flex"
    assert records[1]["type"] == "event_msg"
    assert records[1]["payload"]["type"] == "task_started"
    assert records[1]["payload"]["model_context_window"] == 400000
    assert records[3]["type"] == "turn_context"
    assert records[3]["payload"]["model"] == "gpt-5.3-codex"
    assert records[3]["payload"]["model_spec"] == "gpt-5.3-codex?service_tier=flex"
    assert records[6]["type"] == "event_msg"
    assert records[6]["payload"] == {
        "type": "token_count",
        "info": {
            "total_token_usage": {
                "input_tokens": 120,
                "cached_input_tokens": 12,
                "output_tokens": 30,
                "reasoning_output_tokens": 7,
                "total_tokens": 150,
            },
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
        "status": "success",
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
        "status": "success",
    }


def test_session_trace_exporter_preserves_user_content_alongside_tool_outputs(
    tmp_path: Path,
) -> None:
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
        "status": "success",
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


def test_session_trace_exporter_requires_agent_when_multiple_histories_exist(
    tmp_path: Path,
) -> None:
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
