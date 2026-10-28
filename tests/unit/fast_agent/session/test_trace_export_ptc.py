"""PTC audit metadata survives persisted history without exporting filtered results."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent, Tool

from fast_agent.config import ProgrammaticToolCallingSettings
from fast_agent.constants import FAST_AGENT_PTC_CALLS
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import load_messages, save_json
from fast_agent.privacy.sanitizer import PrivacyFilterModelInfo, RedactionSpan, SanitizedText
from fast_agent.ptc import ProgrammaticToolRuntime
from fast_agent.session.trace_export_atif import (
    AtifRunSource,
    _sanitize_trajectory,
    build_atif_trajectory,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.session.atif_models import AtifTrajectory


def _export(code: str, result: CallToolResult, path: Path) -> AtifTrajectory:
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[],
            tool_calls={
                "parent": CallToolRequest(
                    params=CallToolRequestParams(name="execute_python", arguments={"code": code})
                )
            },
        ),
        PromptMessageExtended(role="user", content=[], tool_results={"parent": result}),
    ]
    save_json(messages, str(path))
    return build_atif_trajectory(
        AtifRunSource(
            session_id="ptc-audit",
            agent_name="test",
            model_name="passthrough",
            provider="fast-agent",
            history=load_messages(str(path)),
            message_timestamps=(None, None),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_atif_ptc_captures_dynamic_arguments_not_filtered_outputs(
    tmp_path: Path, empty: bool
) -> None:
    pytest.importorskip("pydantic_monty")
    dispatched: list[dict[str, Any]] = []

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        dispatched.append({"id": call_id, "tool": name, "arguments": deepcopy(arguments)})
        if name == "list_files":
            return CallToolResult(structured_content={"names": ["alpha", "beta"]}, content=[])
        return CallToolResult(
            content=[TextContent(type="text", text="RAW_FILTERED_FILE_CONTENT")],
            is_error=arguments["path"].endswith("beta.txt"),
        )

    code = (
        "pass"
        if empty
        else """files = await list_files()
results = await asyncio.gather(*[
    settle(read_file(path="/private/" + name + ".txt"))
    for name in files["names"]
])
print(len(results))
"""
    )
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    runtime.model_tools(
        [
            Tool(name="list_files", input_schema={"type": "object", "properties": {}}),
            Tool(
                name="read_file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
        ]
    )
    try:
        result = await runtime.execute({"code": code}, dispatch)
    finally:
        await runtime.close()
    assert not result.is_error
    trajectory = _export(code, result, tmp_path / "history.json")
    [step] = trajectory.steps
    assert step.tool_calls is not None
    assert step.tool_calls[0].arguments == {"code": code}
    assert step.observation is not None
    [observation] = step.observation.results
    assert observation.extra is not None
    assert observation.extra["ptc_calls_schema"] == "fast-agent.ptc-calls/v1"
    records = observation.extra["ptc_calls"]
    assert [
        {key: record[key] for key in ("id", "tool", "arguments")} for record in records
    ] == dispatched
    if empty:
        assert records == []
    else:
        assert [record["arguments"] for record in records] == [
            {},
            {"path": "/private/alpha.txt"},
            {"path": "/private/beta.txt"},
        ]
        assert [record["is_error"] for record in records] == [False, False, True]
        assert all(record["status"] == "completed" for record in records)
    # Neither metadata nor observation content dumps the nested results.
    assert "RAW_FILTERED_FILE_CONTENT" not in json.dumps(trajectory.to_json_dict())
    visible = json.dumps([block.model_dump(mode="json") for block in result.content])
    assert "/private/alpha.txt" not in visible
    assert "ptc_calls" not in visible


@pytest.mark.parametrize("calls", [None, [], [{"id": "old", "tool": "read", "is_error": False}]])
def test_atif_ptc_legacy_and_missing_metadata(
    tmp_path: Path, calls: list[dict[str, object]] | None
) -> None:
    result = CallToolResult(content=[])
    if calls is not None:
        result.meta = {FAST_AGENT_PTC_CALLS: calls}
    trajectory = _export("pass", result, tmp_path / "history.json")
    observation = trajectory.steps[0].observation
    assert observation is not None
    extra = observation.results[0].extra
    assert extra is not None
    if calls is None:
        assert "ptc_calls" not in extra
        assert "ptc_calls_schema" not in extra
    else:
        assert extra["ptc_calls"] == calls
        assert extra["ptc_calls_schema"] == "fast-agent.ptc-calls/v1"


class _SecretSanitizer:
    model_info = PrivacyFilterModelInfo(backend="test")

    def sanitize_text(self, text: str) -> SanitizedText:
        index = text.find("secret")
        return SanitizedText(
            text=text.replace("secret", "[REDACTED]"),
            spans=() if index < 0 else (RedactionSpan(label="secret", start=index, end=index + 6),),
        )


def test_ptc_arguments_use_the_same_privacy_filter_as_ordinary_arguments(tmp_path: Path) -> None:
    calls = [
        {
            "id": "secret-id",
            "tool": "secret-tool",
            "arguments": {"payload": [{"path": "/secret/file"}]},
            "is_error": False,
            "status": "completed",
        }
    ]
    result = CallToolResult(content=[], meta={FAST_AGENT_PTC_CALLS: calls})
    trajectory = _export("pass", result, tmp_path / "history.json")

    summary = _sanitize_trajectory(trajectory, _SecretSanitizer())

    assert summary is not None and summary.total == 1
    observation = trajectory.steps[0].observation
    assert observation is not None
    extra = observation.results[0].extra
    assert extra is not None
    [record] = extra["ptc_calls"]
    assert record["arguments"] == {"payload": [{"path": "/[REDACTED]/file"}]}
    # Provenance and the persisted source history are not rewritten by filtering.
    assert record["id"] == "secret-id"
    assert record["tool"] == "secret-tool"
    assert "/secret/file" in (tmp_path / "history.json").read_text()
