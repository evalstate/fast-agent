from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.tools.local_filesystem_runtime import LocalFilesystemRuntime

if TYPE_CHECKING:
    from pathlib import Path


def _build_agent(tmp_path: Path) -> McpAgent:
    agent = McpAgent(
        config=AgentConfig(name="test", instruction="Instruction", servers=[]),
        context=Context(),
    )
    agent.set_filesystem_runtime(
        LocalFilesystemRuntime(
            logging.getLogger("edit-file-integration"),
            working_directory=tmp_path,
            enable_write=False,
            enable_apply_patch=False,
            enable_edit_file=True,
        )
    )
    return agent


def _result_payload(result: CallToolResult) -> dict[str, object]:
    payload = result.structuredContent
    assert isinstance(payload, dict)
    return payload


def _result_text_payload(result: CallToolResult) -> dict[str, object]:
    content = result.content
    assert content is not None
    assert isinstance(content[0], TextContent)
    return json.loads(content[0].text)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_is_exposed_as_internal_runtime_tool(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    try:
        tool_names = {tool.name for tool in (await agent.list_tools()).tools}
        assert "edit_file" in tool_names
        assert "write_text_file" not in tool_names
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_replaces_unique_multiline_match_and_returns_structured_diff(
    tmp_path: Path,
) -> None:
    project_file = tmp_path / "src" / "utils.py"
    project_file.parent.mkdir(parents=True, exist_ok=True)
    project_file.write_text(
        'def hello():\n    print("world")\n',
        encoding="utf-8",
        newline="",
    )

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "src/utils.py",
                "old_string": 'def hello():\n    print("world")',
                "new_string": 'def hello():\n    print("hello")',
            },
        )
        payload = _result_payload(result)

        assert result.isError is False
        assert project_file.read_text(encoding="utf-8", newline="") == (
            'def hello():\n    print("hello")\n'
        )
        assert payload["success"] is True
        assert payload["path"] == "src/utils.py"
        assert payload["line_start"] == 1
        assert payload["line_end"] == 2
        assert payload["replacements"] == 1
        assert isinstance(payload["diff"], str)
        assert '-    print("world")' in payload["diff"]
        assert '+    print("hello")' in payload["diff"]
        assert _result_text_payload(result) == payload
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_replace_all_updates_each_non_overlapping_match(tmp_path: Path) -> None:
    target_file = tmp_path / "notes.txt"
    target_file.write_text("value one\nvalue two\n", encoding="utf-8", newline="")

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "notes.txt",
                "old_string": "value",
                "new_string": "item",
                "replace_all": True,
            },
        )
        payload = _result_payload(result)

        assert result.isError is False
        assert target_file.read_text(encoding="utf-8", newline="") == "item one\nitem two\n"
        assert payload["success"] is True
        assert payload["replacements"] == 2
        assert payload["line_start"] == 1
        assert payload["line_end"] == 2
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_reports_multiple_matches_with_line_locations(tmp_path: Path) -> None:
    target_file = tmp_path / "ambiguous.txt"
    target_file.write_text("alpha\nvalue\nbeta\nvalue\n", encoding="utf-8", newline="")

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "ambiguous.txt",
                "old_string": "value",
                "new_string": "item",
            },
        )
        payload = _result_payload(result)

        assert result.isError is True
        assert target_file.read_text(encoding="utf-8", newline="") == "alpha\nvalue\nbeta\nvalue\n"
        assert payload == {
            "success": False,
            "error": "multiple_matches",
            "message": (
                "Found 2 matches for old_string in ambiguous.txt. "
                "Use replace_all=True to replace all, or provide more surrounding context "
                "to uniquely identify the target."
            ),
            "path": "ambiguous.txt",
            "matches": [
                {"line_start": 2, "line_end": 2},
                {"line_start": 4, "line_end": 4},
            ],
        }
        assert _result_text_payload(result) == payload
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_reports_no_match_without_modifying_file(tmp_path: Path) -> None:
    target_file = tmp_path / "notes.txt"
    target_file.write_text("alpha\nbeta\n", encoding="utf-8", newline="")

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "notes.txt",
                "old_string": "missing",
                "new_string": "replacement",
            },
        )
        payload = _result_payload(result)

        assert result.isError is True
        assert target_file.read_text(encoding="utf-8", newline="") == "alpha\nbeta\n"
        assert payload == {
            "success": False,
            "error": "no_match",
            "message": (
                "old_string was not found in the file. Re-read the file and use the exact "
                "current text."
            ),
            "path": "notes.txt",
        }
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_reports_missing_file_and_directory_errors(tmp_path: Path) -> None:
    directory = tmp_path / "nested"
    directory.mkdir()

    agent = _build_agent(tmp_path)
    try:
        missing_result = await agent.call_tool(
            "edit_file",
            {
                "path": "missing.txt",
                "old_string": "alpha",
                "new_string": "beta",
            },
        )
        directory_result = await agent.call_tool(
            "edit_file",
            {
                "path": "nested",
                "old_string": "alpha",
                "new_string": "beta",
            },
        )

        assert _result_payload(missing_result) == {
            "success": False,
            "error": "file_not_found",
            "message": (
                "File not found: missing.txt. Check the path, or use write_text_file to create it."
            ),
            "path": "missing.txt",
        }
        assert _result_payload(directory_result) == {
            "success": False,
            "error": "is_directory",
            "message": (
                "Path is a directory, not a file: nested. Use the correct file path."
            ),
            "path": "nested",
        }
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_deletion_preserves_missing_trailing_newline(tmp_path: Path) -> None:
    target_file = tmp_path / "delete.txt"
    target_file.write_text("alpha\n# TODO remove\nomega", encoding="utf-8", newline="")

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "delete.txt",
                "old_string": "# TODO remove\n",
                "new_string": "",
            },
        )
        payload = _result_payload(result)

        assert result.isError is False
        assert target_file.read_text(encoding="utf-8", newline="") == "alpha\nomega"
        assert payload["success"] is True
        assert payload["line_start"] == 2
        assert payload["line_end"] == 2
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_can_replace_entire_file_and_preserve_trailing_newline(tmp_path: Path) -> None:
    target_file = tmp_path / "whole.txt"
    target_file.write_text("alpha\nbeta\n", encoding="utf-8", newline="")

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "whole.txt",
                "old_string": "alpha\nbeta\n",
                "new_string": "gamma\ndelta\n",
            },
        )
        payload = _result_payload(result)

        assert result.isError is False
        assert target_file.read_text(encoding="utf-8", newline="") == "gamma\ndelta\n"
        assert payload["success"] is True
        assert payload["line_start"] == 1
        assert payload["line_end"] == 2
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_can_delete_entire_file_to_empty_contents(tmp_path: Path) -> None:
    target_file = tmp_path / "empty_after_delete.txt"
    target_file.write_text("remove me", encoding="utf-8", newline="")

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "empty_after_delete.txt",
                "old_string": "remove me",
                "new_string": "",
            },
        )
        payload = _result_payload(result)

        assert result.isError is False
        assert target_file.read_text(encoding="utf-8", newline="") == ""
        assert payload["success"] is True
        assert payload["line_start"] == 1
        assert payload["line_end"] == 1
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_reports_multiline_ambiguity_with_match_locations(tmp_path: Path) -> None:
    target_file = tmp_path / "multi.txt"
    target_file.write_text(
        "header\nfoo\nbar\nmiddle\nfoo\nbar\nfooter\n",
        encoding="utf-8",
        newline="",
    )

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "multi.txt",
                "old_string": "foo\nbar",
                "new_string": "baz\nqux",
            },
        )
        payload = _result_payload(result)

        assert result.isError is True
        assert payload == {
            "success": False,
            "error": "multiple_matches",
            "message": (
                "Found 2 matches for old_string in multi.txt. "
                "Use replace_all=True to replace all, or provide more surrounding context "
                "to uniquely identify the target."
            ),
            "path": "multi.txt",
            "matches": [
                {"line_start": 2, "line_end": 3},
                {"line_start": 5, "line_end": 6},
            ],
        }
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_is_whitespace_exact_and_supports_unicode_success(tmp_path: Path) -> None:
    spaced_file = tmp_path / "indent.py"
    spaced_file.write_text("def run():\n\treturn 1\n", encoding="utf-8", newline="")
    unicode_file = tmp_path / "unicode.txt"
    unicode_file.write_text("café\n", encoding="utf-8", newline="")

    agent = _build_agent(tmp_path)
    try:
        whitespace_result = await agent.call_tool(
            "edit_file",
            {
                "path": "indent.py",
                "old_string": "    return 1",
                "new_string": "    return 2",
            },
        )
        unicode_result = await agent.call_tool(
            "edit_file",
            {
                "path": "unicode.txt",
                "old_string": "café",
                "new_string": "naïve",
            },
        )

        assert whitespace_result.isError is True
        assert _result_payload(whitespace_result) == {
            "success": False,
            "error": "no_match",
            "message": (
                "old_string was not found in the file. Re-read the file and use the exact "
                "current text."
            ),
            "path": "indent.py",
        }
        assert unicode_result.isError is False
        assert unicode_file.read_text(encoding="utf-8", newline="") == "naïve\n"
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_preserves_crlf_when_matching_exact_windows_newlines(tmp_path: Path) -> None:
    target_file = tmp_path / "crlf.txt"
    target_file.write_bytes(b"alpha\r\nbeta\r\n")

    agent = _build_agent(tmp_path)
    try:
        result = await agent.call_tool(
            "edit_file",
            {
                "path": "crlf.txt",
                "old_string": "alpha\r\nbeta\r\n",
                "new_string": "gamma\r\ndelta\r\n",
            },
        )
        payload = _result_payload(result)

        assert result.isError is False
        assert target_file.read_bytes() == b"gamma\r\ndelta\r\n"
        assert payload["success"] is True
        assert payload["line_start"] == 1
        assert payload["line_end"] == 2
        assert isinstance(payload["diff"], str)
        assert "\r\n" in payload["diff"]
    finally:
        await agent._aggregator.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_file_reports_empty_old_string_no_op_and_encoding_errors(
    tmp_path: Path,
) -> None:
    empty_target = tmp_path / "empty.txt"
    empty_target.write_text("alpha\n", encoding="utf-8", newline="")
    encoded_target = tmp_path / "latin1.txt"
    encoded_target.write_bytes(b"\xff\xfealpha\n")

    agent = _build_agent(tmp_path)
    try:
        empty_result = await agent.call_tool(
            "edit_file",
            {
                "path": "empty.txt",
                "old_string": "",
                "new_string": "beta",
            },
        )
        no_op_result = await agent.call_tool(
            "edit_file",
            {
                "path": "empty.txt",
                "old_string": "alpha",
                "new_string": "alpha",
            },
        )
        encoding_result = await agent.call_tool(
            "edit_file",
            {
                "path": "latin1.txt",
                "old_string": "alpha",
                "new_string": "beta",
            },
        )

        assert _result_payload(empty_result) == {
            "success": False,
            "error": "empty_old_string",
            "message": "Empty search string is not allowed.",
            "path": "empty.txt",
        }
        assert _result_payload(no_op_result) == {
            "success": False,
            "error": "no_op",
            "message": "old_string and new_string are identical. No change is needed.",
            "path": "empty.txt",
        }
        assert _result_payload(encoding_result) == {
            "success": False,
            "error": "encoding_error",
            "message": (
                "File could not be decoded as UTF-8: latin1.txt. Check file encoding."
            ),
            "path": "latin1.txt",
        }
    finally:
        await agent._aggregator.close()
