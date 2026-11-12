from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pytest
from mcp_types import TextContent

from fast_agent.config import Settings, ShellSettings
from fast_agent.tools.shell_process import process_result_metadata
from fast_agent.tools.shell_runtime import ShellRuntime

if TYPE_CHECKING:
    from pathlib import Path


def _runtime(tmp_path: Path, *, durable: bool) -> ShellRuntime:
    root = None
    if durable:
        root = tmp_path / "processes"
        root.mkdir(mode=0o700)
    return ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("background-settle-test"),
        durable_process_root=root,
        config=Settings(
            shell_execution=ShellSettings(tool_profile="minimal_process", show_bash=False)
        ),
    )


def _text(result) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


@pytest.mark.asyncio
async def test_durable_background_service_that_exits_immediately_is_reported_as_failed(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, durable=True)
    result = await runtime.call_tool(
        "bash", {"command": "echo port-in-use >&2; exit 3", "run_in_background": True}
    )
    text = _text(result)
    metadata = process_result_metadata(result)
    assert metadata is not None
    assert result.is_error is True
    assert metadata["process_status"] == "failed"
    assert metadata.get("exit_code") == 3
    assert "port-in-use" in text
    assert "still running" not in text

    # The failure tail is a peek: a later poll still reports the unread output.
    status = await runtime.call_tool(
        "process", {"process_id": str(metadata["process_id"]), "action": "status"}
    )
    assert "port-in-use" in _text(status)


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_healthy_background_service_is_reported_running(
    tmp_path: Path, durable: bool
) -> None:
    runtime = _runtime(tmp_path, durable=durable)
    result = await runtime.call_tool("bash", {"command": "sleep 30", "run_in_background": True})
    metadata = process_result_metadata(result)
    assert metadata is not None
    assert metadata["process_status"] == "running"
    assert metadata["process_yield_reason"] == "background"
    assert not result.is_error
    match = re.search(r"process-[0-9a-z]+", _text(result))
    assert match is not None
    await runtime.call_tool("process", {"process_id": match.group(0), "action": "stop"})
