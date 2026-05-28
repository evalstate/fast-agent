from __future__ import annotations

import logging

from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.tools.local_filesystem_runtime import LocalFilesystemRuntime
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.tools.skill_reader import SkillReader
from fast_agent.tools.tool_sources import tool_source


def test_shell_runtime_stamps_execute_as_shell() -> None:
    runtime = ShellRuntime("for test", logging.getLogger(__name__))

    assert runtime.tool is not None
    assert tool_source(runtime.tool) == "shell"


def test_acp_terminal_runtime_stamps_execute_as_acp_terminal() -> None:
    runtime = ACPTerminalRuntime(
        connection=object(),
        session_id="session",
        activation_reason="for test",
    )

    assert tool_source(runtime.tool) == "acp_terminal"


def test_local_filesystem_runtime_stamps_enabled_tools_as_shell() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger(__name__),
        enable_apply_patch=True,
        enable_edit_file=True,
        enable_attach_media="on",
    )

    assert {tool.name: tool_source(tool) for tool in runtime.tools} == {
        "read_text_file": "shell",
        "write_text_file": "shell",
        "apply_patch": "shell",
        "edit_file": "shell",
        "attach_media": "shell",
    }


def test_acp_filesystem_runtime_stamps_enabled_tools_as_acp_filesystem() -> None:
    runtime = ACPFilesystemRuntime(
        connection=object(),
        session_id="session",
        activation_reason="for test",
    )

    assert {tool.name: tool_source(tool) for tool in runtime.tools} == {
        "read_text_file": "acp_filesystem",
        "write_text_file": "acp_filesystem",
    }


def test_skill_reader_stamps_read_skill_as_skill() -> None:
    runtime = SkillReader([], logging.getLogger(__name__))

    assert tool_source(runtime.tool) == "skill"
