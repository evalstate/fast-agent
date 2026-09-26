from __future__ import annotations

import logging
import re

import pytest
from mcp_types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.tools.apply_patch_tool import (
    OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY,
    get_openai_responses_custom_tool_payload,
)
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.tools.shell_tool_definitions import (
    build_freeform_shell_tool,
    parse_freeform_shell_input,
)


def _parse(text: str):
    return parse_freeform_shell_input({"input": text}, tool_name="shell")


def test_plain_command_is_raw_text() -> None:
    command = "printf '%s\\n' '^(?:\"[^\"\\\\]*\")$' > p.txt && cat -A p.txt"
    parsed = _parse(command)
    assert parsed.command == command
    assert parsed.background is False
    assert parsed.hard_timeout_seconds is None
    assert parsed.cwd is None


def test_multiline_heredoc_is_preserved() -> None:
    command = "cat > s.py <<'PY'\nprint('a\\tb')\nPY\npython s.py"
    assert _parse(command).command == command


@pytest.mark.parametrize(
    ("pragma", "background", "timeout", "cwd"),
    [
        ('# @shell: {"background": true}', True, None, None),
        ('# @shell: {"timeout": 300}', False, 300, None),
        ('  # @shell: {"workdir": "/app", "timeout": 60}', False, 60, "/app"),
        ("# @shell: {}", False, None, None),
    ],
)
def test_pragma_options(
    pragma: str, background: bool, timeout: int | None, cwd: str | None
) -> None:
    parsed = _parse(f"{pragma}\npython -m http.server 8080")
    assert parsed.command == "python -m http.server 8080"
    assert parsed.background is background
    assert parsed.hard_timeout_seconds == timeout
    assert parsed.cwd == cwd


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("# @shell: {background: true}\nls", "must be a JSON object"),
        ('# @shell: ["x"]\nls', "must be a JSON object"),
        ('# @shell: {"bogus": 1}\nls', "unknown shell option"),
        ('# @shell: {"background": true}', "no command"),
        ("   \n  ", "no command"),
        ("sleep 100 &", "Shell-level backgrounding was not executed"),
        ('# @shell: {"background": true, "timeout": 5}\nsleep 9', "cannot be combined"),
    ],
)
def test_invalid_input_is_rejected_with_actionable_error(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _parse(text)


def test_tool_is_a_responses_custom_grammar_tool() -> None:
    tool = build_freeform_shell_tool(shell_name="bash", tool_name="shell")
    payload = get_openai_responses_custom_tool_payload(tool)
    assert payload is not None
    assert payload["type"] == "custom"
    assert payload["format"]["syntax"] == "lark"
    assert "# @shell:" in payload["format"]["definition"]
    assert "not JSON" in (tool.description or "")


@pytest.mark.asyncio
async def test_runtime_executes_raw_input_and_background_pragma() -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("freeform-test"),
        model_shell_tool_name="shell",
        config=Settings(
            shell_execution=ShellSettings(tool_profile="freeform_shell", show_bash=False)
        ),
    )
    assert [tool.name for tool in runtime.tools] == ["shell", "process"]
    assert (runtime.tools[0].meta or {}).get(OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY)
    assert "returned by shell" in (runtime.tools[1].description or "")

    result = await runtime.call_tool("shell", {"input": 'echo "quoted \\\\ ok"'})
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert not result.is_error
    assert "quoted \\ ok" in block.text

    started = await runtime.call_tool("shell", {"input": '# @shell: {"background": true}\nsleep 5'})
    started_block = started.content[0]
    assert isinstance(started_block, TextContent)
    match = re.search(r"process-[0-9a-z]+", started_block.text)
    assert match is not None
    await runtime.call_tool("process", {"process_id": match.group(0), "action": "stop"})


@pytest.mark.asyncio
async def test_gpt6_luna_agent_exposes_freeform_shell_when_selected() -> None:
    agent = McpAgent(
        config=AgentConfig(
            name="test",
            instruction="Instruction",
            servers=[],
            shell=True,
            model="codexresponses.gpt-6-luna?reasoning=low",
        ),
        context=Context(
            config=Settings(shell_execution=ShellSettings(tool_profile="freeform_shell"))
        ),
    )
    try:
        tools = {tool.name: tool for tool in (await agent.list_tools()).tools}
        assert (tools["shell"].meta or {}).get(OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY)
        assert {"process", "read_text_file", "write_text_file", "edit_file"} <= set(tools)
    finally:
        await agent._aggregator.close()
