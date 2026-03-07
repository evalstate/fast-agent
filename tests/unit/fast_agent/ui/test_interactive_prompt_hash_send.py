from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.ui.interactive_prompt import InteractivePrompt

if TYPE_CHECKING:
    from mcp.types import PromptMessage

    from fast_agent.types import PromptMessageExtended


@pytest.mark.asyncio
async def test_execute_hash_send_quiet_loads_response_without_status_chatter(capsys) -> None:
    prompt_ui = InteractivePrompt()
    calls: list[tuple[str, str]] = []

    async def quiet_send(
        message: str | PromptMessage | PromptMessageExtended,
        agent_name: str,
    ) -> str:
        assert isinstance(message, str)
        calls.append((message, agent_name))
        return "delegated response"

    execution = await prompt_ui._execute_hash_send(
        send_func=quiet_send,
        target_agent="helper",
        message="hello",
        quiet=True,
        clear_progress_for_agent=lambda _agent: None,
        clear_ctrl_c_interrupt=lambda: None,
        handle_inflight_cancel=lambda: None,
        last_assistant_message_cancelled=lambda _agent: False,
    )

    output = capsys.readouterr().out
    assert calls == [("hello", "helper")]
    assert execution.buffer_prefill == "delegated response"
    assert "Asking helper" not in output
    assert "loaded into input buffer" not in output


@pytest.mark.asyncio
async def test_execute_hash_send_verbose_keeps_existing_status_chatter(capsys) -> None:
    prompt_ui = InteractivePrompt()

    async def verbose_send(
        message: str | PromptMessage | PromptMessageExtended,
        agent_name: str,
    ) -> str:
        assert isinstance(message, str)
        return f"{agent_name}: {message}"

    execution = await prompt_ui._execute_hash_send(
        send_func=verbose_send,
        target_agent="helper",
        message="hello",
        quiet=False,
        clear_progress_for_agent=lambda _agent: None,
        clear_ctrl_c_interrupt=lambda: None,
        handle_inflight_cancel=lambda: None,
        last_assistant_message_cancelled=lambda _agent: False,
    )

    output = capsys.readouterr().out
    assert execution.buffer_prefill == "helper: hello"
    assert "Asking helper" in output
    assert "Response from helper loaded into input buffer" in output


@pytest.mark.asyncio
async def test_execute_hash_send_quiet_suppresses_real_shell_output(capsys) -> None:
    prompt_ui = InteractivePrompt()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        timeout_seconds=10,
    )

    async def quiet_shell_send(
        message: str | PromptMessage | PromptMessageExtended,
        agent_name: str,
    ) -> str:
        assert isinstance(message, str)
        assert agent_name == "helper"
        result = await runtime.execute({"command": "echo hello"})
        content = result.content
        assert content is not None
        first_item = content[0]
        assert isinstance(first_item, TextContent)
        return first_item.text

    execution = await prompt_ui._execute_hash_send(
        send_func=quiet_shell_send,
        target_agent="helper",
        message="run shell",
        quiet=True,
        clear_progress_for_agent=lambda _agent: None,
        clear_ctrl_c_interrupt=lambda: None,
        handle_inflight_cancel=lambda: None,
        last_assistant_message_cancelled=lambda _agent: False,
    )

    output = capsys.readouterr().out
    assert execution.buffer_prefill is not None
    assert "hello" in execution.buffer_prefill
    assert "process exit code was 0" in execution.buffer_prefill
    assert "hello" not in output
    assert "exit code" not in output
    assert "Asking helper" not in output
