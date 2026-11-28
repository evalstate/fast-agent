from __future__ import annotations

from typing import Any, cast

import pytest

from fast_agent.ui.prompt.turn_input_commands import (
    execute_concurrent_command,
    parse_concurrent_command,
)


@pytest.mark.parametrize(
    "text",
    [
        "/usage",
        "/mcp status",
        "/tools",
        "/mcp",
        "/skills",
        "/process",
        "/warm status",
        "/help",
        "/commands",
        "/history",
        "/system",
        "/markdown",
        "/environment",
        "/check",
        "/model status",
        "/prompts",
        "/session list",
        "/subagents list",
        "/subagents status",
        "/events",
        "/a2a status",
        "/a2a tasks",
        "!",
        "!   ",
        "!!",
        "!echo hello",
        "!git status",
    ],
)
def test_parse_concurrent_command_allows_readonly_and_shell(text: str) -> None:
    assert parse_concurrent_command(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "/clear",
        "/compact",
        "/model sonnet",
        "/session new",
        "/session resume 1",
        "/warm 5",
        "/process attach 1",
        "/process kill 1",
        "/prompt load foo",
        "@other",
        "#other ping",
        "!!!echo hi",
    ],
)
def test_parse_concurrent_command_rejects_mutating(text: str) -> None:
    assert parse_concurrent_command(text) is None


@pytest.mark.asyncio
async def test_execute_concurrent_shell_command_prints_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        "fast_agent.ui.prompt.turn_input_commands.rich_print",
        lambda msg="": printed.append(str(msg)),
    )

    parsed = parse_concurrent_command("!echo dock_concurrent_ok")
    assert parsed is not None

    await execute_concurrent_command(
        parsed,
        raw_text="!echo dock_concurrent_ok",
        prompt_prefix="❯",
        interactive_prompt=cast("Any", object()),
        prompt_provider=cast("Any", object()),
        agent_name="agent",
        available_agents=["agent"],
    )

    combined = "\n".join(printed)
    assert "!echo dock_concurrent_ok" in combined
    assert "dock_concurrent_ok" in combined


@pytest.mark.asyncio
async def test_execute_concurrent_slash_command_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    dispatched: list[Any] = []

    monkeypatch.setattr(
        "fast_agent.ui.prompt.turn_input_commands.rich_print",
        lambda msg="": printed.append(str(msg)),
    )
    monkeypatch.setattr(
        "fast_agent.ui.prompt.special_commands.handle_special_commands",
        lambda parsed, _provider: parsed,
    )

    async def fake_dispatch(_prompt: Any, payload: Any, **_kwargs: Any) -> Any:
        dispatched.append(payload)
        return None

    monkeypatch.setattr(
        "fast_agent.ui.interactive.command_dispatch.dispatch_command_payload",
        fake_dispatch,
    )

    parsed = parse_concurrent_command("/usage")
    assert parsed is not None

    await execute_concurrent_command(
        parsed,
        raw_text="/usage",
        prompt_prefix="❯",
        interactive_prompt=cast("Any", object()),
        prompt_provider=cast("Any", object()),
        agent_name="agent",
        available_agents=["agent"],
    )

    assert len(dispatched) == 1
    assert any("/usage" in line for line in printed)


def test_suspend_console_output_buffers_and_flushes_in_order() -> None:
    from io import StringIO

    from fast_agent.ui.console import SurrogateSafeConsole, suspend_console_output

    buf = StringIO()
    test_console = SurrogateSafeConsole(file=buf, force_terminal=False, color_system=None)

    test_console.print("before")
    assert "before" in buf.getvalue()

    with suspend_console_output():
        test_console.print("first-buffered")
        test_console.print("second-buffered")
        assert "first-buffered" not in buf.getvalue()
        assert "second-buffered" not in buf.getvalue()

    out = buf.getvalue()
    assert "first-buffered" in out
    assert "second-buffered" in out
    assert out.index("before") < out.index("first-buffered") < out.index("second-buffered")
