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
    ],
)
def test_parse_concurrent_command_allows_readonly_slash_commands(text: str) -> None:
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
        "!",
        "!   ",
        "!echo hi",
        "!!echo hi",
        "!!!echo hi",
    ],
)
def test_parse_concurrent_command_defers_shell_and_other_input(text: str) -> None:
    assert parse_concurrent_command(text) is None


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
