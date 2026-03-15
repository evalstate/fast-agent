from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_repo_hook():
    repo_root = Path(__file__).resolve().parents[4]
    hook_path = repo_root / ".fast-agent" / "hooks" / "ripgrep_loop_guard.py"
    spec = importlib.util.spec_from_file_location("test_repo_ripgrep_loop_guard", hook_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_ctx(command: str) -> SimpleNamespace:
    tool_call = SimpleNamespace(
        params=SimpleNamespace(name="execute", arguments={"command": command})
    )
    user_message = SimpleNamespace(
        role="user",
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "repo_root": "/home/shaun/source/fast-agent-pr",
                        "objective": "find mcp connect",
                        "max_commands": 4,
                    }
                ),
            }
        ],
    )
    ctx = SimpleNamespace(
        hook_type="before_tool_call",
        message=SimpleNamespace(tool_calls={"call_1": tool_call}),
        runner=SimpleNamespace(delta_messages=[user_message]),
        agent_name="ripgrep_search[1]",
        message_history=[],
    )
    return ctx


@pytest.mark.asyncio
async def test_ripgrep_loop_guard_does_not_append_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_repo_hook()
    monkeypatch.setattr(module, "_supports_pcre2", lambda _ctx: True)

    command = "rg -n 'mcp connect' -g '*' src tests docs README examples"
    ctx = _build_ctx(command)

    await module.ripgrep_loop_guard(ctx)

    tool_call = ctx.message.tool_calls["call_1"]
    assert tool_call.params.arguments["command"] == command


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "find src -name '*.py'",
        "fd mcp src",
        "ls src | wc -l",
    ],
)
async def test_ripgrep_loop_guard_allows_simple_shell_commands(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    module = _load_repo_hook()
    monkeypatch.setattr(module, "_supports_pcre2", lambda _ctx: True)
    ctx = _build_ctx(command)

    await module.ripgrep_loop_guard(ctx)

    tool_call = ctx.message.tool_calls["call_1"]
    assert tool_call.params.arguments["command"] == command
