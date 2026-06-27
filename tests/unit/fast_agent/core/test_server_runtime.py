from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.core.server_runtime import (
    ServerRuntimeContext,
    render_mcp_tool_name,
    run_mcp_server,
)

if TYPE_CHECKING:
    from fast_agent.core.fastagent import ManagedRunState, RunSettings, RuntimeCallbacks


def test_render_mcp_tool_name_uses_requested_agent_name() -> None:
    assert render_mcp_tool_name("ask_{agent}", agent="reviewer") == "ask_reviewer"


def test_render_mcp_tool_name_uses_cli_default_agent_placeholder() -> None:
    assert render_mcp_tool_name("ask_{agent}", agent=None) == "ask_agent"
    assert render_mcp_tool_name(None, agent="reviewer") is None


@pytest.mark.asyncio
async def test_run_mcp_server_threads_tool_name_template(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_options = []

    async def fake_run_harness_mcp_app_server(**kwargs):
        captured_options.append(kwargs["options"])

    monkeypatch.setattr(
        "fast_agent.mcp.server.harness_app_server.run_harness_mcp_app_server",
        fake_run_harness_mcp_app_server,
    )
    context = ServerRuntimeContext(
        app_name="demo",
        args=SimpleNamespace(
            server_name=None,
            server_description=None,
            tool_name_template="ask_{agent}",
            tool_description=None,
            agent="reviewer",
            transport="http",
            host="127.0.0.1",
            port=8000,
            instance_scope="shared",
        ),
        callbacks=cast("RuntimeCallbacks", SimpleNamespace(instance_factory=lambda: object())),
        state=cast(
            "ManagedRunState",
            SimpleNamespace(runtime=SimpleNamespace(shell_executor=object())),
        ),
        config=None,
        skills_directory_override=None,
        settings=cast("RunSettings", SimpleNamespace()),
        acp_server_factory=lambda: object,
    )

    await run_mcp_server(context)

    assert captured_options[0].tool_name == "ask_reviewer"
    assert captured_options[0].default_agent == "reviewer"
