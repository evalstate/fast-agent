from __future__ import annotations

import pytest

from fast_agent.command_actions import PluginRuntimeFacade
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult


@pytest.mark.asyncio
async def test_plugin_runtime_mcp_methods_default_to_current_agent() -> None:
    calls: list[tuple[str, str, str]] = []

    async def attach(agent_name, server_name, server_config=None, options=None):
        calls.append(("attach", agent_name, server_name))
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=[],
            prompts_added=[],
            warnings=[],
        )

    async def detach(agent_name, server_name):
        calls.append(("detach", agent_name, server_name))
        return MCPDetachResult(
            server_name=server_name,
            detached=True,
            tools_removed=[],
            prompts_removed=[],
        )

    async def list_attached(agent_name):
        calls.append(("list-attached", agent_name, ""))
        return ["github"]

    async def list_detached(agent_name):
        calls.append(("list-detached", agent_name, ""))
        return ["linear"]

    runtime = PluginRuntimeFacade(
        current_agent_name="dev",
        attach_mcp_server_callback=attach,
        detach_mcp_server_callback=detach,
        list_attached_mcp_servers_callback=list_attached,
        list_configured_detached_mcp_servers_callback=list_detached,
    )

    attach_result = await runtime.attach_mcp_server(server_name="github")
    detach_result = await runtime.detach_mcp_server(server_name="github")
    attached = await runtime.list_attached_mcp_servers()
    detached = await runtime.list_configured_detached_mcp_servers()

    assert attach_result.attached is True
    assert detach_result.detached is True
    assert attached == ("github",)
    assert detached == ("linear",)
    assert calls == [
        ("attach", "dev", "github"),
        ("detach", "dev", "github"),
        ("list-attached", "dev", ""),
        ("list-detached", "dev", ""),
    ]


@pytest.mark.asyncio
async def test_plugin_runtime_mcp_methods_accept_explicit_agent() -> None:
    calls: list[tuple[str, str]] = []

    async def list_attached(agent_name):
        calls.append(("list-attached", agent_name))
        return []

    runtime = PluginRuntimeFacade(
        current_agent_name="dev",
        list_attached_mcp_servers_callback=list_attached,
    )

    assert await runtime.list_attached_mcp_servers(agent_name="planner") == ()
    assert calls == [("list-attached", "planner")]


@pytest.mark.asyncio
async def test_plugin_runtime_mcp_methods_raise_when_unavailable() -> None:
    runtime = PluginRuntimeFacade(current_agent_name="dev")

    with pytest.raises(RuntimeError, match="attachment is not available"):
        await runtime.attach_mcp_server(server_name="github")
