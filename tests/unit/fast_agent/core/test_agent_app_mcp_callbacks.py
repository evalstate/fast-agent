from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.core.agent_app import AgentApp
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config = SimpleNamespace(default=True)


@pytest.mark.asyncio
async def test_agent_app_mcp_callback_roundtrip() -> None:
    calls: list[tuple[str, str]] = []

    async def attach_cb(agent_name, server_name, server_config, options):
        del server_config, options
        calls.append(("attach", f"{agent_name}:{server_name}"))
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=["demo.echo"],
            prompts_added=["demo.prompt"],
            warnings=[],
        )

    async def detach_cb(agent_name, server_name):
        calls.append(("detach", f"{agent_name}:{server_name}"))
        return MCPDetachResult(
            server_name=server_name,
            detached=True,
            tools_removed=["demo.echo"],
            prompts_removed=["demo.prompt"],
        )

    async def list_attached_cb(agent_name):
        calls.append(("list_attached", agent_name))
        return ["demo"]

    async def list_configured_cb(agent_name):
        calls.append(("list_configured", agent_name))
        return ["other"]

    app = AgentApp(
        agents={"main": cast("AgentProtocol", _Agent("main"))},
        attach_mcp_server_callback=attach_cb,
        detach_mcp_server_callback=detach_cb,
        list_attached_mcp_servers_callback=list_attached_cb,
        list_configured_detached_mcp_servers_callback=list_configured_cb,
    )

    attach_result = await app.attach_mcp_server("main", "demo")
    detach_result = await app.detach_mcp_server("main", "demo")
    attached = await app.list_attached_mcp_servers("main")
    configured = await app.list_configured_detached_mcp_servers("main")

    assert attach_result.tools_added == ["demo.echo"]
    assert detach_result.prompts_removed == ["demo.prompt"]
    assert attached == ["demo"]
    assert configured == ["other"]
    assert calls == [
        ("attach", "main:demo"),
        ("detach", "main:demo"),
        ("list_attached", "main"),
        ("list_configured", "main"),
    ]


@pytest.mark.asyncio
async def test_agent_app_mcp_methods_require_callbacks() -> None:
    app = AgentApp(agents={"main": cast("AgentProtocol", _Agent("main"))})

    with pytest.raises(RuntimeError, match="attachment"):
        await app.attach_mcp_server("main", "demo")

    with pytest.raises(RuntimeError, match="detachment"):
        await app.detach_mcp_server("main", "demo")

    with pytest.raises(RuntimeError, match="listing"):
        await app.list_attached_mcp_servers("main")
