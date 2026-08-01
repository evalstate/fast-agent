import asyncio
import os
import subprocess

import pytest
from mcp_types import TextContent


async def _list_tools_call_count(client) -> int:
    result = await client.call_tool("list_tools_call_count")
    content = result.content[0]
    assert isinstance(content, TextContent)
    return int(content.text)


async def _read_resource_call_count(client) -> int:
    result = await client.call_tool("read_resource_call_count")
    content = result.content[0]
    assert isinstance(content, TextContent)
    return int(content.text)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_modern_tool_subscription_refreshes_cache(
    fast_agent,
    mcp_test_ports,
    wait_for_port,
) -> None:
    process = subprocess.Popen(
        ["uv", "run", "subscription_server.py"],
        cwd=os.path.dirname(__file__),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        await wait_for_port("127.0.0.1", mcp_test_ports["http"], process=process)

        @fast_agent.agent(name="probe", model="passthrough", servers=["modern_http"])
        async def run_probe() -> None:
            async with fast_agent.run() as app:
                before = {tool.name for tool in (await app.probe.list_tools()).tools}
                assert "modern_http__dynamic_echo" not in before

                manager = app.probe._aggregator._persistent_connection_manager
                assert manager is not None
                connection = manager.running_servers["modern_http"]
                assert connection.client is not None
                for _ in range(50):
                    status = (await app.probe.get_server_status())["modern_http"]
                    if status.subscription_state == "open":
                        break
                    await asyncio.sleep(0.1)
                else:
                    pytest.fail("modern subscription did not finish authoritative refresh")

                list_count = await _list_tools_call_count(connection.client)
                await connection.client.list_tools()
                await connection.client.list_tools()
                assert await _list_tools_call_count(connection.client) == list_count

                assert status.protocol_era == "modern"
                assert status.transport_channels is not None
                assert status.transport_channels.listen is not None
                assert status.transport_channels.listen.state == "open"
                assert status.transport_channels.listen.request_count >= 2

                await app.probe.call_tool("add_dynamic_tool", {})
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    after = {tool.name for tool in (await app.probe.list_tools()).tools}
                    if "modern_http__dynamic_echo" in after:
                        break
                else:
                    pytest.fail("tools/list cache was not refreshed by the subscription event")

                status = (await app.probe.get_server_status())["modern_http"]
                assert status.subscription_state == "open"
                assert status.transport_channels is not None
                assert status.transport_channels.listen is not None
                assert status.transport_channels.listen.notification_count >= 1

        await run_probe()
    finally:
        process.terminate()
        process.communicate(timeout=5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_modern_resource_subscription_rotates_and_refreshes_updates(
    fast_agent,
    mcp_test_ports,
    wait_for_port,
) -> None:
    process = subprocess.Popen(
        ["uv", "run", "subscription_server.py"],
        cwd=os.path.dirname(__file__),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        await wait_for_port("127.0.0.1", mcp_test_ports["http"], process=process)

        @fast_agent.agent(name="probe", model="passthrough", servers=["modern_http"])
        async def run_probe() -> None:
            async with fast_agent.run() as app:
                aggregator = app.probe._aggregator
                manager = aggregator._persistent_connection_manager
                assert manager is not None
                connection = manager.running_servers["modern_http"]
                assert connection.client is not None

                status = (await app.probe.get_server_status())["modern_http"]
                assert status.transport_channels is not None
                assert status.transport_channels.listen is not None
                initial_listen_requests = status.transport_channels.listen.request_count

                await app.probe.call_tool("add_dynamic_app", {})
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    if aggregator.selected_materialized_resource_uris("modern_http") == (
                        "ui://component/dynamic",
                        "ui://component/initial",
                    ):
                        break
                else:
                    pytest.fail("resource-list event did not materialize the dynamic app")

                for _ in range(50):
                    status = (await app.probe.get_server_status())["modern_http"]
                    assert status.transport_channels is not None
                    assert status.transport_channels.listen is not None
                    if status.transport_channels.listen.request_count > initial_listen_requests:
                        break
                    await asyncio.sleep(0.1)
                else:
                    pytest.fail("canonical resource URI change did not rotate the listener")

                reads_before = await _read_resource_call_count(connection.client)
                await app.probe.call_tool("update_dynamic_app", {})
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    if await _read_resource_call_count(connection.client) > reads_before:
                        break
                else:
                    pytest.fail("resource update did not force an authoritative resource read")

                assert connection.subscription_state == "open"

        await run_probe()
    finally:
        process.terminate()
        process.communicate(timeout=5)
