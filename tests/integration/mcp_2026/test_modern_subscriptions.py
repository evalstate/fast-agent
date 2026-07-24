import asyncio
import os
import subprocess

import pytest


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

                status = (await app.probe.get_server_status())["modern_http"]
                assert status.protocol_era == "modern"
                assert status.subscription_state == "open"

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

        await run_probe()
    finally:
        process.terminate()
        process.communicate(timeout=5)
