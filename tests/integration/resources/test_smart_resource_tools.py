from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_smart_resource_tools_with_runtime_mcp_attach(fast_agent) -> None:
    fast = fast_agent

    @fast.smart(name="smart_ops", model="passthrough", skills=[])
    async def smart_ops():
        async with fast.run() as app:
            target = "uv run mcp_resource_template_server.py"

            listing = await app.smart_ops.smart_list_resources(
                agent_card_path="smart_resource_worker.md",
                mcp_connect=[target],
            )
            assert "resource://smart/items/{item_id}" in listing

            completed = await app.smart_ops.smart_complete_resource_argument(
                agent_card_path="smart_resource_worker.md",
                template_uri="resource://smart/items/{item_id}",
                argument_name="item_id",
                value="a",
                mcp_connect=[target],
            )
            assert "alpha" in completed.splitlines()

            inspection = await app.smart_ops.smart_get_resource(
                agent_card_path="smart_resource_worker.md",
                resource_uri="resource://smart/items/alpha",
                mcp_connect=[target],
            )
            assert "item:alpha" in inspection

            response = await app.smart_ops.smart_with_resource(
                agent_card_path="smart_resource_worker.md",
                message="hello smart resource",
                resource_uri="resource://smart/items/alpha",
                mcp_connect=[target],
            )
            assert "hello smart resource" in response

    await smart_ops()
