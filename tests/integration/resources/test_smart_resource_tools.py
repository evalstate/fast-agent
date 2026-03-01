from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_smart_internal_resource_tools(fast_agent) -> None:
    fast = fast_agent

    @fast.smart(name="smart_ops", model="passthrough", skills=[])
    async def smart_ops():
        async with fast.run() as app:
            listing = await app.smart_ops.resource_list()
            assert "internal://fast-agent/smart-agent-cards" in listing

            resource = await app.smart_ops.resource_read("internal://fast-agent/smart-agent-cards")
            assert "Agent Card (type: `agent`)" in resource

    await smart_ops()


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
            assert "internal://fast-agent/smart-agent-cards" in listing
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_smart_slash_command_tool_operations(fast_agent) -> None:
    fast = fast_agent

    @fast.smart(name="smart_ops", model="passthrough", skills=[])
    async def smart_ops():
        async with fast.run() as app:
            skills_result = await app.smart_ops.slash_command("/skills list")
            assert "# skills.list" in skills_result

            cards_result = await app.smart_ops.slash_command("/cards list")
            assert "# cards.list" in cards_result

            models_result = await app.smart_ops.slash_command("/models doctor")
            assert "# models.doctor" in models_result

    await smart_ops()
