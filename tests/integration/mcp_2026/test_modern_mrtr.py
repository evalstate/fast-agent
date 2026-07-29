import pytest
from mcp.client.session import ClientRequestContext
from mcp_types import ElicitRequestFormParams, ElicitRequestParams, ElicitResult

from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions


async def accept_profile(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    del context
    assert params.message == "Provide a profile"
    assert isinstance(params, ElicitRequestFormParams)
    assert params.requested_schema["required"] == ["name", "age"]
    return ElicitResult(action="accept", content={"name": "Ada", "age": 37})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_modern_negotiation_and_mrtr(fast_agent) -> None:
    @fast_agent.agent(
        name="probe",
        model="passthrough",
        servers=["modern"],
        elicitation_handler=accept_profile,
    )
    async def run_probe() -> None:
        async with fast_agent.run() as app:
            result = await app.probe.call_tool("create_profile", {})

            assert result.is_error is False
            assert get_text(result.content[0]) == "Ada:37"
            resource = await app.probe.get_resource("modern://status", "modern")
            assert get_text(resource.contents[0]) == "modern-ok"
            prompt = await app.probe.get_prompt(
                "hello",
                {"name": "fast-agent"},
                server_name="modern",
            )
            assert get_text(prompt.messages[0].content) == "Hello, fast-agent"

            status = (await app.probe.get_server_status())["modern"]
            assert status.protocol_version == "2026-07-28"
            assert status.protocol_era == "modern"
            assert status.supported_protocol_versions == ("2026-07-28",)
            assert status.negotiation == "discover"
            assert "initialize" not in status.call_counts
            assert status.subscription_state == "open"
            assert status.ping_ok_count == 0
            assert status.ping_fail_count == 0

    await run_probe()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_protocol_mode_can_force_modern_or_legacy(fast_agent) -> None:
    @fast_agent.agent(
        name="forced",
        model="passthrough",
        servers=["forced_modern", "forced_legacy"],
    )
    async def run_probe() -> None:
        async with fast_agent.run() as app:
            statuses = await app.forced.get_server_status()

            modern = statuses["forced_modern"]
            assert modern.protocol_mode == "modern"
            assert modern.protocol_version == "2026-07-28"
            assert modern.protocol_era == "modern"
            assert modern.negotiation == "adopt"
            assert modern.supported_protocol_versions == ()
            assert "initialize" not in modern.call_counts

            legacy = statuses["forced_legacy"]
            assert legacy.protocol_mode == "legacy"
            assert legacy.protocol_era == "legacy"
            assert legacy.negotiation == "initialize"
            assert legacy.call_counts["initialize"] == 1

            reconnect = MCPAttachOptions(force_reconnect=True)
            await app.forced._aggregator.attach_server(
                server_name="forced_modern", options=reconnect
            )
            await app.forced._aggregator.attach_server(
                server_name="forced_legacy", options=reconnect
            )

            reconnected = await app.forced.get_server_status()
            assert "initialize" not in reconnected["forced_modern"].call_counts
            assert reconnected["forced_legacy"].call_counts["initialize"] == 2

    await run_probe()
