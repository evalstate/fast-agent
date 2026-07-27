import pytest
from mcp.client.session import ClientRequestContext
from mcp_types import ElicitRequestFormParams, ElicitRequestParams, ElicitResult

from fast_agent.mcp.helpers.content_helpers import get_text


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
            assert status.subscription_state == "open"
            assert status.ping_ok_count == 0
            assert status.ping_fail_count == 0

    await run_probe()
