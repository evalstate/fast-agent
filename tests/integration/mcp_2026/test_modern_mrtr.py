import pytest
from mcp.client.session import ClientRequestContext
from mcp_types import (
    ElicitRequestFormParams,
    ElicitRequestParams,
    ElicitRequestURLParams,
    ElicitResult,
)

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


async def approve_sandbox(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    del context
    assert params.message == "Start t4-small sandbox at $0.40 per hour?"
    assert isinstance(params, ElicitRequestFormParams)
    budget_schema = params.requested_schema["properties"]["max_budget_usd"]
    assert budget_schema["type"] == "number"
    assert budget_schema["minimum"] == 0.4
    assert budget_schema["maximum"] == 3.2
    return ElicitResult(
        action="accept",
        content={
            "region": "eu-west-1",
            "duration_hours": 2,
            "max_budget_usd": 1.00,
        },
    )


async def approve_over_budget_sandbox(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    del context
    assert isinstance(params, ElicitRequestFormParams)
    return ElicitResult(
        action="accept",
        content={
            "region": "eu-west-1",
            "duration_hours": 2,
            "max_budget_usd": 0.40,
        },
    )


async def decline_sandbox(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    del context
    assert params.message == "Start t4-small sandbox at $0.40 per hour?"
    return ElicitResult(action="decline")


async def cancel_sandbox(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    del context
    assert params.message == "Start t4-small sandbox at $0.40 per hour?"
    return ElicitResult(action="cancel")


def assert_url_request(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> None:
    assert context.request_id == "authorize_console"
    assert isinstance(params, ElicitRequestURLParams)
    assert params.message == "Authorize browser access to sandbox sim-t4-small-001."
    assert str(params.url) == ("https://example.com/authorize?sandbox=sim-t4-small-001")
    assert params.elicitation_id is None


async def accept_url(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    assert_url_request(context, params)
    return ElicitResult(action="accept")


async def decline_url(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    assert_url_request(context, params)
    return ElicitResult(action="decline")


async def cancel_url(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    assert_url_request(context, params)
    return ElicitResult(action="cancel")


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
            assert status.call_counts["discover"] == 1
            assert "initialize" not in status.call_counts
            assert status.subscription_state == "open"
            assert status.ping_ok_count == 0
            assert status.ping_fail_count == 0

    await run_probe()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sandbox_example_uses_request_scoped_elicitation(fast_agent) -> None:
    @fast_agent.agent(
        name="sandbox",
        model="passthrough",
        servers=["sandbox_example"],
        elicitation_handler=approve_sandbox,
    )
    async def run_sandbox() -> None:
        async with fast_agent.run() as app:
            result = await app.sandbox.call_tool("start_t4_small_sandbox", {})

            assert result.is_error is False
            output = get_text(result.content[0])
            assert output is not None
            assert "SIMULATED: started sandbox sim-t4-small-001" in output
            assert "Region: eu-west-1" in output
            assert "Maximum cost: $0.80" in output
            assert "Maximum budget: $1.00" in output
            assert "No infrastructure or billing was created." in output

            status = (await app.sandbox.get_server_status())["sandbox_example"]
            assert status.protocol_version == "2026-07-28"
            assert status.protocol_era == "modern"

    await run_sandbox()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sandbox_example_enforces_maximum_budget(fast_agent) -> None:
    @fast_agent.agent(
        name="sandbox_over_budget",
        model="passthrough",
        servers=["sandbox_example"],
        elicitation_handler=approve_over_budget_sandbox,
    )
    async def run_sandbox() -> None:
        async with fast_agent.run() as app:
            result = await app.sandbox_over_budget.call_tool("start_t4_small_sandbox", {})

            assert result.is_error is False
            assert get_text(result.content[0]) == (
                "SIMULATED: estimated maximum cost $0.80 exceeds maximum budget $0.40; "
                "no sandbox was created"
            )

    await run_sandbox()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sandbox_example_handles_declined_elicitation(fast_agent) -> None:
    @fast_agent.agent(
        name="sandbox_declined",
        model="passthrough",
        servers=["sandbox_example"],
        elicitation_handler=decline_sandbox,
    )
    async def run_sandbox() -> None:
        async with fast_agent.run() as app:
            result = await app.sandbox_declined.call_tool("start_t4_small_sandbox", {})

            assert result.is_error is False
            assert get_text(result.content[0]) == (
                "SIMULATED: start declined; no sandbox was created"
            )

    await run_sandbox()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sandbox_example_handles_cancelled_elicitation(fast_agent) -> None:
    @fast_agent.agent(
        name="sandbox_cancelled",
        model="passthrough",
        servers=["sandbox_example"],
        elicitation_handler=cancel_sandbox,
    )
    async def run_sandbox() -> None:
        async with fast_agent.run() as app:
            result = await app.sandbox_cancelled.call_tool("start_t4_small_sandbox", {})

            assert result.is_error is False
            assert get_text(result.content[0]) == (
                "SIMULATED: start cancelled; no sandbox was created"
            )

    await run_sandbox()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_url_example_retries_originating_request_after_accept(fast_agent) -> None:
    @fast_agent.agent(
        name="url_accepted",
        model="passthrough",
        servers=["url_example"],
        elicitation_handler=accept_url,
    )
    async def run_url() -> None:
        async with fast_agent.run() as app:
            result = await app.url_accepted.call_tool(
                "request_console_access",
                {"sandbox_id": "sim-t4-small-001"},
            )

            assert result.is_error is False
            assert get_text(result.content[0]) == (
                "SIMULATED: navigation accepted for sim-t4-small-001; "
                "external completion was not verified and no console access was granted"
            )

            status = (await app.url_accepted.get_server_status())["url_example"]
            assert status.protocol_version == "2026-07-28"
            assert status.protocol_era == "modern"

    await run_url()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_url_example_retries_originating_request_after_decline(fast_agent) -> None:
    @fast_agent.agent(
        name="url_declined",
        model="passthrough",
        servers=["url_example"],
        elicitation_handler=decline_url,
    )
    async def run_url() -> None:
        async with fast_agent.run() as app:
            result = await app.url_declined.call_tool(
                "request_console_access",
                {"sandbox_id": "sim-t4-small-001"},
            )

            assert result.is_error is False
            assert get_text(result.content[0]) == (
                "SIMULATED: navigation declined for sim-t4-small-001; no URL was opened"
            )

    await run_url()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_url_example_retries_originating_request_after_cancel(fast_agent) -> None:
    @fast_agent.agent(
        name="url_cancelled",
        model="passthrough",
        servers=["url_example"],
        elicitation_handler=cancel_url,
    )
    async def run_url() -> None:
        async with fast_agent.run() as app:
            result = await app.url_cancelled.call_tool(
                "request_console_access",
                {"sandbox_id": "sim-t4-small-001"},
            )

            assert result.is_error is False
            assert get_text(result.content[0]) == (
                "SIMULATED: navigation cancelled for sim-t4-small-001; no URL was opened"
            )

    await run_url()


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
            assert modern.negotiation == "discover"
            assert modern.supported_protocol_versions == ("2026-07-28",)
            assert modern.call_counts["discover"] == 1
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
            assert reconnected["forced_modern"].call_counts["discover"] == 2
            assert "initialize" not in reconnected["forced_modern"].call_counts
            assert reconnected["forced_legacy"].call_counts["initialize"] == 2

    await run_probe()
