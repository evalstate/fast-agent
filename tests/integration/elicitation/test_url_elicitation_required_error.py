"""Integration tests for URL_ELICITATION_REQUIRED (-32042) handling."""

import pytest


def _captured_output_text(capsys) -> str:
    captured = capsys.readouterr()
    return f"{captured.out}\n{captured.err}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_url_elicitation_required_error_displays_required_elicitations(fast_agent, capsys):
    fast = fast_agent

    @fast.agent("url_required_valid_agent", servers=["url_elicitation_required"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent["url_required_valid_agent"].call_tool("url_required_valid_tool")
            assert result.isError is True
            payload = getattr(result, "_fast_agent_url_elicitation_required", None)
            assert payload is not None
            assert len(payload.elicitations) == 2
            assert payload.elicitations[0].elicitation_id == "valid-1"
            assert payload.elicitations[1].elicitation_id == "valid-2"

    await agent_function()

    output = _captured_output_text(capsys)
    assert output.strip() == ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_url_elicitation_required_error_reports_malformed_data(fast_agent, capsys):
    fast = fast_agent

    @fast.agent("url_required_malformed_agent", servers=["url_elicitation_required"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent["url_required_malformed_agent"].call_tool(
                "url_required_malformed_tool"
            )
            assert result.isError is True
            payload = getattr(result, "_fast_agent_url_elicitation_required", None)
            assert payload is not None
            assert len(payload.elicitations) == 0
            assert any("elicitations is empty" in issue for issue in payload.issues)

    await agent_function()

    output = _captured_output_text(capsys)
    assert output.strip() == ""
