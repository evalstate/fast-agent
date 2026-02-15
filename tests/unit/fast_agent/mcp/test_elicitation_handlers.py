from dataclasses import dataclass
from typing import Any, cast

import pytest
from mcp.types import CallToolResult, ElicitRequestURLParams

from fast_agent.mcp.elicitation_handlers import forms_elicitation_handler
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession


@dataclass
class _ContextWithSession:
    session: MCPAgentClientSession


@pytest.mark.asyncio
async def test_forms_handler_defers_url_elicitation_to_result_payload(capsys) -> None:
    session = object.__new__(MCPAgentClientSession)
    session.session_server_name = "session-server"
    session.server_config = None
    session.agent_name = "test-agent"

    context: Any = _ContextWithSession(session=session)
    params = ElicitRequestURLParams(
        mode="url",
        message="Open browser to continue",
        url="https://example.com/continue",
        elicitationId="form-url-1",
    )

    result = await forms_elicitation_handler(cast("Any", context), cast("Any", params))
    assert result.action == "accept"

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    tool_result = CallToolResult(content=[], isError=False)
    session._attach_pending_url_elicitation_payload_for_request(
        tool_result,
        request_method="tools/call",
    )

    payload = MCPAgentClientSession.get_url_elicitation_required_payload(tool_result)
    assert payload is not None
    assert payload.server_name == "session-server"
    assert payload.request_method == "tools/call"
    assert len(payload.elicitations) == 1
    assert payload.elicitations[0].message == "Open browser to continue"
    assert payload.elicitations[0].url == "https://example.com/continue"
    assert payload.elicitations[0].elicitation_id == "form-url-1"
