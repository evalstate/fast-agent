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
async def test_forms_handler_defers_url_elicitation_when_request_context_active(capsys) -> None:
    session = object.__new__(MCPAgentClientSession)
    session.session_server_name = "session-server"
    session.server_config = None
    session.agent_name = "test-agent"
    session._ensure_url_elicitation_tracking_state()

    request_tracking_id = session._reserve_url_elicitation_request_tracking_id()
    token = session._active_url_elicitation_request_id.set(request_tracking_id)
    try:
        context: Any = _ContextWithSession(session=session)
        params = ElicitRequestURLParams(
            mode="url",
            message="Open browser to continue",
            url="https://example.com/continue",
            elicitationId="form-url-1",
        )

        result = await forms_elicitation_handler(cast("Any", context), cast("Any", params))
        assert result.action == "accept"

        tool_result = CallToolResult(content=[], isError=False)
        session._attach_deferred_url_elicitation_payload_for_active_request(
            tool_result,
            request_method="tools/call",
        )
    finally:
        session._active_url_elicitation_request_id.reset(token)

    payload = MCPAgentClientSession.get_url_elicitation_required_payload(tool_result)
    assert payload is not None
    assert payload.server_name == "session-server"
    assert payload.request_method == "tools/call"
    assert len(payload.elicitations) == 1
    assert payload.elicitations[0].message == "Open browser to continue"
    assert payload.elicitations[0].url == "https://example.com/continue"
    assert payload.elicitations[0].elicitation_id == "form-url-1"

    captured = capsys.readouterr()
    assert captured.out.strip() == ""
