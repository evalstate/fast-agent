"""Tests for URL elicitation required error handling helpers."""

from mcp.shared.exceptions import McpError
from mcp.types import URL_ELICITATION_REQUIRED, CallToolResult, ErrorData

from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from fast_agent.mcp.url_elicitation_required import parse_url_elicitation_required_data


class TestParseUrlElicitationRequiredData:
    def test_parses_valid_data(self) -> None:
        parsed = parse_url_elicitation_required_data(
            {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Authorization is required.",
                        "url": "https://example.com/connect",
                        "elicitationId": "auth-001",
                    }
                ]
            }
        )

        assert parsed.issues == []
        assert len(parsed.elicitations) == 1
        assert parsed.elicitations[0].message == "Authorization is required."
        assert parsed.elicitations[0].url == "https://example.com/connect"
        assert parsed.elicitations[0].elicitationId == "auth-001"

    def test_reports_missing_data(self) -> None:
        parsed = parse_url_elicitation_required_data(None)

        assert parsed.elicitations == []
        assert parsed.issues == ["error.data is missing"]

    def test_reports_non_list_elicitations(self) -> None:
        parsed = parse_url_elicitation_required_data({"elicitations": "not-a-list"})

        assert parsed.elicitations == []
        assert parsed.issues == ["error.data.elicitations must be a list, got str"]

    def test_reports_empty_elicitations_list(self) -> None:
        parsed = parse_url_elicitation_required_data({"elicitations": []})

        assert parsed.elicitations == []
        assert parsed.issues == ["error.data.elicitations is empty"]

    def test_collects_valid_and_invalid_entries(self) -> None:
        parsed = parse_url_elicitation_required_data(
            {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Approve access",
                        "url": "https://example.com/ok",
                        "elicitationId": "ok-1",
                    },
                    {
                        "mode": "url",
                        "message": "Missing URL and elicitation id",
                    },
                    "not-an-object",
                ]
            }
        )

        assert len(parsed.elicitations) == 1
        assert parsed.elicitations[0].elicitationId == "ok-1"
        assert len(parsed.issues) == 2
        assert "error.data.elicitations[1] is invalid" in parsed.issues[0]
        assert parsed.issues[1] == "error.data.elicitations[2] must be an object, got str"

    def test_accepts_snake_case_elicitation_id_with_non_compliant_issue(self) -> None:
        parsed = parse_url_elicitation_required_data(
            {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Authorize",
                        "url": "https://example.com/auth",
                        "elicitation_id": "snake-1",
                    }
                ]
            }
        )

        assert len(parsed.elicitations) == 1
        assert parsed.elicitations[0].elicitationId == "snake-1"
        assert len(parsed.issues) == 1
        assert "non-compliant" in parsed.issues[0]
        assert "elicitation_id" in parsed.issues[0]


class TestUrlElicitationRequiredErrorDetection:
    def _make_session(self) -> MCPAgentClientSession:
        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test"
        return session

    def test_detects_mcp_error_code_32042(self) -> None:
        error = McpError(
            ErrorData(
                code=URL_ELICITATION_REQUIRED,
                message="URL elicitation required",
                data={
                    "elicitations": [
                        {
                            "mode": "url",
                            "message": "Authorize",
                            "url": "https://example.com/auth",
                            "elicitationId": "auth-1",
                        }
                    ]
                },
            )
        )

        assert self._make_session()._is_url_elicitation_required_error(error) is True

    def test_ignores_other_mcp_error_codes(self) -> None:
        error = McpError(ErrorData(code=-32601, message="Method not found"))

        assert self._make_session()._is_url_elicitation_required_error(error) is False

    def test_ignores_non_mcp_exceptions(self) -> None:
        assert self._make_session()._is_url_elicitation_required_error(ValueError("x")) is False


class TestNormalUrlElicitationDeferral:
    def _make_session(self) -> MCPAgentClientSession:
        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test-server"
        session._ensure_url_elicitation_tracking_state()
        return session

    def test_queues_url_elicitation_when_request_context_active(self) -> None:
        session = self._make_session()
        request_tracking_id = session._reserve_url_elicitation_request_tracking_id()
        token = session._active_url_elicitation_request_id.set(request_tracking_id)
        try:
            queued = session.queue_url_elicitation_for_active_request(
                message="Sign in to continue.",
                url="https://example.com/auth",
                elicitation_id="normal-1",
            )
            assert queued is True

            result = CallToolResult(content=[], isError=False)
            session._attach_deferred_url_elicitation_payload_for_active_request(
                result,
                request_method="tools/call",
            )
        finally:
            session._active_url_elicitation_request_id.reset(token)

        payload = MCPAgentClientSession.get_url_elicitation_required_payload(result)
        assert payload is not None
        assert payload.server_name == "test-server"
        assert payload.request_method == "tools/call"
        assert len(payload.elicitations) == 1
        assert payload.elicitations[0].message == "Sign in to continue."
        assert payload.elicitations[0].url == "https://example.com/auth"
        assert payload.elicitations[0].elicitation_id == "normal-1"

    def test_queue_returns_false_without_active_request_context(self) -> None:
        session = self._make_session()

        queued = session.queue_url_elicitation_for_active_request(
            message="Sign in to continue.",
            url="https://example.com/auth",
            elicitation_id=None,
        )

        assert queued is False
