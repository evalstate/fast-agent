import httpx2
import pytest
from mcp.client.auth import OAuthFlowError, OAuthRegistrationError

from fast_agent.core.exceptions import ServerInitializationError
from fast_agent.mcp.failures import (
    classify_mcp_failure,
    redact_mcp_failure_text,
    render_mcp_failure,
)


def test_oauth_failure_uses_typed_cause_and_redacts_target() -> None:
    cause = OAuthRegistrationError("Registration failed")
    outer = ServerInitializationError(
        "MCP initialization failed",
        "Registration failed for https://user:pass@example.com/mcp?token=secret",
        server_name="docs",
    )
    outer.__cause__ = cause

    failure = classify_mcp_failure(
        outer,
        server_name="docs",
        origin="session",
        surface="acp_connect",
        input_ref="https://user:pass@example.com/mcp?token=secret",
    )

    assert failure.kind == "oauth_failed"
    assert failure.stage == "auth"
    assert failure.retry == "user_action"
    assert failure.cause is outer
    assert failure.input_ref.startswith("https://[REDACTED]@example.com/mcp")
    assert "secret" not in failure.input_ref
    assert failure.detail is not None
    assert "user:pass" not in failure.detail
    assert "token=secret" not in failure.detail
    rendered = render_mcp_failure(failure, output_format="markdown")
    assert "**Next:**" in rendered
    assert "Stop/Cancel" in rendered


def test_explicit_auth_rejection_does_not_offer_oauth_override() -> None:
    request = httpx2.Request(
        "POST",
        "https://user:pass@example.com/mcp?access_token=secret",
    )
    cause = httpx2.HTTPStatusError(
        "rejected",
        request=request,
        response=httpx2.Response(401, request=request),
    )

    failure = classify_mcp_failure(
        cause,
        server_name="private",
        origin="session",
        surface="terminal_connect",
        input_ref=str(request.url),
        explicit_auth=True,
    )

    assert failure.kind == "unauthorized"
    assert failure.stage == "auth"
    assert failure.remediation is not None
    assert "supplied credentials" in failure.remediation.casefold()
    assert "OAuth" not in failure.remediation
    assert "secret" not in render_mcp_failure(failure)


def test_oauth_failure_guidance_distinguishes_server_and_exact_endpoint() -> None:
    configured = classify_mcp_failure(
        OAuthFlowError("flow failed"),
        server_name="docs",
        origin="central",
        surface="configured_attach",
        input_ref="docs",
    )
    configured_connect = classify_mcp_failure(
        OAuthFlowError("flow failed"),
        server_name="docs",
        origin="central",
        surface="terminal_connect",
        input_ref="https://example.test/custom/mcp",
    )
    ad_hoc = classify_mcp_failure(
        OAuthFlowError("flow failed"),
        server_name="example",
        origin="session",
        surface="terminal_connect",
        input_ref="https://example.test/custom/mcp?token=secret",
    )

    assert configured.remediation is not None
    assert "auth mcp login docs" in configured.remediation
    assert configured_connect.remediation is not None
    assert "auth mcp login docs" in configured_connect.remediation
    assert ad_hoc.remediation is not None
    assert "auth mcp login --endpoint <exact-mcp-url>" in ad_hoc.remediation
    assert "secret" not in ad_hoc.remediation


@pytest.mark.parametrize(
    ("input_ref", "has_copilot_hint"),
    [
        ("https://githubcopilot.com/mcp/", True),
        ("https://api.githubcopilot.com/mcp/", True),
        ("https://preview.api.githubcopilot.com/mcp/", True),
        ("https://githubcopilot.com.attacker.example/mcp/", False),
        ("https://notgithubcopilot.com/mcp/", False),
        ("https://example.com/mcp?next=https://api.githubcopilot.com/mcp/", False),
        ("https://example.com/githubcopilot.com/mcp/", False),
        ("githubcopilot.com", False),
        ("https://[invalid", False),
    ],
)
def test_oauth_registration_copilot_guidance_requires_copilot_hostname(
    input_ref: str,
    has_copilot_hint: bool,
) -> None:
    failure = classify_mcp_failure(
        OAuthRegistrationError("Registration failed"),
        server_name="copilot",
        origin="session",
        surface="terminal_connect",
        input_ref=input_ref,
    )

    assert failure.remediation is not None
    assert ("GitHub Copilot MCP" in failure.remediation) is has_copilot_hint


def test_connection_failure_is_safe_to_retry_once() -> None:
    failure = classify_mcp_failure(
        ConnectionError("connection reset"),
        server_name="docs",
        origin="central",
        surface="configured_attach",
        input_ref="fast-agent.yaml",
        stage="discover",
    )

    assert failure.kind == "transport"
    assert failure.retry == "safe_once"
    assert failure.stage == "discover"


def test_failure_text_redacts_serialized_headers() -> None:
    redacted = redact_mcp_failure_text(
        'headers={"Authorization": "Bearer top-secret", "X-Api-Key": "also-secret"}'
    )

    assert "top-secret" not in redacted
    assert "also-secret" not in redacted
    assert redacted.count("[REDACTED]") == 2
