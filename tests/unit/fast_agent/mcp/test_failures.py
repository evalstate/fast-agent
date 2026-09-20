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


@pytest.mark.parametrize(
    "detail",
    [
        "https://user:sensitive@[broken/mcp?token=sensitive#sensitive",
        "https://user:sensitive@host:invalid/private/sensitive?flag#sensitive",
        'Authorization: "Basic sensitive value"',
        "Authorization: Bearer sensitive",
        "Set-Cookie: session=sensitive; extra=sensitive",
        'access_token="sensitive value"',
        "refresh-token='sensitive value'",
        "--auth 'sensitive value'",
        '--auth="sensitive value"',
        "Basic sensitive",
        "Bearer 'sensitive value'",
        "https://user:'sensitive'@[broken",
        "api_key=sen\x1b[31msitive",
    ],
)
def test_shared_diagnostic_redaction(detail: str) -> None:
    from fast_agent.mcp.failures import safe_mcp_diagnostic_text

    text = safe_mcp_diagnostic_text("Launch failed: missing module\n" + detail)
    assert "missing module" in text
    assert "sensitive" not in text
    assert "value" not in text


def test_diagnostic_chain_is_bounded_cycle_safe_and_terminal_safe() -> None:
    from fast_agent.mcp.failures import safe_mcp_exception_text

    cause = ValueError("missing module\x1b[31m[bold]\r\x00\u202e")
    outer = ServerInitializationError(
        "Launch failed", "Recent stderr from stdio server:\nmodule unavailable", server_name="test"
    )
    outer.__cause__ = cause
    cause.__context__ = outer
    text = safe_mcp_exception_text(outer)
    assert "ServerInitializationError" in text
    assert "ValueError: missing module" in text
    assert "Recent stderr" in text
    assert "module unavailable" in text
    assert "[bold]" not in text
    assert all(c.isprintable() or c == "\n" for c in text)
    group = ExceptionGroup("many", [ValueError("x" * 10000) for _ in range(20)])
    assert len(safe_mcp_exception_text(group)) <= 4000


def test_sdk_error_retains_code_not_untrusted_data() -> None:
    from mcp.shared.exceptions import MCPError

    from fast_agent.mcp.failures import safe_mcp_exception_text

    error = MCPError(
        code=-32602, message="Invalid discovery response", data={"private": "sensitive"}
    )
    text = safe_mcp_exception_text(error)
    assert "Invalid discovery response" in text
    assert "-32602" in text
    assert "sensitive" not in text


def test_validation_error_omits_configuration_input() -> None:
    from pydantic import TypeAdapter, ValidationError

    from fast_agent.mcp.failures import safe_mcp_exception_text

    with pytest.raises(ValidationError) as raised:
        TypeAdapter(int).validate_python({"environment": "unlabeled-sensitive-value"})
    text = safe_mcp_exception_text(raised.value)
    assert "ValidationError" in text
    assert "int_type" in text
    assert "unlabeled-sensitive-value" not in text


def test_safe_diagnostic_text_preserves_status_and_exit_codes() -> None:
    from fast_agent.mcp.failures import safe_mcp_diagnostic_text

    assert "404" in safe_mcp_diagnostic_text("Server returned status code: 404")
    assert "exit code: 1" in safe_mcp_diagnostic_text("exit code: 1 while spawning npx")


def test_safe_diagnostic_text_redacts_credentials_and_neutralizes_terminal_output() -> None:
    from fast_agent.mcp.failures import safe_mcp_diagnostic_text

    text = safe_mcp_diagnostic_text(
        "\x1b[31mAuthorization: Bearer abc123\x1b[0m [bold]https://user:pw@h.example/p?tok=x"
    )
    assert "abc123" not in text
    assert "pw" not in text
    assert "tok=x" not in text
    assert "h.example" in text
    assert "\x1b" not in text
    assert "[bold]" not in text
