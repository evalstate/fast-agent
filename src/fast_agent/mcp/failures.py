"""Typed MCP failures at user-facing product boundaries."""

from __future__ import annotations

import asyncio
import re
import shlex
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import httpx2
from mcp.client.auth import OAuthFlowError, OAuthRegistrationError
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from fast_agent.core.exceptions import (
    FastAgentError,
    ServerInitializationError,
    walk_exception_chain,
)
from fast_agent.mcp.connect_targets import redact_mcp_url
from fast_agent.mcp.oauth_client import OAuthCallbackTimeoutError, OAuthFlowCancelledError

type MCPFailureOrigin = Literal["central", "card", "session"]
type MCPFailureSurface = Literal[
    "harness_startup",
    "configured_attach",
    "startup_url",
    "startup_stdio",
    "terminal_connect",
    "acp_connect",
]
type MCPFailureStage = Literal[
    "parse",
    "auth",
    "launch",
    "initialize",
    "discover",
    "operation",
    "reconnect",
    "shutdown",
]
type MCPFailureKind = Literal[
    "invalid_input",
    "unauthorized",
    "oauth_failed",
    "timeout",
    "protocol",
    "session_lost",
    "transport",
    "process",
    "server",
    "cancelled",
    "internal",
]
type MCPFailureRetry = Literal["never", "user_action", "safe_once"]
type MCPFailureFormat = Literal["terminal", "markdown", "cli"]

_GITHUB_COPILOT_HOST = "githubcopilot.com"
_URL_RE = re.compile(r"https?://[^\s<>`]+", re.IGNORECASE)
_QUOTED_OR_TOKEN = r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}\"']+)"
# Header-style credentials keyed by a well-known name. Bare words such as "code",
# "token" or "secret" are deliberately excluded: they destroy useful diagnostics
# (HTTP status codes, exit codes) for little gain.
_SECRET_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|api[_-]?key|client[_-]?secret|"
    r"(?:access|refresh|id)[_-]?token|password)\s*[\"']?\s*[:=]\s*"
    r"(?:(?:bearer|basic)\s+)?" + _QUOTED_OR_TOKEN
)
# Cookie values are opaque and multi-part; redact the rest of the line.
_COOKIE_RE = re.compile(r"(?i)\b((?:set-)?cookie)\s*[\"']?\s*[:=][^\n]*")
_SCHEME_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+" + _QUOTED_OR_TOKEN)
_AUTH_OPTION_RE = re.compile(r"(?i)(--auth(?:=|\s+))" + _QUOTED_OR_TOKEN)
# Terminal escape sequences: OSC (title/hyperlink) and CSI (colour/cursor) forms.
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True, slots=True)
class MCPFailure:
    server_name: str | None
    origin: MCPFailureOrigin
    surface: MCPFailureSurface
    input_ref: str
    stage: MCPFailureStage
    kind: MCPFailureKind
    summary: str
    detail: str | None
    retry: MCPFailureRetry
    remediation: str | None
    cause: BaseException = field(repr=False, compare=False)


def redact_mcp_failure_text(value: str, *, keep_url_path: bool = True) -> str:
    """Redact URLs and common credential forms in diagnostic text.

    Pass ``keep_url_path=False`` for untrusted text: hosted MCP endpoints often
    embed API keys in the URL path, so only the scheme and host are retained.
    """

    def redact_url(match: re.Match[str]) -> str:
        value = match.group(0)
        url = value.rstrip(".,);]\"'")
        trailing = value[len(url) :]
        if not keep_url_path:
            with suppress(ValueError):  # malformed URLs are fully redacted by redact_mcp_url
                parsed = urlsplit(url)
                path = "/[REDACTED]" if parsed.path.strip("/") else parsed.path
                url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        return f"{redact_mcp_url(url)}{trailing}"

    redacted = _URL_RE.sub(redact_url, value)
    redacted = _COOKIE_RE.sub(r"\1: [REDACTED]", redacted)
    redacted = _SECRET_RE.sub(r"\1: [REDACTED]", redacted)
    redacted = _SCHEME_RE.sub(r"\1 [REDACTED]", redacted)
    return _AUTH_OPTION_RE.sub(r"\1[REDACTED]", redacted)


def safe_mcp_diagnostic_text(value: str) -> str:
    """Make untrusted diagnostic text safe to render in a terminal.

    This is primarily about terminal safety (escape sequences, Rich markup, size),
    plus the same targeted credential redaction used for structured failures.
    Secret-bearing values should be removed at the source by type (see
    ``safe_mcp_exception_text``) rather than by guessing at free-text patterns.
    """
    value = _OSC_RE.sub("", value)
    value = _CSI_RE.sub("", value)
    value = "".join(c for c in value if c.isprintable() or c == "\n")
    value = redact_mcp_failure_text(value, keep_url_path=False)
    # Neutralize Rich/terminal markup even when a caller does not quote the text.
    value = value.replace("[", "［").replace("]", "］")
    return value if len(value) <= 2000 else value[:1997] + "..."


def safe_mcp_exception_text(error: BaseException) -> str:
    """Summarize a cycle-safe exception chain, redacting by exception type where possible."""
    parts: list[str] = []
    for index, cause in enumerate(walk_exception_chain(error)):
        if index == 8:
            parts.append("Further causes omitted.")
            break
        # SDK MCPError carries the protocol message/code as typed fields.
        if isinstance(cause, MCPError):
            detail = f"{cause.message} (MCP error {cause.code})"
        elif isinstance(cause, ValidationError):
            # Pydantic's default string includes raw configuration input values.
            detail = "; ".join(
                f"{item['type']}: {item['msg']}"
                for item in cause.errors(
                    include_input=False, include_context=False, include_url=False
                )
            )
        else:
            detail = str(cause)
        parts.append(safe_mcp_diagnostic_text(f"{type(cause).__name__}: {detail}"))
    text = "\nCaused by: ".join(parts)
    return text if len(text) <= 4000 else text[:3997] + "..."


def classify_mcp_failure(
    cause: BaseException,
    *,
    server_name: str | None,
    origin: MCPFailureOrigin,
    surface: MCPFailureSurface,
    input_ref: str,
    stage: MCPFailureStage = "initialize",
    explicit_auth: bool = False,
) -> MCPFailure:
    exceptions = list(walk_exception_chain(cause))
    selected = _select_typed_cause(exceptions)
    kind, resolved_stage = _failure_kind_and_stage(selected, stage)
    summary = _failure_summary(kind)
    detail = _failure_detail(cause, selected)
    remediation = _failure_remediation(
        kind,
        selected=selected,
        server_name=server_name,
        origin=origin,
        surface=surface,
        input_ref=input_ref,
        explicit_auth=explicit_auth,
    )
    return MCPFailure(
        server_name=server_name,
        origin=origin,
        surface=surface,
        input_ref=redact_mcp_failure_text(input_ref),
        stage=resolved_stage,
        kind=kind,
        summary=summary,
        detail=detail,
        retry=_failure_retry(kind),
        remediation=remediation,
        cause=cause,
    )


def _select_typed_cause(exceptions: list[BaseException]) -> BaseException:
    priorities = (
        asyncio.CancelledError,
        OAuthFlowCancelledError,
        OAuthCallbackTimeoutError,
        OAuthRegistrationError,
        OAuthFlowError,
        httpx2.HTTPStatusError,
        TimeoutError,
        MCPError,
        FileNotFoundError,
        PermissionError,
        ConnectionError,
        OSError,
        ValueError,
        ServerInitializationError,
    )
    for error_type in priorities:
        if selected := next((exc for exc in exceptions if isinstance(exc, error_type)), None):
            return selected
    return exceptions[0]


def _failure_kind_and_stage(
    cause: BaseException,
    default_stage: MCPFailureStage,
) -> tuple[MCPFailureKind, MCPFailureStage]:
    if isinstance(cause, (asyncio.CancelledError, OAuthFlowCancelledError)):
        return "cancelled", default_stage
    if isinstance(cause, OAuthCallbackTimeoutError):
        return "timeout", "auth"
    if isinstance(cause, (OAuthRegistrationError, OAuthFlowError)):
        return "oauth_failed", "auth"
    if isinstance(cause, httpx2.HTTPStatusError):
        if cause.response.status_code in {401, 403}:
            return "unauthorized", "auth"
        return ("server" if cause.response.status_code >= 500 else "transport"), default_stage
    if isinstance(cause, TimeoutError):
        return "timeout", default_stage
    if isinstance(cause, MCPError):
        if cause.code == -32600 and cause.message == "Session terminated":
            return "session_lost", default_stage
        return "protocol", default_stage
    if isinstance(cause, (FileNotFoundError, PermissionError)):
        return "process", "launch"
    if isinstance(cause, (ConnectionError, OSError)):
        return "transport", default_stage
    if isinstance(cause, ValueError):
        return "invalid_input", "parse"
    return "internal", default_stage


def _failure_summary(kind: MCPFailureKind) -> str:
    return {
        "invalid_input": "The MCP target or configuration is invalid.",
        "unauthorized": "The MCP server rejected authentication.",
        "oauth_failed": "MCP OAuth authorization failed.",
        "timeout": "The MCP server connection timed out.",
        "protocol": "MCP protocol negotiation or initialization failed.",
        "session_lost": "The MCP server session was lost.",
        "transport": "The MCP transport failed.",
        "process": "The MCP server process could not be started.",
        "server": "The MCP server returned an error.",
        "cancelled": "The MCP operation was cancelled.",
        "internal": "The MCP connection failed.",
    }[kind]


def _failure_detail(outer: BaseException, selected: BaseException) -> str | None:
    if isinstance(selected, httpx2.HTTPStatusError):
        response = selected.response
        return f"HTTP {response.status_code} {response.reason_phrase}".strip()
    if isinstance(selected, MCPError):
        code = f" ({selected.code})" if selected.code is not None else ""
        return redact_mcp_failure_text(f"{selected.message}{code}")
    if isinstance(outer, FastAgentError) and outer.details:
        return redact_mcp_failure_text(outer.details)
    detail = str(selected).strip()
    return redact_mcp_failure_text(detail) if detail else None


def _is_github_copilot_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    hostname = parsed.hostname
    return bool(
        parsed.scheme in {"http", "https"}
        and hostname
        and (hostname == _GITHUB_COPILOT_HOST or hostname.endswith(f".{_GITHUB_COPILOT_HOST}"))
    )


def _failure_remediation(
    kind: MCPFailureKind,
    *,
    selected: BaseException,
    server_name: str | None,
    origin: MCPFailureOrigin,
    surface: MCPFailureSurface,
    input_ref: str,
    explicit_auth: bool,
) -> str | None:
    if kind == "invalid_input":
        if surface in {"terminal_connect", "acp_connect"} and server_name:
            return (
                f"If '{server_name}' is configured, use `/mcp attach {server_name}`; "
                "otherwise choose `--name <different-name>` or correct the target."
            )
        return "Correct the target or configuration and retry."
    if kind == "unauthorized":
        if explicit_auth:
            return "The supplied credentials were rejected; verify or replace them before retrying."
        return "Authenticate with OAuth or supply valid bearer credentials, then retry."
    if kind == "oauth_failed":
        login_target = f"`fast-agent auth mcp login {server_name}`" if server_name else None
        try:
            first_input = shlex.split(input_ref)[0]
        except (IndexError, ValueError):
            first_input = input_ref
        if (
            origin != "central"
            and surface in {"startup_url", "terminal_connect", "acp_connect"}
            and "://" in first_input
        ):
            login_target = "`fast-agent auth mcp login --endpoint <exact-mcp-url>`"
        guidance = (
            f"Run {login_target} on the fast-agent host, then retry."
            if login_target
            else "Authenticate on the fast-agent host, then retry."
        )
        if isinstance(selected, OAuthRegistrationError):
            guidance = (
                "Configure `--client-metadata-url <https-url>` (CIMD) or use bearer "
                "authentication with `--auth <token>`, then retry."
            )
            if _is_github_copilot_url(input_ref):
                guidance += " GitHub Copilot MCP commonly requires `--auth $GITHUB_TOKEN`."
        if surface == "acp_connect":
            guidance += " Use the ACP client's Stop/Cancel action to cancel an in-flight attempt."
        return guidance
    if kind == "timeout":
        return "Verify server/network startup, increase `--timeout` if appropriate, and retry."
    if kind == "process":
        return "Verify the executable, arguments, permissions, and working directory."
    if kind == "session_lost":
        return "Reconnect once; replay only operations that are safe to repeat."
    if kind == "transport":
        return "Verify the server address and network or process transport, then retry."
    if kind == "protocol":
        return "Check server protocol compatibility and the configured protocol mode."
    if kind == "server":
        return "Check the MCP server logs and configuration before retrying."
    return None


def _failure_retry(kind: MCPFailureKind) -> MCPFailureRetry:
    if kind in {"unauthorized", "oauth_failed", "timeout", "process", "server"}:
        return "user_action"
    if kind in {"session_lost", "transport"}:
        return "safe_once"
    return "never"


def render_mcp_failure(
    failure: MCPFailure,
    *,
    output_format: MCPFailureFormat = "terminal",
) -> str:
    title = {
        "configured_attach": "Failed to attach configured MCP server",
        "harness_startup": "Failed to start configured MCP server",
        "startup_url": "Failed to connect startup MCP server",
        "startup_stdio": "Failed to start startup MCP server",
        "terminal_connect": "Failed to connect MCP server",
        "acp_connect": "Failed to connect MCP server",
    }[failure.surface]
    server = f" '{failure.server_name}'" if failure.server_name else ""
    first_line = f"{title}{server}: {failure.summary}"
    fields = [
        ("Stage", failure.stage),
        ("Target", failure.input_ref),
        ("Details", failure.detail),
        ("Next", failure.remediation),
    ]
    if output_format == "markdown":
        lines = [f"**{first_line}**"]
        lines.extend(f"**{label}:** {value}" for label, value in fields if value)
        return "\n\n".join(lines)
    lines = [first_line]
    lines.extend(f"{label}: {value}" for label, value in fields if value)
    return "\n".join(lines)
