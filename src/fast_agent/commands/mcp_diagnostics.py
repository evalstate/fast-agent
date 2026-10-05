"""Read-only MCP startup diagnostics shared by all command surfaces."""

from fast_agent.mcp.failures import safe_mcp_diagnostic_text
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.utils.markdown import markdown_code_span


def diagnostics_usage_error(tokens: list[str]) -> str | None:
    if tokens[0].lower() == "auth":
        # `/mcp auth <server>` is a login; only the bare form renders diagnostics.
        return "Usage: /mcp auth" if len(tokens) != 1 else None
    if len(tokens) > 2 or (len(tokens) == 2 and not tokens[1].strip()):
        return "Usage: /mcp error [server]"
    return None


def render_mcp_diagnostics(agent: object, tokens: list[str]) -> str:
    if error := diagnostics_usage_error(tokens):
        return error
    auth = tokens[0].lower() == "auth"
    if not isinstance(agent, McpAgentProtocol):
        return "MCP diagnostics are unavailable for this agent."
    aggregator = agent.aggregator
    snapshots = aggregator.startup_status
    server = tokens[1] if len(tokens) == 2 else None
    if server is not None and not any(s.server_name == server for s in snapshots):
        return "Unknown startup server. Use /mcp list to discover servers for this agent."
    failures = aggregator.get_startup_errors(server)
    if auth:
        failures = tuple(s for s in failures if s.state == "auth")
    lines = ["MCP authentication diagnostics" if auth else "MCP startup errors"]
    if not failures:
        lines.append(
            "No authentication waits or failures recorded."
            if auth
            else "No startup errors recorded."
        )
    for status in failures:
        name = markdown_code_span(safe_mcp_diagnostic_text(status.server_name))
        lines.append(f"\nServer {name} — {status.state}")
        lines.append(
            f"Phase: {status.phase}; transport: {status.transport or 'unknown'}; "
            f"duration: {status.duration_seconds:.2f}s; "
            f"recorded: {status.timestamp.isoformat() if status.timestamp else 'unknown'}"
        )
        detail = status.failure_detail or "No failure detail was recorded."
        # Quote external text rather than interpreting it as Markdown or terminal control codes.
        safe = safe_mcp_diagnostic_text(detail)
        lines.extend(markdown_code_span(line) for line in safe.splitlines())
        if status.state == "auth":
            lines.append(
                "Authentication is required: run "
                + markdown_code_span(f"/mcp auth {safe_mcp_diagnostic_text(status.server_name)}")
                + " to log in and connect (add `--device` to log in without a browser here)."
            )
        else:
            lines.append(
                "Check the configured executable, environment, endpoint, permissions and network "
                "against the failure above. After correcting the cause, use /mcp attach <server> "
                "for a failed startup attachment; use /mcp reconnect <server> if already attached."
            )
    for status in aggregator.startup_history:
        if status.resolved_at is not None and (server is None or status.server_name == server):
            if auth and status.state != "auth":
                continue
            lines.append(
                f"Resolved {markdown_code_span(safe_mcp_diagnostic_text(status.server_name))} "
                f"({status.phase}, {status.duration_seconds:.2f}s) at "
                f"{status.resolved_at.isoformat()}: "
                f"{markdown_code_span(safe_mcp_diagnostic_text(status.failure_detail or ''))}"
            )
    pending = sum(s.state == "pending" for s in snapshots)
    if pending:
        lines.append(
            f"{pending} server(s) still starting; run diagnostics again after startup settles."
        )
    if auth or any(s.state == "auth" for s in failures):
        lines.append(
            "Log in from this session with `/mcp auth <server>` (`--device` for a code "
            "you can enter on another device). Inspect configured authentication with "
            "`fast-agent auth mcp show <server>` or stored OAuth resources with "
            "`fast-agent auth mcp credentials`. For token-based authentication, correct the "
            "configured credential outside chat. This command does not expose credentials. "
            "Auth failures may be recorded as errors; also check `/mcp error`."
        )
    return "\n\n".join(lines)
