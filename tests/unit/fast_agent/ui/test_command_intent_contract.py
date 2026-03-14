from __future__ import annotations

import pytest

from fast_agent.ui.command_payloads import (
    CommandPayload,
    HashAgentCommand,
    HistoryShowCommand,
    LoadHistoryCommand,
    McpConnectCommand,
    McpListCommand,
    McpSessionCommand,
    ShowHistoryCommand,
    UnknownCommand,
)
from fast_agent.ui.prompt import parse_special_input

type ExpectedParseResult = str | CommandPayload


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        pytest.param(
            "/history analyst",
            ShowHistoryCommand(agent="analyst"),
            id="history-bare-target",
        ),
        pytest.param(
            '/history "show"',
            ShowHistoryCommand(agent="show"),
            id="history-quoted-subcommand-collision",
        ),
        pytest.param(
            "/history show analyst",
            HistoryShowCommand(agent="analyst"),
            id="history-show-target",
        ),
        pytest.param(
            "/history load",
            LoadHistoryCommand(
                filename=None,
                error="Filename required for /history load",
            ),
            id="history-load-missing-filename",
        ),
        pytest.param(
            "/mcp list",
            McpListCommand(),
            id="mcp-list",
        ),
        pytest.param(
            "/mcp session use demo sess-123",
            McpSessionCommand(
                action="use",
                server_identity="demo",
                session_id="sess-123",
                title=None,
                clear_all=False,
                error=None,
            ),
            id="mcp-session-use",
        ),
        pytest.param(
            "/mcp session use demo",
            McpSessionCommand(
                action="use",
                server_identity=None,
                session_id=None,
                title=None,
                clear_all=False,
                error="Usage: /mcp session use <server_or_mcp_name> <session_id>",
            ),
            id="mcp-session-use-invalid-arity",
        ),
        pytest.param(
            "/connect https://example.com/mcp",
            McpConnectCommand(
                target_text="https://example.com/mcp",
                parsed_mode="url",
                server_name=None,
                auth_token=None,
                timeout_seconds=None,
                trigger_oauth=None,
                reconnect_on_disconnect=None,
                force_reconnect=False,
                error=None,
            ),
            id="connect-alias-url",
        ),
        pytest.param(
            "/connect @modelcontextprotocol/server-everything",
            McpConnectCommand(
                target_text="@modelcontextprotocol/server-everything",
                parsed_mode="npx",
                server_name=None,
                auth_token=None,
                timeout_seconds=None,
                trigger_oauth=None,
                reconnect_on_disconnect=None,
                force_reconnect=False,
                error=None,
            ),
            id="connect-alias-npx-scoped-package",
        ),
        pytest.param(
            "/connect uvx demo-server",
            McpConnectCommand(
                target_text="uvx demo-server",
                parsed_mode="uvx",
                server_name=None,
                auth_token=None,
                timeout_seconds=None,
                trigger_oauth=None,
                reconnect_on_disconnect=None,
                force_reconnect=False,
                error=None,
            ),
            id="connect-alias-uvx",
        ),
        pytest.param(
            "/connect python demo_server.py",
            McpConnectCommand(
                target_text="python demo_server.py",
                parsed_mode="stdio",
                server_name=None,
                auth_token=None,
                timeout_seconds=None,
                trigger_oauth=None,
                reconnect_on_disconnect=None,
                force_reconnect=False,
                error=None,
            ),
            id="connect-alias-stdio",
        ),
        pytest.param(
            "#review hello world",
            HashAgentCommand(agent_name="review", message="hello world", quiet=False),
            id="hash-agent-with-message",
        ),
        pytest.param(
            "#review",
            HashAgentCommand(agent_name="review", message="", quiet=False),
            id="hash-agent-without-message",
        ),
        pytest.param(
            "##review hello world",
            HashAgentCommand(agent_name="review", message="hello world", quiet=True),
            id="hash-agent-quiet",
        ),
        pytest.param(
            "/does-not-exist",
            UnknownCommand(command="/does-not-exist"),
            id="unknown-command-fallback",
        ),
    ],
)
def test_parse_special_input_intent_contract(
    raw_input: str,
    expected: ExpectedParseResult,
) -> None:
    assert parse_special_input(raw_input) == expected
