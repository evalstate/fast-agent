from fast_agent.ui.command_payloads import (
    McpConnectCommand,
    McpDisconnectCommand,
    McpListCommand,
    ShowMcpStatusCommand,
)
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_mcp_status_backwards_compatible() -> None:
    result = parse_special_input("/mcp")
    assert isinstance(result, ShowMcpStatusCommand)


def test_parse_mcp_list_command() -> None:
    result = parse_special_input("/mcp list")
    assert isinstance(result, McpListCommand)


def test_parse_connect_alias_detects_url_mode() -> None:
    result = parse_special_input("/connect https://example.com/mcp")
    assert isinstance(result, McpConnectCommand)
    assert result.parsed_mode == "url"
    assert result.target_text == "https://example.com/mcp"


def test_parse_mcp_connect_extracts_flags() -> None:
    result = parse_special_input(
        "/mcp connect --name docs --timeout 7 --no-oauth --no-reconnect npx my-server"
    )
    assert isinstance(result, McpConnectCommand)
    assert result.server_name == "docs"
    assert result.timeout_seconds == 7.0
    assert result.trigger_oauth is False
    assert result.reconnect_on_disconnect is False
    assert result.parsed_mode == "npx"
    assert result.error is None


def test_parse_mcp_disconnect() -> None:
    result = parse_special_input("/mcp disconnect local")
    assert isinstance(result, McpDisconnectCommand)
    assert result.server_name == "local"
    assert result.error is None


def test_parse_connect_alias_detects_npx_for_scoped_package() -> None:
    result = parse_special_input("/connect @modelcontextprotocol/server-everything")
    assert isinstance(result, McpConnectCommand)
    assert result.parsed_mode == "npx"
