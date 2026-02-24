import shlex

from fast_agent.ui.command_payloads import (
    McpConnectCommand,
    McpDisconnectCommand,
    McpListCommand,
    McpSessionCommand,
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
        "/mcp connect --name docs --auth secret-token --timeout 7 --no-oauth --no-reconnect npx my-server"
    )
    assert isinstance(result, McpConnectCommand)
    assert result.server_name == "docs"
    assert result.auth_token == "secret-token"
    assert result.timeout_seconds == 7.0
    assert result.trigger_oauth is False
    assert result.reconnect_on_disconnect is False
    assert result.parsed_mode == "npx"
    assert result.error is None


def test_parse_mcp_connect_preserves_quoted_target_arguments() -> None:
    result = parse_special_input('/mcp connect demo-server --root "My Folder" --name docs')
    assert isinstance(result, McpConnectCommand)
    assert shlex.split(result.target_text) == ["demo-server", "--root", "My Folder"]
    assert result.server_name == "docs"


def test_parse_mcp_disconnect() -> None:
    result = parse_special_input("/mcp disconnect local")
    assert isinstance(result, McpDisconnectCommand)
    assert result.server_name == "local"
    assert result.error is None


def test_parse_connect_alias_detects_npx_for_scoped_package() -> None:
    result = parse_special_input("/connect @modelcontextprotocol/server-everything")
    assert isinstance(result, McpConnectCommand)
    assert result.parsed_mode == "npx"


def test_parse_mcp_session_defaults_to_list() -> None:
    result = parse_special_input("/mcp session")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "list"
    assert result.server_identity is None
    assert result.error is None


def test_parse_mcp_session_server_shortcut() -> None:
    result = parse_special_input("/mcp session demo-server")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "list"
    assert result.server_identity == "demo-server"
    assert result.error is None


def test_parse_mcp_session_new_with_title() -> None:
    result = parse_special_input('/mcp session new demo --title "Demo Run"')
    assert isinstance(result, McpSessionCommand)
    assert result.action == "new"
    assert result.server_identity == "demo"
    assert result.title == "Demo Run"
    assert result.error is None


def test_parse_mcp_session_resume() -> None:
    result = parse_special_input("/mcp session resume demo sess-123")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "use"
    assert result.server_identity == "demo"
    assert result.session_id == "sess-123"
    assert result.error is None


def test_parse_mcp_session_clear_all_without_args() -> None:
    result = parse_special_input("/mcp session clear")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "clear"
    assert result.clear_all is True
    assert result.server_identity is None
    assert result.error is None
