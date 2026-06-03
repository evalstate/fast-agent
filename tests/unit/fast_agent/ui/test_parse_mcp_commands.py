import shlex

from fast_agent.commands.mcp_command_intents import MCP_SERVER_NAME_ACTIONS, MCP_TOP_LEVEL_ACTIONS
from fast_agent.ui.command_payloads import (
    CommandError,
    McpConnectCommand,
    McpDisconnectCommand,
    McpListCommand,
    McpReconnectCommand,
    McpSessionCommand,
    ShowMcpStatusCommand,
    UnknownCommand,
)
from fast_agent.ui.enhanced_prompt import parse_special_input
from fast_agent.ui.prompt import parser as prompt_parser


def test_mcp_server_command_table_covers_name_based_commands() -> None:
    assert set(prompt_parser._MCP_SERVER_COMMAND_TYPES) == set(MCP_SERVER_NAME_ACTIONS)
    assert prompt_parser._MCP_SERVER_COMMAND_TYPES["disconnect"] is McpDisconnectCommand
    assert prompt_parser._MCP_SERVER_COMMAND_TYPES["reconnect"] is McpReconnectCommand


def test_mcp_token_parser_table_covers_non_connect_top_level_actions() -> None:
    assert set(prompt_parser._MCP_TOKEN_PARSERS) | {"connect"} == set(MCP_TOP_LEVEL_ACTIONS)


def test_parse_mcp_status_backwards_compatible() -> None:
    result = parse_special_input("/mcp")
    assert isinstance(result, ShowMcpStatusCommand)


def test_parse_mcp_list() -> None:
    result = parse_special_input("/mcp list")
    assert isinstance(result, McpListCommand)


def test_parse_mcp_list_matches_case_insensitively() -> None:
    result = parse_special_input("/MCP LIST")
    assert isinstance(result, McpListCommand)


def test_parse_mcp_list_rejects_extra_args() -> None:
    result = parse_special_input("/mcp list demo")
    assert isinstance(result, CommandError)
    assert result.message == "Usage: /mcp list"


def test_parse_mcp_list_rejects_invalid_quoting_as_command_error() -> None:
    result = parse_special_input('/mcp list "unterminated')
    assert isinstance(result, CommandError)
    assert "Invalid arguments:" in result.message


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


def test_parse_mcp_connect_preserves_unresolved_auth_reference() -> None:
    result = parse_special_input("/mcp connect https://example.com/mcp --auth $DOCS_TOKEN")
    assert isinstance(result, McpConnectCommand)
    assert result.auth_token == "$DOCS_TOKEN"
    assert result.error is None


def test_parse_mcp_connect_preserves_quoted_target_arguments() -> None:
    result = parse_special_input('/mcp connect --name docs demo-server --root "My Folder"')
    assert isinstance(result, McpConnectCommand)
    assert shlex.split(result.target_text) == ["demo-server", "--root", "My Folder"]
    assert result.server_name == "docs"


def test_parse_mcp_connect_accepts_documented_trailing_npx_flags() -> None:
    result = parse_special_input("/mcp connect npx demo-server --name docs --timeout 7")
    assert isinstance(result, McpConnectCommand)
    assert result.server_name == "docs"
    assert result.timeout_seconds == 7.0
    assert result.request is not None
    assert result.request.target.args == ("demo-server",)


def test_parse_mcp_connect_preserves_quoted_windows_path() -> None:
    result = parse_special_input('/mcp connect "C:\\Program Files\\Tool\\tool.exe" --flag')
    assert isinstance(result, McpConnectCommand)
    assert result.request is not None
    assert result.request.target.command == "C:\\Program Files\\Tool\\tool.exe"
    assert result.request.target.args == ("--flag",)


def test_connect_alias_matches_mcp_connect() -> None:
    alias = parse_special_input('/connect --name docs demo-server --root "My Folder"')
    explicit = parse_special_input('/mcp connect --name docs demo-server --root "My Folder"')
    assert isinstance(alias, McpConnectCommand)
    assert isinstance(explicit, McpConnectCommand)
    assert alias.request == explicit.request


def test_parse_mcp_disconnect() -> None:
    result = parse_special_input("/mcp disconnect local")
    assert isinstance(result, McpDisconnectCommand)
    assert result.server_name == "local"
    assert result.error is None


def test_parse_mcp_disconnect_matches_case_insensitively() -> None:
    result = parse_special_input("/MCP DISCONNECT local")
    assert isinstance(result, McpDisconnectCommand)
    assert result.server_name == "local"
    assert result.error is None


def test_parse_mcp_disconnect_rejects_extra_args() -> None:
    result = parse_special_input("/mcp disconnect local extra")
    assert isinstance(result, McpDisconnectCommand)
    assert result.server_name is None
    assert result.error == "Usage: /mcp disconnect <server_name>"


def test_parse_mcp_disconnect_rejects_empty_server_name() -> None:
    result = parse_special_input('/mcp disconnect ""')
    assert isinstance(result, McpDisconnectCommand)
    assert result.server_name is None
    assert result.error == "Usage: /mcp disconnect <server_name>"


def test_parse_mcp_reconnect_rejects_empty_server_name() -> None:
    result = parse_special_input('/mcp reconnect ""')
    assert isinstance(result, McpReconnectCommand)
    assert result.server_name is None
    assert result.error == "Usage: /mcp reconnect <server_name>"


def test_parse_mcp_reconnect() -> None:
    result = parse_special_input("/mcp reconnect local")
    assert isinstance(result, McpReconnectCommand)
    assert result.server_name == "local"
    assert result.error is None


def test_parse_mcp_reconnect_rejects_extra_args() -> None:
    result = parse_special_input("/mcp reconnect local extra")
    assert isinstance(result, McpReconnectCommand)
    assert result.server_name is None
    assert result.error == "Usage: /mcp reconnect <server_name>"


def test_parse_mcp_session_server_shortcut() -> None:
    result = parse_special_input("/mcp session demo-server")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "list"
    assert result.server_identity == "demo-server"
    assert result.error is None


def test_parse_mcp_session_server_shortcut_preserves_case() -> None:
    result = parse_special_input("/mcp session MyServer")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "list"
    assert result.server_identity == "MyServer"
    assert result.error is None


def test_parse_mcp_session_new_with_title() -> None:
    result = parse_special_input('/mcp session new demo --title "Demo Run"')
    assert isinstance(result, McpSessionCommand)
    assert result.action == "new"
    assert result.server_identity == "demo"
    assert result.title == "Demo Run"
    assert result.error is None


def test_parse_mcp_session_invalid_arguments_stays_session_command() -> None:
    result = parse_special_input('/mcp session "unterminated')
    assert isinstance(result, McpSessionCommand)
    assert result.error is not None
    assert "Invalid arguments:" in result.error


def test_parse_unknown_mcp_subcommand_preserves_context() -> None:
    result = parse_special_input("/mcp frob target")
    assert result == UnknownCommand(command="/mcp frob target")


def test_parse_unknown_mcp_subcommand_invalid_quoting_is_not_connect() -> None:
    result = parse_special_input('/mcp frob "unterminated')
    assert isinstance(result, CommandError)
    assert "Invalid arguments:" in result.message


def test_parse_mcp_session_resume() -> None:
    result = parse_special_input("/mcp session resume demo sess-123")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "use"
    assert result.server_identity == "demo"
    assert result.session_id == "sess-123"
    assert result.error is None


def test_parse_mcp_session_clear_requires_target_or_all_flag() -> None:
    result = parse_special_input("/mcp session clear")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "clear"
    assert result.clear_all is False
    assert result.server_identity is None
    assert result.error == "Usage: /mcp session clear <server|--all>"


def test_parse_mcp_session_clear_all_with_explicit_flag() -> None:
    result = parse_special_input("/mcp session clear --all")
    assert isinstance(result, McpSessionCommand)
    assert result.action == "clear"
    assert result.clear_all is True
    assert result.server_identity is None
    assert result.error is None
