from fast_agent.commands.mcp_command_intents import (
    MCP_TOP_LEVEL_ACTION_DESCRIPTIONS,
    MCP_TOP_LEVEL_ACTIONS,
    parse_mcp_server_name_tokens,
)


def test_mcp_completion_descriptions_cover_top_level_actions() -> None:
    assert tuple(MCP_TOP_LEVEL_ACTION_DESCRIPTIONS) == MCP_TOP_LEVEL_ACTIONS


def test_mcp_server_name_rejects_whitespace_name() -> None:
    intent = parse_mcp_server_name_tokens(["disconnect", "   "], usage="usage")

    assert intent.server_name is None
    assert intent.error == "usage"


def test_mcp_server_name_strips_outer_whitespace_only() -> None:
    intent = parse_mcp_server_name_tokens(["disconnect", "  My Server  "], usage="usage")

    assert intent.server_name == "My Server"
    assert intent.error is None
