from fast_agent.commands.mcp_command_intents import (
    MCP_SESSION_ACTION_DESCRIPTIONS,
    MCP_SESSION_CLEAR_ACTION,
    MCP_SESSION_NEW_ACTIONS,
    MCP_SESSION_SERVER_SCOPED_ACTIONS,
    MCP_SESSION_USAGE,
    MCP_SESSION_USE_ACTIONS,
    MCP_TOP_LEVEL_ACTION_DESCRIPTIONS,
    MCP_TOP_LEVEL_ACTIONS,
    McpSessionIntent,
    parse_mcp_server_name_tokens,
    parse_mcp_session_tokens,
)


def test_mcp_completion_descriptions_cover_top_level_actions() -> None:
    assert tuple(MCP_TOP_LEVEL_ACTION_DESCRIPTIONS) == MCP_TOP_LEVEL_ACTIONS


def test_mcp_session_completion_descriptions_cover_parser_actions_and_resume_alias() -> None:
    for action in ("list", "jar", "new", "create", "use", "clear", "resume"):
        assert action in MCP_SESSION_ACTION_DESCRIPTIONS


def test_mcp_session_completion_groups_cover_described_actions() -> None:
    grouped_actions = (
        *MCP_SESSION_SERVER_SCOPED_ACTIONS,
        *MCP_SESSION_USE_ACTIONS,
        MCP_SESSION_CLEAR_ACTION,
    )

    assert set(grouped_actions) == set(MCP_SESSION_ACTION_DESCRIPTIONS)
    assert "resume" in MCP_SESSION_USE_ACTIONS
    assert set(MCP_SESSION_NEW_ACTIONS) <= set(MCP_SESSION_SERVER_SCOPED_ACTIONS)


def test_mcp_session_shortcut_preserves_server_identity_case() -> None:
    intent = parse_mcp_session_tokens(["MyServer"])

    assert intent == McpSessionIntent(
        action="list",
        server_identity="MyServer",
        session_id=None,
        title=None,
        clear_all=False,
        error=None,
    )


def test_mcp_session_action_normalization_preserves_identity_case() -> None:
    intent = parse_mcp_session_tokens([" LIST ", "  MyServer  "])

    assert intent.action == "list"
    assert intent.server_identity == "MyServer"
    assert intent.error is None


def test_mcp_session_shortcut_strips_outer_whitespace_only() -> None:
    intent = parse_mcp_session_tokens(["  My Server  "])

    assert intent.server_identity == "My Server"
    assert intent.error is None


def test_mcp_session_shortcut_rejects_empty_server_identity() -> None:
    intent = parse_mcp_session_tokens([""])

    assert intent.action == "list"
    assert intent.server_identity is None
    assert intent.error == MCP_SESSION_USAGE


def test_mcp_session_shortcut_rejects_whitespace_server_identity() -> None:
    intent = parse_mcp_session_tokens(["   "])

    assert intent.action == "list"
    assert intent.server_identity is None
    assert intent.error == MCP_SESSION_USAGE


def test_mcp_session_shortcut_rejects_long_option_tokens() -> None:
    for token in ("--all", "--title"):
        intent = parse_mcp_session_tokens([token])

        assert intent.action == "list"
        assert intent.server_identity is None
        assert intent.error == f"Unknown flag: {token}"


def test_mcp_server_name_rejects_whitespace_name() -> None:
    intent = parse_mcp_server_name_tokens(["disconnect", "   "], usage="usage")

    assert intent.server_name is None
    assert intent.error == "usage"


def test_mcp_server_name_strips_outer_whitespace_only() -> None:
    intent = parse_mcp_server_name_tokens(["disconnect", "  My Server  "], usage="usage")

    assert intent.server_name == "My Server"
    assert intent.error is None


def test_mcp_session_use_requires_server_and_session_id() -> None:
    intent = parse_mcp_session_tokens(["use", "demo"])

    assert intent.action == "use"
    assert intent.server_identity is None
    assert intent.session_id is None
    assert intent.error == "Usage: /mcp session use <server_or_mcp_name> <session_id>"


def test_mcp_session_use_rejects_empty_server_or_session_id() -> None:
    for args in (
        ["use", "", "session-1"],
        ["use", "demo", ""],
        ["use", "   ", "session-1"],
        ["use", "demo", "   "],
    ):
        intent = parse_mcp_session_tokens(args)

        assert intent.action == "use"
        assert intent.server_identity is None
        assert intent.session_id is None
        assert intent.error == "Usage: /mcp session use <server_or_mcp_name> <session_id>"


def test_mcp_session_use_strips_outer_whitespace() -> None:
    intent = parse_mcp_session_tokens(["use", "  My Server  ", "  session-1  "])

    assert intent.server_identity == "My Server"
    assert intent.session_id == "session-1"
    assert intent.error is None


def test_mcp_session_list_and_jar_reject_extra_args_with_action_usage() -> None:
    list_intent = parse_mcp_session_tokens(["list", "one", "two"])
    jar_intent = parse_mcp_session_tokens(["jar", "one", "two"])

    assert list_intent.action == "list"
    assert list_intent.error == "Usage: /mcp session list [<server_or_mcp_name>]"
    assert jar_intent.action == "jar"
    assert jar_intent.error == "Usage: /mcp session jar [<server_or_mcp_name>]"


def test_mcp_session_list_and_jar_reject_whitespace_server_identity() -> None:
    list_intent = parse_mcp_session_tokens(["list", "   "])
    jar_intent = parse_mcp_session_tokens(["jar", "   "])

    assert list_intent.action == "list"
    assert list_intent.error == "Usage: /mcp session list [<server_or_mcp_name>]"
    assert jar_intent.action == "jar"
    assert jar_intent.error == "Usage: /mcp session jar [<server_or_mcp_name>]"


def test_mcp_session_new_rejects_empty_title_value() -> None:
    intent = parse_mcp_session_tokens(["new", "demo", "--title", ""])

    assert intent.action == "new"
    assert intent.title is None
    assert intent.error == "Missing value for --title"


def test_mcp_session_new_rejects_whitespace_server_identity() -> None:
    intent = parse_mcp_session_tokens(["new", "   "])

    assert intent.action == "new"
    assert intent.server_identity is None
    assert intent.error == MCP_SESSION_USAGE


def test_mcp_session_new_accepts_title_value_forms() -> None:
    split_value = parse_mcp_session_tokens(["new", "demo", "--title", "-draft"])
    equals_value = parse_mcp_session_tokens(["new", "demo", "--title=planning"])
    create_alias = parse_mcp_session_tokens(["create", "demo", "--title", "Planning"])

    assert split_value.title == "-draft"
    assert split_value.error is None
    assert equals_value.title == "planning"
    assert equals_value.error is None
    assert create_alias.action == "new"
    assert create_alias.server_identity == "demo"
    assert create_alias.title == "Planning"
    assert create_alias.error is None


def test_mcp_session_new_rejects_duplicate_title_flags() -> None:
    split_duplicate = parse_mcp_session_tokens(
        ["new", "demo", "--title", "First", "--title", "Second"]
    )
    equals_duplicate = parse_mcp_session_tokens(
        ["new", "demo", "--title=First", "--title=Second"]
    )

    assert split_duplicate.title == "First"
    assert split_duplicate.error == "Duplicate flag: --title"
    assert equals_duplicate.title == "First"
    assert equals_duplicate.error == "Duplicate flag: --title"


def test_mcp_session_new_rejects_unknown_and_extra_arguments() -> None:
    unknown = parse_mcp_session_tokens(["new", "demo", "--unknown"])
    extra = parse_mcp_session_tokens(["new", "demo", "extra"])

    assert unknown.action == "new"
    assert unknown.server_identity == "demo"
    assert unknown.error == "Unknown flag: --unknown"
    assert extra.action == "new"
    assert extra.server_identity == "demo"
    assert extra.error == "Unexpected argument: extra"


def test_mcp_session_clear_requires_target_or_all_flag() -> None:
    intent = parse_mcp_session_tokens(["clear"])

    assert intent.action == "clear"
    assert intent.server_identity is None
    assert intent.clear_all is False
    assert intent.error == "Usage: /mcp session clear <server|--all>"


def test_mcp_session_clear_all_requires_all_flag() -> None:
    intent = parse_mcp_session_tokens(["clear", "--all"])

    assert intent == McpSessionIntent(
        action="clear",
        server_identity=None,
        session_id=None,
        title=None,
        clear_all=True,
        error=None,
    )


def test_mcp_session_clear_rejects_empty_server_identity() -> None:
    intent = parse_mcp_session_tokens(["clear", ""])

    assert intent.action == "clear"
    assert intent.error == "Usage: /mcp session clear <server|--all>"


def test_mcp_session_clear_rejects_whitespace_server_identity() -> None:
    intent = parse_mcp_session_tokens(["clear", "   "])

    assert intent.action == "clear"
    assert intent.error == "Usage: /mcp session clear <server|--all>"


def test_mcp_session_clear_rejects_duplicate_all_flag() -> None:
    intent = parse_mcp_session_tokens(["clear", "--all", "--all"])

    assert intent.action == "clear"
    assert intent.clear_all is True
    assert intent.error == "Duplicate flag: --all"


def test_mcp_session_clear_rejects_all_with_server_identity() -> None:
    intent = parse_mcp_session_tokens(["clear", "demo", "--all"])

    assert intent.action == "clear"
    assert intent.server_identity == "demo"
    assert intent.clear_all is True
    assert intent.error == "Use either a server name or --all"
