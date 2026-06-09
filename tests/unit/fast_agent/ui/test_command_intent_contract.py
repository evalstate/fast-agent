from __future__ import annotations

import pytest

from fast_agent.commands.shared_command_intents import (
    SESSION_COMMAND_COMPLETION_DESCRIPTIONS,
    SESSION_SIMPLE_PAYLOAD_ACTIONS,
)
from fast_agent.ui.command_payloads import (
    AgentCommand,
    AttachCommand,
    CardsCommand,
    CheckCommand,
    ClearSessionsCommand,
    CommandError,
    CommandPayload,
    CreateSessionCommand,
    ForkSessionCommand,
    HashAgentCommand,
    HistoryViewCommand,
    ListPromptsCommand,
    LoadAgentCardCommand,
    LoadHistoryCommand,
    LoadPromptCommand,
    McpConnectCommand,
    McpListCommand,
    ResumeSessionCommand,
    SaveHistoryCommand,
    TitleSessionCommand,
    UnknownCommand,
)
from fast_agent.ui.prompt import parse_special_input
from fast_agent.ui.prompt import parser as prompt_parser

type ExpectedParseResult = str | CommandPayload | dict[str, object]


def test_session_payload_factory_table_matches_shared_simple_actions() -> None:
    assert frozenset(prompt_parser._SESSION_PAYLOAD_FACTORIES) == SESSION_SIMPLE_PAYLOAD_ACTIONS


def test_history_turn_error_formatters_cover_shared_error_codes() -> None:
    assert frozenset(prompt_parser._HISTORY_TURN_ERROR_FORMATTERS) == {"missing", "invalid"}


def test_session_completion_descriptions_cover_parser_actions() -> None:
    assert set(SESSION_COMMAND_COMPLETION_DESCRIPTIONS) == {
        "list",
        "new",
        "resume",
        "title",
        "fork",
        "delete",
        "clear",
        "pin",
        "export",
    }


def test_slash_parser_static_dispatch_tables_cover_expected_commands() -> None:
    assert frozenset(prompt_parser._SIMPLE_SLASH_FACTORIES) == {
        "help",
        "system",
        "usage",
        "markdown",
        "reload",
        "mcpstatus",
        "tools",
        "prompts",
        "exit",
        "stop",
    }
    assert frozenset(prompt_parser._COMMAND_PARSERS) == {
        "history",
        "session",
        "card",
        "agent",
        "a2a",
        "tasks",
        "mcp",
        "connect",
        "prompt",
        "model",
        "models",
        "attach",
        "check",
        "commands",
    }
    assert frozenset(prompt_parser._SLASH_ACTION_FACTORIES) == {
        "skills",
        "cards",
        "plugins",
    }
    assert frozenset(prompt_parser._SLASH_ALIAS_PARSERS) == {
        "save_history",
        "save",
        "load_history",
        "load",
        "resume",
        "fast",
    }
    assert frozenset(prompt_parser._PROMPT_SUBCOMMAND_PARSERS) == {
        "load",
    }


@pytest.mark.parametrize(
    ("raw_input", "expected"),
    [
        pytest.param(
            "/attach",
            AttachCommand(paths=(), clear=False, error=None),
            id="attach-open-prompt",
        ),
        pytest.param(
            '/attach "./report one.pdf" ../two.png',
            AttachCommand(paths=("./report one.pdf", "../two.png"), clear=False, error=None),
            id="attach-paths",
        ),
        pytest.param(
            "/attach CLEAR",
            AttachCommand(paths=(), clear=True, error=None),
            id="attach-clear",
        ),
        pytest.param(
            "/history analyst",
            HistoryViewCommand(agent="analyst"),
            id="history-bare-target",
        ),
        pytest.param(
            '/history "show"',
            HistoryViewCommand(agent="show"),
            id="history-quoted-subcommand-collision",
        ),
        pytest.param(
            "/history show analyst",
            HistoryViewCommand(agent="analyst", view="table"),
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
            '/prompt load "my prompt.md"',
            LoadPromptCommand(filename="my prompt.md", error=None),
            id="prompt-load-quoted-path",
        ),
        pytest.param(
            "/prompt Example.JSON",
            LoadPromptCommand(filename="Example.JSON", error=None),
            id="prompt-bare-uppercase-json-path",
        ),
        pytest.param(
            '/prompt load "unterminated',
            CommandError(message="Invalid /prompt arguments: No closing quotation"),
            id="prompt-load-unterminated-quote",
        ),
        pytest.param(
            "/mcp list",
            McpListCommand(),
            id="mcp-list",
        ),
        pytest.param(
            "/card card.yml extra",
            LoadAgentCardCommand(
                filename="card.yml",
                add_tool=False,
                remove_tool=False,
                error="Unexpected arguments: extra",
            ),
            id="card-rejects-extra-args",
        ),
        pytest.param(
            "/agent @alpha --tool --rm",
            AgentCommand(
                agent_name="alpha",
                add_tool=True,
                remove_tool=True,
                dump=False,
                error=None,
            ),
            id="agent-tool-remove-alias",
        ),
        pytest.param(
            "/session new review",
            CreateSessionCommand(session_name="review"),
            id="session-new",
        ),
        pytest.param(
            "/session resume sess-123",
            ResumeSessionCommand(session_id="sess-123"),
            id="session-resume",
        ),
        pytest.param(
            '/session "resume" sess-123',
            ResumeSessionCommand(session_id="sess-123"),
            id="session-quoted-resume",
        ),
        pytest.param(
            '/resume "sess 123"',
            ResumeSessionCommand(session_id="sess 123"),
            id="resume-alias-quoted-id",
        ),
        pytest.param(
            "/resume sess 123",
            ResumeSessionCommand(session_id="sess 123"),
            id="resume-alias-unquoted-multi-token-id",
        ),
        pytest.param(
            "/session title Sprint Notes",
            TitleSessionCommand(title="Sprint Notes"),
            id="session-title",
        ),
        pytest.param(
            '/session title "unterminated',
            CommandError(message="Invalid /session arguments: No closing quotation"),
            id="session-title-unterminated",
        ),
        pytest.param(
            "/session fork forked run",
            ForkSessionCommand(title="forked run"),
            id="session-fork",
        ),
        pytest.param(
            "/session delete all",
            ClearSessionsCommand(target="all"),
            id="session-delete",
        ),
        pytest.param(
            "/save notes.md",
            SaveHistoryCommand(filename="notes.md"),
            id="save-alias",
        ),
        pytest.param(
            "/load",
            LoadHistoryCommand(filename=None, error="Filename required for /history load"),
            id="load-alias-missing-filename",
        ),
        pytest.param(
            "/cards registry",
            CardsCommand(action="registry", argument=None),
            id="cards-action-alias",
        ),
        pytest.param(
            "/prompts",
            ListPromptsCommand(),
            id="prompts-list-alias",
        ),
        pytest.param(
            "/tools extra",
            CommandError(message="Unexpected arguments for /tools: extra"),
            id="simple-command-rejects-extra-args",
        ),
        pytest.param(
            "/check models --for-model gpt-5",
            CheckCommand(argument="models --for-model gpt-5"),
            id="check-command",
        ),
        pytest.param(
            "/connect https://example.com/mcp",
            {
                "kind": "mcp_connect",
                "target_text": "https://example.com/mcp",
                "parsed_mode": "url",
                "server_name": None,
                "error": None,
            },
            id="connect-alias-url",
        ),
        pytest.param(
            "/connect @modelcontextprotocol/server-everything",
            {
                "kind": "mcp_connect",
                "target_text": "@modelcontextprotocol/server-everything",
                "parsed_mode": "npx",
                "server_name": None,
                "error": None,
            },
            id="connect-alias-npx-scoped-package",
        ),
        pytest.param(
            "/connect uvx demo-server",
            {
                "kind": "mcp_connect",
                "target_text": "uvx demo-server",
                "parsed_mode": "uvx",
                "server_name": None,
                "error": None,
            },
            id="connect-alias-uvx",
        ),
        pytest.param(
            "/connect python demo_server.py",
            {
                "kind": "mcp_connect",
                "target_text": "python demo_server.py",
                "parsed_mode": "stdio",
                "server_name": None,
                "error": None,
            },
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
        pytest.param(
            "/   ",
            "",
            id="whitespace-only-slash",
        ),
        pytest.param(
            "   /   ",
            "",
            id="leading-whitespace-slash",
        ),
    ],
)
def test_parse_special_input_intent_contract(
    raw_input: str,
    expected: ExpectedParseResult,
) -> None:
    actual = parse_special_input(raw_input)
    if isinstance(expected, dict):
        assert isinstance(actual, McpConnectCommand)
        assert actual.kind == expected["kind"]
        assert actual.target_text == expected["target_text"]
        assert actual.parsed_mode == expected["parsed_mode"]
        assert actual.server_name == expected["server_name"]
        assert actual.error == expected["error"]
        return
    assert actual == expected


def test_parse_attach_uses_windows_aware_tokenization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fast_agent.utils.commandline.os.name", "nt")

    actual = parse_special_input(r'/attach C:\tmp\foo.txt "C:\Program Files\bar.txt"')

    assert actual == AttachCommand(
        paths=(r"C:\tmp\foo.txt", r"C:\Program Files\bar.txt"),
        clear=False,
        error=None,
    )


def test_parse_hash_agent_command_ignores_leading_whitespace() -> None:
    actual = parse_special_input("  ##review please check this")

    assert actual == HashAgentCommand(
        agent_name="review",
        message="please check this",
        quiet=True,
    )
