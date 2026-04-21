from fast_agent.commands.shared_command_intents import (
    HistoryActionIntent,
    parse_current_agent_history_intent,
    parse_session_command_intent,
    should_default_export_agent,
)


def test_parse_current_agent_history_intent_unquotes_quoted_arguments() -> None:
    assert parse_current_agent_history_intent('/history load "my history.json"'.removeprefix("/history ")) == (
        HistoryActionIntent(action="load", argument="my history.json")
    )

    assert parse_current_agent_history_intent('/history show "agent name"'.removeprefix("/history ")) == (
        HistoryActionIntent(action="show", argument="agent name")
    )

    assert parse_current_agent_history_intent('/history detail "5"'.removeprefix("/history ")) == (
        HistoryActionIntent(action="detail", turn_index=5)
    )


def test_parse_session_command_intent_parses_export_options() -> None:
    intent = parse_session_command_intent(
        'export latest --agent dev --output "trace file.jsonl" --hf-dataset owner/dataset --hf-dataset-path exports/'
    )

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_agent == "dev"
    assert intent.export_output == "trace file.jsonl"
    assert intent.export_hf_dataset == "owner/dataset"
    assert intent.export_hf_dataset_path == "exports/"
    assert intent.export_error is None


def test_parse_session_command_intent_preserves_windows_export_paths() -> None:
    intent = parse_session_command_intent(
        r"export C:\tmp\session.json --output C:\tmp\trace.jsonl"
    )

    assert intent.action == "export"
    assert intent.export_target == r"C:\tmp\session.json"
    assert intent.export_output == r"C:\tmp\trace.jsonl"
    assert intent.export_error is None


def test_parse_session_command_intent_preserves_quoted_windows_output_paths() -> None:
    intent = parse_session_command_intent(
        r'export latest --output "C:\tmp\trace file.jsonl"'
    )

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_output == r"C:\tmp\trace file.jsonl"
    assert intent.export_error is None


def test_parse_session_command_intent_supports_escaped_spaces_in_export_options() -> None:
    intent = parse_session_command_intent(
        r"export latest --agent dev\ agent --output trace\ file.jsonl"
    )

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_agent == "dev agent"
    assert intent.export_output == "trace file.jsonl"
    assert intent.export_error is None


def test_should_default_export_agent_only_for_current_session_target() -> None:
    assert should_default_export_agent(None, current_session_id="2604201303-x5MNlH") is True
    assert should_default_export_agent(None, current_session_id=None) is False
    assert should_default_export_agent("latest", current_session_id="2604201303-x5MNlH") is False
    assert should_default_export_agent("LATEST", current_session_id="2604201303-x5MNlH") is False
    assert (
        should_default_export_agent("2604201303-x5MNlH", current_session_id="2604201303-x5MNlH")
        is False
    )


def test_parse_session_command_intent_reports_export_option_errors() -> None:
    intent = parse_session_command_intent("export latest --agent")

    assert intent.action == "export"
    assert intent.export_error == "Missing value for --agent"


def test_parse_session_command_intent_rejects_unknown_export_options() -> None:
    intent = parse_session_command_intent("export latest --format codex")

    assert intent.action == "export"
    assert intent.export_error == "Unknown export option: --format"


def test_parse_session_command_intent_supports_export_help() -> None:
    intent = parse_session_command_intent("export latest --help")

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_help is True
    assert intent.export_error is None
