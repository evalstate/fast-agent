from fast_agent.commands.shared_command_intents import (
    AgentCommandIntent,
    CardCommandIntent,
    HistoryActionIntent,
    parse_agent_command_intent,
    parse_card_command_intent,
    parse_current_agent_history_intent,
    parse_session_command_intent,
    should_default_export_agent,
)


def test_parse_agent_command_intent_uses_structured_actions() -> None:
    assert parse_agent_command_intent(None) == AgentCommandIntent(action="status")
    assert parse_agent_command_intent("list") == AgentCommandIntent(action="list")
    assert parse_agent_command_intent("use reviewer") == AgentCommandIntent(
        action="use",
        agent_name="reviewer",
    )
    assert parse_agent_command_intent('tool add "code reviewer"') == AgentCommandIntent(
        action="tool_add",
        agent_name="code reviewer",
    )
    assert parse_agent_command_intent("tool remove reviewer") == AgentCommandIntent(
        action="tool_remove",
        agent_name="reviewer",
    )


def test_parse_card_command_intent_uses_structured_actions() -> None:
    assert parse_card_command_intent(None) == CardCommandIntent(action="show")
    assert parse_card_command_intent("show reviewer") == CardCommandIntent(
        action="show",
        agent_name="reviewer",
    )
    assert parse_card_command_intent('load "cards/reviewer.md" --as-tool') == CardCommandIntent(
        action="load",
        source="cards/reviewer.md",
        as_tool=True,
    )


def test_parse_current_agent_history_intent_unquotes_quoted_arguments() -> None:
    assert parse_current_agent_history_intent(
        '/history load "my history.json"'.removeprefix("/history ")
    ) == (HistoryActionIntent(action="load", argument="my history.json"))

    assert parse_current_agent_history_intent(
        '/history show "agent name"'.removeprefix("/history ")
    ) == (HistoryActionIntent(action="show", argument="agent name"))

    assert parse_current_agent_history_intent('/history detail "5"'.removeprefix("/history ")) == (
        HistoryActionIntent(action="detail", turn_index=5)
    )


def test_parse_current_agent_history_review_defaults_to_latest_turn() -> None:
    assert parse_current_agent_history_intent("review") == HistoryActionIntent(action="review")
    assert parse_current_agent_history_intent("review 3") == HistoryActionIntent(
        action="review",
        turn_index=3,
    )


def test_parse_session_command_intent_parses_export_options() -> None:
    intent = parse_session_command_intent(
        'export latest --agent dev --output "trace file.jsonl" --format atif '
        "--hf-url hf://buckets/me/traces/ "
        "--hf-dataset owner/dataset "
        "--hf-dataset-path exports/ --privacy-filter --privacy-filter-path /tmp/model "
        "--download-privacy-filter --privacy-filter-device cpu "
        "--privacy-filter-variant q4f16 --show-redactions"
    )

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_agent == "dev"
    assert intent.export_output == "trace file.jsonl"
    assert intent.export_format == "atif"
    assert intent.export_hf_url == "hf://buckets/me/traces/"
    assert intent.export_hf_dataset == "owner/dataset"
    assert intent.export_hf_dataset_path == "exports/"
    assert intent.export_privacy_filter is True
    assert intent.export_privacy_filter_path == "/tmp/model"
    assert intent.export_download_privacy_filter is True
    assert intent.export_privacy_filter_device == "cpu"
    assert intent.export_privacy_filter_variant == "q4f16"
    assert intent.export_show_redactions is True
    assert intent.export_error is None


def test_parse_session_command_intent_accepts_privacy_filter_quant_alias() -> None:
    intent = parse_session_command_intent(
        "export latest --privacy-filter --privacy-filter-quant=q8"
    )

    assert intent.action == "export"
    assert intent.export_privacy_filter is True
    assert intent.export_privacy_filter_variant == "q8"
    assert intent.export_error is None


def test_parse_session_command_intent_normalizes_latest_export_target() -> None:
    intent = parse_session_command_intent("export LATEST")

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_error is None


def test_parse_session_command_intent_preserves_windows_export_paths() -> None:
    intent = parse_session_command_intent(r"export C:\tmp\session.json --output C:\tmp\trace.jsonl")

    assert intent.action == "export"
    assert intent.export_target == r"C:\tmp\session.json"
    assert intent.export_output == r"C:\tmp\trace.jsonl"
    assert intent.export_error is None


def test_parse_session_command_intent_preserves_quoted_windows_output_paths() -> None:
    intent = parse_session_command_intent(r'export latest --output "C:\tmp\trace file.jsonl"')

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


def test_parse_session_command_intent_accepts_export_format_alias() -> None:
    intent = parse_session_command_intent("export latest --export-format atif")

    assert intent.action == "export"
    assert intent.export_format == "atif"
    assert intent.export_error is None


def test_parse_session_command_intent_supports_export_help() -> None:
    intent = parse_session_command_intent("export latest --help")

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_help is True
    assert intent.export_error is None
