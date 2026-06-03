from fast_agent.commands import shared_command_intents
from fast_agent.commands.command_catalog import command_action_names
from fast_agent.commands.shared_command_intents import (
    ADD_TOOL_ACTION,
    DUMP_TOOL_ACTION,
    HISTORY_COMMAND_COMPLETION_DESCRIPTIONS,
    MODEL_COMMAND_ACTION_CATEGORIES,
    MODEL_DIRECT_HANDLER_ACTIONS,
    MODEL_MANAGER_COMMAND_ACTIONS,
    MODEL_VALUE_COMMAND_ACTIONS,
    REMOVE_TOOL_ACTION,
    SESSION_COMMAND_COMPLETION_DESCRIPTIONS,
    TOOL_MUTATION_ACTIONS,
    AgentToolIntent,
    CardLoadIntent,
    HistoryActionIntent,
    ModelCommandIntent,
    parse_agent_tool_intent,
    parse_card_load_intent,
    parse_current_agent_history_intent,
    parse_model_command_intent,
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


def test_parse_current_agent_history_intent_parses_editor_actions() -> None:
    assert parse_current_agent_history_intent("rewind 2") == HistoryActionIntent(
        action="rewind",
        turn_index=2,
    )
    assert parse_current_agent_history_intent("clear last analyst") == HistoryActionIntent(
        action="clear_last",
        argument="analyst",
    )
    assert parse_current_agent_history_intent("clear all") == HistoryActionIntent(
        action="clear_all"
    )
    assert parse_current_agent_history_intent("fix reviewer") == HistoryActionIntent(
        action="fix",
        argument="reviewer",
    )
    assert parse_current_agent_history_intent("webclear researcher") == HistoryActionIntent(
        action="webclear",
        argument="researcher",
    )


def test_parse_current_agent_history_intent_normalizes_action_tokens() -> None:
    assert parse_current_agent_history_intent("  CLEAR last analyst  ") == HistoryActionIntent(
        action="clear_last",
        argument="analyst",
    )


def test_history_completion_descriptions_cover_parser_actions() -> None:
    expected_actions = (
        set(shared_command_intents._SIMPLE_HISTORY_ACTIONS)
        | set(shared_command_intents._HISTORY_ACTION_PARSERS)
    )

    assert expected_actions <= set(HISTORY_COMMAND_COMPLETION_DESCRIPTIONS)


def test_parse_current_agent_history_intent_validates_turn_actions() -> None:
    assert parse_current_agent_history_intent("detail") == HistoryActionIntent(
        action="detail",
        turn_error="missing",
    )
    assert parse_current_agent_history_intent("detail later") == HistoryActionIntent(
        action="detail",
        turn_error="invalid",
    )
    assert parse_current_agent_history_intent("rewind") == HistoryActionIntent(
        action="rewind",
        turn_error="missing",
    )
    assert parse_current_agent_history_intent("rewind previous") == HistoryActionIntent(
        action="rewind",
        turn_error="invalid",
    )
    assert parse_current_agent_history_intent("detail 0") == HistoryActionIntent(
        action="detail",
        turn_error="invalid",
    )
    assert parse_current_agent_history_intent("rewind -1") == HistoryActionIntent(
        action="rewind",
        turn_error="invalid",
    )


def test_parse_current_agent_history_intent_falls_back_on_quoting_errors() -> None:
    assert parse_current_agent_history_intent('show "agent name') == HistoryActionIntent(
        action="show",
        argument='"agent name',
    )


def test_parse_card_load_intent_handles_flags_and_aliases() -> None:
    intent = parse_card_load_intent("card.yml --tool --rm")
    assert intent == CardLoadIntent(
        filename="card.yml",
        tool_action=REMOVE_TOOL_ACTION,
        error=None,
    )
    assert intent.add_tool is True
    assert intent.remove_tool is True


def test_tool_mutation_actions_cover_add_tool_property_compatibility() -> None:
    assert TOOL_MUTATION_ACTIONS == frozenset((ADD_TOOL_ACTION, REMOVE_TOOL_ACTION))
    assert CardLoadIntent(filename="card.yml", tool_action=ADD_TOOL_ACTION).add_tool is True
    assert CardLoadIntent(filename="card.yml", tool_action=REMOVE_TOOL_ACTION).add_tool is True
    assert AgentToolIntent(agent_name="alpha", tool_action=DUMP_TOOL_ACTION).add_tool is False


def test_parse_card_load_intent_rejects_missing_and_extra_arguments() -> None:
    assert parse_card_load_intent("") == CardLoadIntent(
        filename=None,
        tool_action=None,
        error="Filename required for /card",
    )
    assert parse_card_load_intent("card.yml extra") == CardLoadIntent(
        filename="card.yml",
        tool_action=None,
        error="Unexpected arguments: extra",
    )


def test_parse_card_load_intent_rejects_empty_quoted_filename() -> None:
    assert parse_card_load_intent('"" --tool') == CardLoadIntent(
        filename=None,
        tool_action=None,
        error="Filename required for /card",
    )


def test_parse_agent_tool_intent_handles_flags_and_aliases() -> None:
    intent = parse_agent_tool_intent("@alpha --tool --rm")
    assert intent == AgentToolIntent(
        agent_name="alpha",
        tool_action=REMOVE_TOOL_ACTION,
        error=None,
    )
    assert intent.add_tool is True
    assert intent.remove_tool is True
    assert intent.dump is False


def test_parse_agent_tool_intent_allows_missing_tool_agent_by_default() -> None:
    assert parse_agent_tool_intent("--tool") == AgentToolIntent(
        agent_name=None,
        tool_action=ADD_TOOL_ACTION,
        error=None,
    )


def test_parse_agent_tool_intent_can_require_tool_agent() -> None:
    assert parse_agent_tool_intent("--tool", require_tool_agent=True) == AgentToolIntent(
        agent_name=None,
        tool_action=ADD_TOOL_ACTION,
        error="Agent name is required for /agent --tool",
    )


def test_parse_agent_tool_intent_rejects_conflicts_and_extra_arguments() -> None:
    assert parse_agent_tool_intent("alpha --tool --dump").error == (
        "Use either --tool or --dump, not both."
    )
    assert parse_agent_tool_intent("alpha --tool extra").error == (
        "Unexpected arguments: extra"
    )


def test_parse_agent_tool_intent_rejects_empty_subjects() -> None:
    assert parse_agent_tool_intent('"" --tool').error == "Agent name cannot be empty."
    assert parse_agent_tool_intent("@ --tool").error == "Agent name cannot be empty."


def test_parse_model_command_intent_parses_shared_actions() -> None:
    assert parse_model_command_intent("verbosity low") == ModelCommandIntent(
        action="verbosity",
        argument="low",
    )
    assert parse_model_command_intent("catalog anthropic --all") == ModelCommandIntent(
        action="catalog",
        argument="anthropic --all",
    )


def test_parse_model_command_intent_accepts_catalogued_actions() -> None:
    for action in command_action_names("model"):
        assert parse_model_command_intent(action).action == action


def test_model_command_action_classifications_cover_catalog() -> None:
    catalog_actions = frozenset(command_action_names("model"))

    assert frozenset(MODEL_COMMAND_ACTION_CATEGORIES) == catalog_actions
    assert MODEL_VALUE_COMMAND_ACTIONS | MODEL_MANAGER_COMMAND_ACTIONS == catalog_actions
    assert MODEL_VALUE_COMMAND_ACTIONS.isdisjoint(MODEL_MANAGER_COMMAND_ACTIONS)
    assert MODEL_DIRECT_HANDLER_ACTIONS == MODEL_VALUE_COMMAND_ACTIONS


def test_parse_model_command_intent_accepts_catalogued_aliases() -> None:
    assert parse_model_command_intent("--help").action == "help"
    assert parse_model_command_intent("-h").action == "help"


def test_parse_model_command_intent_handles_quoted_subcommand() -> None:
    assert parse_model_command_intent('"switch" gpt-5') == ModelCommandIntent(
        action="switch",
        argument="gpt-5",
    )


def test_parse_model_command_intent_unquotes_single_argument() -> None:
    assert parse_model_command_intent('switch "gpt-5-mini"') == ModelCommandIntent(
        action="switch",
        argument="gpt-5-mini",
    )
    assert parse_model_command_intent('switch ""') == ModelCommandIntent(
        action="switch",
        argument="",
    )


def test_parse_model_command_intent_uses_default_and_reports_unknown() -> None:
    assert parse_model_command_intent("") == ModelCommandIntent(action="reasoning")
    assert parse_model_command_intent(None, default_action="doctor") == ModelCommandIntent(
        action="doctor"
    )
    assert parse_model_command_intent("nonsense value") == ModelCommandIntent(
        action="unknown",
        argument="value",
        raw_subcommand="nonsense",
    )


def test_parse_model_command_intent_reports_shell_parse_errors() -> None:
    assert parse_model_command_intent('switch "unterminated') == ModelCommandIntent(
        action="unknown",
        error="No closing quotation",
    )


def test_parse_session_command_intent_parses_pin_value_and_target() -> None:
    intent = parse_session_command_intent('pin off "review session"')

    assert intent.action == "pin"
    assert intent.pin_value == "off"
    assert intent.pin_target == "review session"


def test_session_command_action_tables_cover_parser_surface() -> None:
    parser_subcommands = frozenset(
        shared_command_intents._SIMPLE_SESSION_ACTIONS
    ) | frozenset(
        shared_command_intents._SESSION_SPECIAL_ACTION_PARSERS
    )

    assert parser_subcommands == frozenset(SESSION_COMMAND_COMPLETION_DESCRIPTIONS)


def test_parse_session_command_intent_handles_quoted_subcommand() -> None:
    intent = parse_session_command_intent('"resume" session-123')

    assert intent.action == "resume"
    assert intent.argument == "session-123"


def test_parse_session_command_intent_normalizes_action_tokens() -> None:
    intent = parse_session_command_intent("  PIN OFF review session  ")

    assert intent.action == "pin"
    assert intent.pin_value == "off"
    assert intent.pin_target == "review session"


def test_parse_session_command_intent_treats_pin_text_as_target() -> None:
    intent = parse_session_command_intent("pin review session")

    assert intent.action == "pin"
    assert intent.pin_value is None
    assert intent.pin_target == "review session"


def test_parse_session_command_intent_treats_pin_number_as_target() -> None:
    intent = parse_session_command_intent("pin 1")

    assert intent.action == "pin"
    assert intent.pin_value is None
    assert intent.pin_target == "1"


def test_parse_session_command_intent_reports_shell_quoting_errors() -> None:
    intent = parse_session_command_intent('title "unterminated')

    assert intent.action == "error"
    assert intent.argument is not None
    assert "No closing quotation" in intent.argument


def test_parse_session_command_intent_parses_export_options() -> None:
    intent = parse_session_command_intent(
        'export latest --agent dev --output "trace file.jsonl" --hf-dataset owner/dataset '
        '--hf-dataset-path exports/ --privacy-filter --privacy-filter-path /tmp/model '
        '--download-privacy-filter --privacy-filter-device cpu '
        '--privacy-filter-variant q4f16 --show-redactions'
    )

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_agent == "dev"
    assert intent.export_output == "trace file.jsonl"
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
    intent = parse_session_command_intent("export latest --privacy-filter --privacy-filter-quant=q8")

    assert intent.action == "export"
    assert intent.export_privacy_filter is True
    assert intent.export_privacy_filter_variant == "q8"
    assert intent.export_error is None


def test_parse_session_command_intent_normalizes_latest_export_target() -> None:
    intent = parse_session_command_intent("export LATEST")

    assert intent.action == "export"
    assert intent.export_target == "latest"
    assert intent.export_error is None

    padded_intent = parse_session_command_intent("export ' latest '")
    assert padded_intent.export_target == "latest"
    assert padded_intent.export_error is None


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


def test_parse_session_command_intent_rejects_empty_export_option_values() -> None:
    cases = {
        "export latest --agent=": "Missing value for --agent",
        'export latest --agent ""': "Missing value for --agent",
        "export latest --output=": "Missing value for --output",
        "export latest --hf-dataset=": "Missing value for --hf-dataset",
        "export latest --hf-dataset-path=": "Missing value for --hf-dataset-path",
        "export latest --privacy-filter-path=": "Missing value for --privacy-filter-path",
        "export latest --privacy-filter-device=": "Missing value for --privacy-filter-device",
        "export latest --privacy-filter-variant=": "Missing value for --privacy-filter-variant",
        "export latest --privacy-filter-quant=": "Missing value for --privacy-filter-quant",
    }

    for command, error in cases.items():
        intent = parse_session_command_intent(command)

        assert intent.action == "export"
        assert intent.export_error == error


def test_parse_session_command_intent_rejects_duplicate_export_value_options() -> None:
    cases = {
        "export latest --agent dev --agent reviewer": "Duplicate export option: --agent",
        "export latest --output trace.jsonl --output other.jsonl": (
            "Duplicate export option: --output"
        ),
        "export latest --privacy-filter-variant q4 --privacy-filter-quant q8": (
            "Duplicate export option: --privacy-filter-variant"
        ),
    }

    for command, error in cases.items():
        intent = parse_session_command_intent(command)

        assert intent.action == "export"
        assert intent.export_error == error


def test_parse_session_command_intent_does_not_consume_next_option_as_value() -> None:
    intent = parse_session_command_intent("export latest --agent --output trace.jsonl")

    assert intent.action == "export"
    assert intent.export_error == "Missing value for --agent"


def test_parse_session_command_intent_accepts_dash_prefixed_value_with_equals() -> None:
    intent = parse_session_command_intent("export latest --agent=-agent")

    assert intent.action == "export"
    assert intent.export_agent == "-agent"
    assert intent.export_error is None


def test_parse_session_command_intent_rejects_unknown_export_options() -> None:
    intent = parse_session_command_intent("export latest --format codex")

    assert intent.action == "export"
    assert intent.export_error == "Unknown export option: --format"


def test_parse_session_command_intent_supports_export_help_flags() -> None:
    long_help_intent = parse_session_command_intent("export latest --help")
    short_help_intent = parse_session_command_intent("export latest -h")

    for intent in (long_help_intent, short_help_intent):
        assert intent.action == "export"
        assert intent.export_target == "latest"
        assert intent.export_help is True
        assert intent.export_error is None
