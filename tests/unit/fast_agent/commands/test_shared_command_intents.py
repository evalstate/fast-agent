from fast_agent.commands.shared_command_intents import (
    HistoryActionIntent,
    parse_current_agent_history_intent,
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
