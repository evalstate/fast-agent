from pathlib import Path

from fast_agent.core.agent_card_validation import scan_agent_card_directory


def test_scan_agent_cards_reports_invalid_history_json(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text('{"messages": ["bad\x00"]}', encoding="utf-8")

    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: test_agent",
                "messages:",
                "  - history.json",
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    assert len(results) == 1

    errors = results[0].errors
    assert any("History file failed to load" in err for err in errors)
    assert any("Failed to parse JSON prompt file" in err for err in errors)
