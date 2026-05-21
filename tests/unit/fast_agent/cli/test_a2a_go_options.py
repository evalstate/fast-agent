from pathlib import Path

import pytest
import typer

from fast_agent.cli.commands.go import _materialize_a2a_agent_cards


def test_materialize_a2a_agent_card() -> None:
    tempdir, paths = _materialize_a2a_agent_cards(
        ["http://127.0.0.1:41241/.well-known/agent-card.json"],
        transport="rest",
        oauth=True,
    )
    try:
        assert len(paths) == 1
        text = Path(paths[0]).read_text(encoding="utf-8")
        assert "type: a2a" in text
        assert "name: a2a_remote" in text
        assert "url: http://127.0.0.1:41241" in text
        assert "transport: HTTP+JSON" in text
        assert "auth:" in text
        assert "  oauth: true" in text
        assert "relative_card_path: /.well-known/agent-card.json" in text
    finally:
        tempdir.cleanup()


def test_materialize_a2a_rejects_bad_transport() -> None:
    with pytest.raises(typer.BadParameter):
        _materialize_a2a_agent_cards(["http://127.0.0.1:41241"], transport="bogus")
