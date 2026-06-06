from __future__ import annotations

from fast_agent.acp.slash.handlers import cards_manager as cards_slash_handler


def test_parse_cards_arguments_normalizes_aliases() -> None:
    assert cards_slash_handler._parse_cards_arguments("show alpha") == ("readme", "alpha")
    assert cards_slash_handler._parse_cards_arguments("install alpha") == ("add", "alpha")
    assert cards_slash_handler._parse_cards_arguments(None) == ("list", "")
