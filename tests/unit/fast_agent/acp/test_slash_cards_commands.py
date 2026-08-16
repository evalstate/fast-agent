from __future__ import annotations

from fast_agent.acp.slash.handlers import cards_manager as packs_slash_handler


def test_parse_packs_arguments_normalizes_aliases() -> None:
    assert packs_slash_handler._parse_packs_arguments("show alpha") == ("readme", "alpha")
    assert packs_slash_handler._parse_packs_arguments("install alpha") == ("add", "alpha")
    assert packs_slash_handler._parse_packs_arguments(None) == ("list", "")
