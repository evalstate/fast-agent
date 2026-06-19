from fast_agent.acp.slash.handlers import plugins as plugins_slash_handler


def test_parse_plugins_arguments_normalizes_aliases() -> None:
    assert plugins_slash_handler._parse_plugins_arguments("install alpha") == ("add", "alpha")
    assert plugins_slash_handler._parse_plugins_arguments("marketplace") == ("available", "")
    assert plugins_slash_handler._parse_plugins_arguments(None) == ("list", "")
