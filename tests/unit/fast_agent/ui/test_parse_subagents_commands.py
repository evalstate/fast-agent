from fast_agent.ui.command_payloads import SubagentsCommand
from fast_agent.ui.prompt.parser import parse_special_input


def test_parse_subagents_defaults_to_list() -> None:
    assert parse_special_input("/subagents") == SubagentsCommand(action="list")


def test_parse_subagents_structured_actions() -> None:
    for action in ("list", "status", "on", "off", "toggle", "help"):
        assert parse_special_input(f"/subagents {action}") == SubagentsCommand(action=action)


def test_parse_subagents_rejects_unknown_or_extra_arguments() -> None:
    unknown = parse_special_input("/subagents wat")
    extra = parse_special_input("/subagents status extra")

    assert isinstance(unknown, SubagentsCommand)
    assert unknown.error == "Unknown /subagents action: wat"
    assert isinstance(extra, SubagentsCommand)
    assert extra.error == "Usage: /subagents [list|status|on|off|toggle|help]"
