import pytest

from fast_agent.ui.command_payloads import WarmCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/warm", WarmCommand()),
        ("/warm 45", WarmCommand(minutes=45)),
        ("/warm off", WarmCommand(action="off")),
        ("/warm STATUS", WarmCommand(action="status")),
        ("/warm ping", WarmCommand(action="ping")),
        ("/warm PING", WarmCommand(action="ping")),
    ],
)
def test_warm_command_forms(text: str, expected: WarmCommand) -> None:
    assert parse_special_input(text) == expected


@pytest.mark.parametrize("text", ["/warm 0", "/warm 2000", "/warm soon", "/warm 5 off"])
def test_warm_command_rejects_unusable_arguments(text: str) -> None:
    result = parse_special_input(text)
    assert isinstance(result, WarmCommand)
    assert result.error is not None
