from fast_agent.ui.command_payloads import A2ACommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_a2a_defaults_to_status() -> None:
    result = parse_special_input("/a2a")
    assert isinstance(result, A2ACommand)
    assert result.action == "status"
    assert result.argument is None
    assert result.error is None


def test_parse_a2a_status_target() -> None:
    result = parse_special_input("/a2a status remote")
    assert isinstance(result, A2ACommand)
    assert result.action == "status"
    assert result.argument == "remote"


def test_parse_a2a_connect_preserves_arguments() -> None:
    result = parse_special_input("/a2a connect http://127.0.0.1:41241 --transport JSONRPC")
    assert isinstance(result, A2ACommand)
    assert result.action == "connect"
    assert result.argument == "http://127.0.0.1:41241 --transport JSONRPC"
    assert result.error is None


def test_parse_a2a_unknown_action_reports_error() -> None:
    result = parse_special_input("/a2a wat remote")
    assert isinstance(result, A2ACommand)
    assert result.action == "wat"
    assert result.argument == "remote"
    assert result.error == "Unknown /a2a action: wat"


def test_parse_a2a_transport_target() -> None:
    result = parse_special_input("/a2a transport remote")
    assert isinstance(result, A2ACommand)
    assert result.action == "transport"
    assert result.argument == "remote"


def test_parse_a2a_help_variants() -> None:
    for command in ["/a2a help", "/a2a ?", "/a2a -h", "/a2a --help", "/a2a commands"]:
        result = parse_special_input(command)
        assert isinstance(result, A2ACommand)
        assert result.error is None
