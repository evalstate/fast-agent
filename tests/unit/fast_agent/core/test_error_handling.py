from fast_agent.core.error_handling import handle_error
from fast_agent.core.exceptions import FastAgentError


def test_handle_error_prints_fast_agent_error_details(capsys) -> None:
    handle_error(
        FastAgentError("Bad config", "Missing server"),
        "Configuration Error",
        "Check fast-agent.yaml",
    )

    captured = capsys.readouterr()

    assert "Configuration Error" in captured.err
    assert "Bad config" in captured.err
    assert "Details:" in captured.err
    assert "Missing server" in captured.err
    assert "Check fast-agent.yaml" in captured.err


def test_handle_error_prints_generic_exception_without_dynamic_fields(capsys) -> None:
    handle_error(ValueError("plain failure"), "Error")

    captured = capsys.readouterr()

    assert "Error" in captured.err
    assert "plain failure" in captured.err
    assert "Details:" not in captured.err
