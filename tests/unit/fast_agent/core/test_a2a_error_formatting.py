from fast_agent.core.direct_factory import _format_a2a_initialization_error


def test_format_a2a_initialization_error_uses_cause_chain() -> None:
    cause = TimeoutError()
    exc = RuntimeError("wrapper")
    exc.__cause__ = cause

    message = _format_a2a_initialization_error(
        name="a2a_remote",
        url="http://127.0.0.1:41242",
        transport="JSONRPC",
        exc=exc,
    )

    assert "Unable to initialize A2A agent 'a2a_remote' via JSONRPC" in message
    assert "http://127.0.0.1:41242" in message
    assert "TimeoutError" in message
    assert "Check that the A2A server is running" in message
