"""Unit tests for ACPTerminalRuntime."""

import pytest
from types import SimpleNamespace

from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.constants import DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT


class RecordingConnection:
    """Simple async connection that records requests and returns preset responses."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    async def send_request(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, params or {}))
        if self._responses:
            return self._responses.pop(0)
        return {}


def build_runtime(
    responses: list[dict],
    session_id: str = "test-session",
    default_limit: int | None = None,
):
    """Create a runtime wired to a recording connection."""
    conn = RecordingConnection(responses)
    runtime = ACPTerminalRuntime(
        connection=SimpleNamespace(_conn=conn),
        session_id=session_id,
        activation_reason="test",
        timeout_seconds=90,
        default_output_byte_limit=default_limit
        if default_limit is not None
        else DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    )
    return runtime, conn


@pytest.mark.asyncio
async def test_env_parameter_transformation_from_object():
    """Test that env parameter is transformed from object to array format."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},  # terminal/create
            {"exitCode": 0, "signal": None},  # terminal/wait_for_exit
            {"output": "test output", "truncated": False, "exitCode": 0},  # terminal/output
            {},  # terminal/release
        ]
    )

    # Execute command with env as object (LLM-friendly format)
    arguments = {
        "command": "env",
        "env": {
            "PATH": "/usr/local/bin",
            "HOME": "/home/testuser",
            "CUSTOM_VAR": "test123",
        },
    }

    await runtime.execute(arguments)

    # Verify terminal/create was called with transformed env (array format)
    method, create_params = conn.calls[0]
    assert method == "terminal/create"

    # Check that env was transformed to array format
    assert "env" in create_params
    assert isinstance(create_params["env"], list)
    assert len(create_params["env"]) == 3

    # Verify each env item has name and value
    env_dict = {item["name"]: item["value"] for item in create_params["env"]}
    assert env_dict["PATH"] == "/usr/local/bin"
    assert env_dict["HOME"] == "/home/testuser"
    assert env_dict["CUSTOM_VAR"] == "test123"


@pytest.mark.asyncio
async def test_env_parameter_passthrough_for_array():
    """Test that env parameter in array format is passed through unchanged."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},  # terminal/create
            {"exitCode": 0, "signal": None},  # terminal/wait_for_exit
            {"output": "test output", "truncated": False, "exitCode": 0},  # terminal/output
            {},  # terminal/release
        ]
    )

    # Execute command with env already in array format
    env_array = [
        {"name": "PATH", "value": "/usr/local/bin"},
        {"name": "HOME", "value": "/home/testuser"},
    ]
    arguments = {
        "command": "env",
        "env": env_array,
    }

    await runtime.execute(arguments)

    # Verify terminal/create was called with env unchanged
    _, create_params = conn.calls[0]

    assert create_params["env"] == env_array


@pytest.mark.asyncio
async def test_optional_parameters_passed_correctly():
    """Test that all optional parameters are passed to terminal/create."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    # Execute command with all optional parameters
    arguments = {
        "command": "ls",
        "args": ["-la", "/tmp"],
        "env": {"DEBUG": "true"},
        "cwd": "/home/testuser",
        "outputByteLimit": 10000,
    }

    await runtime.execute(arguments)

    # Verify all parameters were passed to terminal/create
    _, create_params = conn.calls[0]

    assert create_params["command"] == "ls"
    assert create_params["args"] == ["-la", "/tmp"]
    assert create_params["env"] == [{"name": "DEBUG", "value": "true"}]
    assert create_params["cwd"] == "/home/testuser"
    assert create_params["outputByteLimit"] == 10000


@pytest.mark.asyncio
async def test_default_output_byte_limit_used_when_missing():
    """Ensure a sensible default output limit is applied."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": "echo default"})

    _, create_params = conn.calls[0]
    assert create_params["outputByteLimit"] == DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT


@pytest.mark.asyncio
async def test_custom_default_output_byte_limit_overrides_baseline():
    """Verify callers can override the default terminal output limit."""
    custom_limit = 12345
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-override"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ],
        default_limit=custom_limit,
    )

    await runtime.execute({"command": "echo custom"})

    _, create_params = conn.calls[0]
    assert create_params["outputByteLimit"] == custom_limit


@pytest.mark.asyncio
async def test_session_id_in_all_terminal_requests():
    """Test that sessionId IS included in all terminal method parameters (per ACP spec)."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ],
        session_id="test-session-123",
    )

    await runtime.execute({"command": "echo test"})

    # Verify sessionId IS in all terminal method parameters (per ACP spec)
    # The ACP specification requires sessionId in all terminal methods
    expected_calls = [
        ("terminal/create", True),
        ("terminal/wait_for_exit", True),
        ("terminal/output", True),
        ("terminal/release", True),
    ]

    assert len(conn.calls) == 4

    for (expected_method, should_have_session), (method_name, params) in zip(expected_calls, conn.calls):
        assert method_name == expected_method
        if should_have_session:
            assert "sessionId" in params
            assert params["sessionId"] == "test-session-123"


# Note: Timeout handling is difficult to test reliably in unit tests due to
# asyncio.wait_for() behavior with mocks. The timeout path is tested manually and
# the code includes proper sessionId in all cleanup calls (kill, output, release).
# The test_session_id_in_all_terminal_requests test above verifies sessionId is
# included in the successful path for all terminal methods.
