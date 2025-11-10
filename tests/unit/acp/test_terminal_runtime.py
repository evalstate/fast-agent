"""Unit tests for ACPTerminalRuntime."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fast_agent.acp.terminal_runtime import ACPTerminalRuntime


@pytest.mark.asyncio
async def test_env_parameter_transformation_from_object():
    """Test that env parameter is transformed from object to array format."""
    # Setup mock connection
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock()

    # Mock terminal responses
    mock_conn._conn.send_request.side_effect = [
        {"terminalId": "terminal-1"},  # terminal/create
        {"exitCode": 0, "signal": None},  # terminal/wait_for_exit
        {"output": "test output", "truncated": False, "exitCode": 0},  # terminal/output
        {},  # terminal/release
    ]

    runtime = ACPTerminalRuntime(
        connection=mock_conn,
        session_id="test-session",
        activation_reason="test",
        timeout_seconds=90,
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
    create_call = mock_conn._conn.send_request.call_args_list[0]
    assert create_call[0][0] == "terminal/create"
    create_params = create_call[0][1]

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
    # Setup mock connection
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock()

    # Mock terminal responses
    mock_conn._conn.send_request.side_effect = [
        {"terminalId": "terminal-1"},  # terminal/create
        {"exitCode": 0, "signal": None},  # terminal/wait_for_exit
        {"output": "test output", "truncated": False, "exitCode": 0},  # terminal/output
        {},  # terminal/release
    ]

    runtime = ACPTerminalRuntime(
        connection=mock_conn,
        session_id="test-session",
        activation_reason="test",
        timeout_seconds=90,
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
    create_call = mock_conn._conn.send_request.call_args_list[0]
    create_params = create_call[0][1]

    assert create_params["env"] == env_array


@pytest.mark.asyncio
async def test_optional_parameters_passed_correctly():
    """Test that all optional parameters are passed to terminal/create."""
    # Setup mock connection
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock()

    # Mock terminal responses
    mock_conn._conn.send_request.side_effect = [
        {"terminalId": "terminal-1"},
        {"exitCode": 0, "signal": None},
        {"output": "test output", "truncated": False, "exitCode": 0},
        {},
    ]

    runtime = ACPTerminalRuntime(
        connection=mock_conn,
        session_id="test-session",
        activation_reason="test",
        timeout_seconds=90,
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
    create_call = mock_conn._conn.send_request.call_args_list[0]
    create_params = create_call[0][1]

    assert create_params["command"] == "ls"
    assert create_params["args"] == ["-la", "/tmp"]
    assert create_params["env"] == [{"name": "DEBUG", "value": "true"}]
    assert create_params["cwd"] == "/home/testuser"
    assert create_params["outputByteLimit"] == 10000


@pytest.mark.asyncio
async def test_no_session_id_in_terminal_requests():
    """Test that sessionId is NOT included in terminal method parameters."""
    # Setup mock connection
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock()

    # Mock terminal responses
    mock_conn._conn.send_request.side_effect = [
        {"terminalId": "terminal-1"},
        {"exitCode": 0, "signal": None},
        {"output": "test output", "truncated": False, "exitCode": 0},
        {},
    ]

    runtime = ACPTerminalRuntime(
        connection=mock_conn,
        session_id="test-session",
        activation_reason="test",
        timeout_seconds=90,
    )

    await runtime.execute({"command": "echo test"})

    # Verify sessionId is NOT in any terminal method parameters
    for call in mock_conn._conn.send_request.call_args_list:
        method_name = call[0][0]
        params = call[0][1]
        assert "sessionId" not in params, f"sessionId should not be in {method_name} params"
