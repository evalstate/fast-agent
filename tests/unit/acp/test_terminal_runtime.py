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
async def test_session_id_in_all_terminal_requests():
    """Test that sessionId IS included in all terminal method parameters (per ACP spec)."""
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
        session_id="test-session-123",
        activation_reason="test",
        timeout_seconds=90,
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

    assert len(mock_conn._conn.send_request.call_args_list) == 4

    for i, (expected_method, should_have_session) in enumerate(expected_calls):
        call = mock_conn._conn.send_request.call_args_list[i]
        method_name = call[0][0]
        params = call[0][1]

        assert method_name == expected_method, f"Call {i}: expected {expected_method}, got {method_name}"

        if should_have_session:
            assert "sessionId" in params, f"sessionId should be in {method_name} params per ACP spec"
            assert params["sessionId"] == "test-session-123"


@pytest.mark.asyncio
async def test_terminal_timeout_includes_session_id():
    """Test that sessionId is included in kill/output requests during timeout."""
    import asyncio

    # Setup mock connection
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()

    # Mock terminal responses - make wait_for_exit hang to trigger timeout
    async def slow_wait(*args, **kwargs):
        await asyncio.sleep(100)  # Will be interrupted by timeout

    # Set up side effects: create, wait_for_exit (slow), kill, output, release
    mock_conn._conn.send_request.side_effect = [
        {"terminalId": "terminal-1"},  # terminal/create
        slow_wait(),  # terminal/wait_for_exit (will trigger timeout)
        {},  # terminal/kill
        {"output": "partial", "truncated": True, "exitCode": None},  # terminal/output
        {},  # terminal/release
    ]

    runtime = ACPTerminalRuntime(
        connection=mock_conn,
        session_id="test-session",
        activation_reason="test",
        timeout_seconds=0.1,  # Very short timeout to trigger quickly
    )

    result = await runtime.execute({"command": "sleep 1000"})
    assert result.isError

    # Verify sessionId in all calls after timeout
    calls = mock_conn._conn.send_request.call_args_list
    # Should have: create, wait_for_exit, kill, output, release
    assert len(calls) == 5

    # Check create call
    create_call = calls[0]
    assert create_call[0][0] == "terminal/create"
    assert "sessionId" in create_call[0][1]

    # Check wait_for_exit call (the one that timed out)
    wait_call = calls[1]
    assert wait_call[0][0] == "terminal/wait_for_exit"
    assert "sessionId" in wait_call[0][1]

    # Check kill call has sessionId
    kill_call = calls[2]
    assert kill_call[0][0] == "terminal/kill"
    assert "sessionId" in kill_call[0][1]

    # Check output call has sessionId
    output_call = calls[3]
    assert output_call[0][0] == "terminal/output"
    assert "sessionId" in output_call[0][1]

    # Check release call has sessionId
    release_call = calls[4]
    assert release_call[0][0] == "terminal/release"
    assert "sessionId" in release_call[0][1]
