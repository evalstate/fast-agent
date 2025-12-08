"""Tests for ACP terminal lifecycle management (create/release)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402


@pytest.mark.unit
def test_terminal_id_generation() -> None:
    """Test that client generates sequential terminal IDs."""
    client = TestClient()

    # IDs should be sequential
    assert client._terminal_count == 0

    # No terminals exist initially
    assert len(client.terminals) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_create_lifecycle() -> None:
    """Test complete terminal create/release lifecycle."""
    client = TestClient()

    # Create first terminal
    result1 = await client.create_terminal(command="echo hello", session_id="test-session")
    terminal_id1 = result1.terminalId

    assert terminal_id1 == "terminal-1"
    assert len(client.terminals) == 1
    assert client.terminals[terminal_id1]["command"] == "echo hello"

    # Create second terminal
    result2 = await client.create_terminal(command="pwd", session_id="test-session")
    terminal_id2 = result2.terminalId

    assert terminal_id2 == "terminal-2"
    assert len(client.terminals) == 2

    # Release first terminal
    await client.release_terminal(session_id="test-session", terminal_id=terminal_id1)
    assert terminal_id1 not in client.terminals
    assert len(client.terminals) == 1

    # Second terminal still exists
    assert terminal_id2 in client.terminals

    # Release second terminal
    await client.release_terminal(session_id="test-session", terminal_id=terminal_id2)
    assert len(client.terminals) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_output_retrieval() -> None:
    """Test retrieving output from terminal."""
    client = TestClient()

    # Create terminal
    result = await client.create_terminal(command="echo test output", session_id="test-session")
    terminal_id = result.terminalId

    # Get output
    output = await client.terminal_output(session_id="test-session", terminal_id=terminal_id)

    assert "Executed: echo test output" in output.output
    assert output.truncated is False

    # Cleanup
    await client.release_terminal(session_id="test-session", terminal_id=terminal_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_wait_for_exit() -> None:
    """Test waiting for terminal exit."""
    client = TestClient()

    # Create terminal
    result = await client.create_terminal(command="echo test", session_id="test-session")
    terminal_id = result.terminalId

    # Wait for exit (immediate in test client)
    exit_result = await client.wait_for_terminal_exit(
        session_id="test-session", terminal_id=terminal_id
    )

    assert exit_result.exitCode == 0
    assert exit_result.signal is None

    # Cleanup
    await client.release_terminal(session_id="test-session", terminal_id=terminal_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_kill() -> None:
    """Test killing a terminal."""
    client = TestClient()

    # Create terminal
    result = await client.create_terminal(command="sleep 100", session_id="test-session")
    terminal_id = result.terminalId

    # Kill it
    await client.kill_terminal(session_id="test-session", terminal_id=terminal_id)

    # Check it was marked as killed
    assert client.terminals[terminal_id]["exit_code"] == -1
    assert client.terminals[terminal_id]["completed"] is True

    # Wait should now show killed
    exit_result = await client.wait_for_terminal_exit(
        session_id="test-session", terminal_id=terminal_id
    )
    assert exit_result.exitCode is None
    assert exit_result.signal == "SIGKILL"

    # Cleanup
    await client.release_terminal(session_id="test-session", terminal_id=terminal_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_release_cleanup() -> None:
    """Test that release properly cleans up terminal state."""
    client = TestClient()

    # Create multiple terminals
    terminals = []
    for i in range(3):
        result = await client.create_terminal(command=f"echo {i}", session_id="test-session")
        terminals.append(result.terminalId)

    assert len(client.terminals) == 3

    # Release all
    for terminal_id in terminals:
        await client.release_terminal(session_id="test-session", terminal_id=terminal_id)

    # All should be gone
    assert len(client.terminals) == 0

    # Releasing non-existent terminal should not error
    await client.release_terminal(session_id="test-session", terminal_id="nonexistent")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_missing_id() -> None:
    """Test operations on missing terminal ID."""
    client = TestClient()

    # Output from non-existent terminal returns empty
    output = await client.terminal_output(session_id="test-session", terminal_id="missing")
    assert output.output == ""
    # TerminalOutputResponse uses exit_status; default is None when missing terminal
    assert getattr(output, "exit_status", None) is None

    # Wait for non-existent terminal
    exit_result = await client.wait_for_terminal_exit(
        session_id="test-session", terminal_id="missing"
    )
    assert exit_result.exitCode is None

    # Kill non-existent terminal (should not error)
    await client.kill_terminal(session_id="test-session", terminal_id="missing")
