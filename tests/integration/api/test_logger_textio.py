"""
Integration tests for the LoggerTextIO class that captures stderr from MCP servers.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_agent.mcp.logger_textio import LoggerTextIO, get_stderr_handler


@pytest.fixture
def test_script_path():
    """Returns the path to the test script that generates stderr output."""
    return Path(__file__).parent / "stderr_test_script.py"


@pytest.fixture
def logger_io():
    """Create a LoggerTextIO instance for testing with proper cleanup."""
    logger_io = LoggerTextIO("test-server")

    yield logger_io

    # Ensure proper cleanup
    logger_io.close()
    if hasattr(logger_io, "_devnull_fd"):
        try:
            os.close(logger_io._devnull_fd)
        except OSError:
            pass


@pytest.mark.integration
def test_logger_textio_fileno(logger_io):
    """Test that fileno returns a valid file descriptor."""
    # Get file descriptor and verify it's a positive integer
    fd = logger_io.fileno()
    assert isinstance(fd, int)
    assert fd > 0

    # Test writing to the file descriptor
    bytes_written = os.write(fd, b"Test message\n")
    assert bytes_written > 0


@pytest.mark.integration
def test_logger_textio_write(logger_io):
    """Test that the write method properly captures and buffers output."""
    # Test complete line
    result = logger_io.write("Complete line\n")
    assert result > 0

    # Test partial line
    result = logger_io.write("Partial ")
    assert result > 0

    # Test completing the partial line
    result = logger_io.write("line completion\n")
    assert result > 0


@pytest.mark.integration
def test_logger_textio_real_process(test_script_path, logger_io):
    """Integration test using a real subprocess with stderr output."""
    # Run the script and capture stderr
    process = subprocess.Popen(
        [sys.executable, str(test_script_path)],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    # Read and process stderr lines
    for line in process.stderr:
        logger_io.write(line)

    # Wait for process to complete
    process.wait()

    # No assertions needed - if any part fails, the test will fail
    # The test verifies that the code executes without exceptions


@pytest.mark.integration
def test_get_stderr_handler():
    """Test that get_stderr_handler returns a valid LoggerTextIO instance."""
    handler = get_stderr_handler("test-handler")

    # Verify it's the right type
    assert isinstance(handler, LoggerTextIO)

    # Verify it has the correct server name
    assert handler.server_name == "test-handler"

    # Verify it has a valid fileno
    fd = handler.fileno()
    assert isinstance(fd, int)
    assert fd > 0

    # Clean up
    handler.close()
