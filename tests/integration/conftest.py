import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest

from fast_agent import FastAgent


# Keep the auto-cleanup fixture
@pytest.fixture(scope="function", autouse=True)
def cleanup_event_bus():
    """Reset the AsyncEventBus between tests using its reset method"""
    # Run the test
    yield

    # Reset the AsyncEventBus after each test
    try:
        # Import the module with the AsyncEventBus
        transport_module = importlib.import_module("fast_agent.core.logging.transport")
        AsyncEventBus = getattr(transport_module, "AsyncEventBus", None)

        # Call the reset method if available
        if AsyncEventBus and hasattr(AsyncEventBus, "reset"):
            AsyncEventBus.reset()
    except Exception:
        pass


@pytest.fixture(scope="session")
def mcp_test_ports():
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "")
    worker_index = 0
    if worker_id.startswith("gw"):
        suffix = worker_id[2:]
        if suffix.isdigit():
            worker_index = int(suffix)

    stride = int(os.getenv("FAST_AGENT_TEST_PORT_STRIDE", "10"))
    offset = worker_index * stride
    ports = {
        "sse": 8723 + offset,
        "http": 8724 + offset,
        "request_http": 8731 + offset,
    }

    env_updates = {
        "FAST_AGENT_TEST_SSE_PORT": str(ports["sse"]),
        "FAST_AGENT_TEST_HTTP_PORT": str(ports["http"]),
    }
    previous = {key: os.getenv(key) for key in env_updates}
    os.environ.update(env_updates)

    yield ports

    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def wait_for_port():
    async def _wait_for_port(
        host: str,
        port: int,
        *,
        process=None,
        timeout: float = 5.0,
        interval: float = 0.1,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if process is not None and process.poll() is not None:
                stdout = ""
                stderr = ""
                if process.stdout or process.stderr:
                    stdout, stderr = process.communicate(timeout=1)
                raise AssertionError(
                    f"Server exited early. stdout={stdout!r} stderr={stderr!r}"
                )
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        f"Server did not start listening on {host}:{port}"
                    )
                await asyncio.sleep(interval)

    return _wait_for_port


# Set the project root directory for tests
@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory as a Path object"""
    # Go up from tests/e2e directory to find project root
    return Path(__file__).parent.parent.parent


# Add a fixture that uses the test file's directory
@pytest.fixture
def fast_agent(request, mcp_test_ports):
    """
    Creates a FastAgent with config from the test file's directory.
    Automatically changes working directory to match the test file location.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory
    os.chdir(test_dir)

    # Explicitly create absolute path to the config file in the test directory
    config_file = os.path.join(test_dir, "fastagent.config.yaml")

    # Avoid pytest args being parsed as FastAgent CLI args.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        # Create agent with local config using absolute path
        agent = FastAgent(
            "Test Agent",
            config_path=config_file,  # Use absolute path to local config in test directory
            ignore_unknown_args=True,
        )
    finally:
        sys.argv = original_argv

    # Provide the agent
    yield agent

    # Restore original directory
    os.chdir(original_cwd)


# Add a fixture that uses the test file's directory
@pytest.fixture
def markup_fast_agent(request, mcp_test_ports):
    """
    Creates a FastAgent with config from the test file's directory.
    Automatically changes working directory to match the test file location.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory
    os.chdir(test_dir)

    # Explicitly create absolute path to the config file in the test directory
    config_file = os.path.join(test_dir, "fastagent.config.markup.yaml")

    # Avoid pytest args being parsed as FastAgent CLI args.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        # Create agent with local config using absolute path
        agent = FastAgent(
            "Test Agent",
            config_path=config_file,  # Use absolute path to local config in test directory
            ignore_unknown_args=True,
        )
    finally:
        sys.argv = original_argv

    # Provide the agent
    yield agent

    # Restore original directory
    os.chdir(original_cwd)


# Add a fixture for auto_sampling disabled tests
@pytest.fixture
def auto_sampling_off_fast_agent(request, mcp_test_ports):
    """
    Creates a FastAgent with auto_sampling disabled config from the test file's directory.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory
    os.chdir(test_dir)

    # Explicitly create absolute path to the config file in the test directory
    config_file = os.path.join(test_dir, "fastagent.config.auto_sampling_off.yaml")

    # Avoid pytest args being parsed as FastAgent CLI args.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        # Create agent with local config using absolute path
        agent = FastAgent(
            "Test Agent",
            config_path=config_file,
            ignore_unknown_args=True,
        )
    finally:
        sys.argv = original_argv

    # Provide the agent
    yield agent

    # Restore original directory
    os.chdir(original_cwd)
