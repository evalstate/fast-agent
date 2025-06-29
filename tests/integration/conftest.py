import importlib
import os
from pathlib import Path

import pytest

from mcp_agent.core.fastagent import FastAgent


# Keep the auto-cleanup fixture
@pytest.fixture(scope="function", autouse=True)
def cleanup_event_bus():
    """Reset the AsyncEventBus between tests using its reset method"""
    # Run the test
    yield

    # Reset the AsyncEventBus after each test
    try:
        # Import the module with the AsyncEventBus
        transport_module = importlib.import_module("mcp_agent.logging.transport")
        AsyncEventBus = getattr(transport_module, "AsyncEventBus", None)

        # Call the reset method if available
        if AsyncEventBus and hasattr(AsyncEventBus, "reset"):
            AsyncEventBus.reset()
    except Exception:
        pass


# Set the project root directory for tests
@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory as a Path object"""
    # Go up from tests/e2e directory to find project root
    return Path(__file__).parent.parent.parent


# Add a fixture that uses the test file's directory
@pytest.fixture
def fast_agent(request):
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

    # Create agent with local config using absolute path
    agent = FastAgent(
        "Test Agent",
        config_path=config_file,  # Use absolute path to local config in test directory
        ignore_unknown_args=True,
    )

    # Provide the agent
    yield agent

    # Restore original directory
    os.chdir(original_cwd)


# Add a fixture that uses the test file's directory
@pytest.fixture
def markup_fast_agent(request):
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

    # Create agent with local config using absolute path
    agent = FastAgent(
        "Test Agent",
        config_path=config_file,  # Use absolute path to local config in test directory
        ignore_unknown_args=True,
    )

    # Provide the agent
    yield agent

    # Restore original directory
    os.chdir(original_cwd)


# Add a fixture for auto_sampling disabled tests
@pytest.fixture
def auto_sampling_off_fast_agent(request):
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

    # Create agent with local config using absolute path
    agent = FastAgent(
        "Test Agent",
        config_path=config_file,
        ignore_unknown_args=True,
    )

    # Provide the agent
    yield agent

    # Restore original directory
    os.chdir(original_cwd)
