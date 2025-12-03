import importlib
import os
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


# Set the project root directory for tests
@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory as a Path object"""
    # Go up from tests/e2e directory to find project root
    return Path(__file__).parent.parent.parent


# Module-scoped FastAgent fixture for performance
@pytest.fixture(scope="module")
def fast_agent_module(request):
    """
    Module-scoped FastAgent instance - created once per test module.
    This significantly improves test performance by avoiding repeated initialization.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory
    os.chdir(test_dir)

    try:
        # Explicitly create absolute path to the config file in the test directory
        config_file = os.path.join(test_dir, "fastagent.config.yaml")

        # Create agent with local config using absolute path
        agent = FastAgent(
            "Test Agent",
            config_path=config_file,
            ignore_unknown_args=True,
        )

        yield agent
    finally:
        # Restore original directory
        os.chdir(original_cwd)


# Function-scoped wrapper that uses the module-scoped agent
@pytest.fixture
def fast_agent(fast_agent_module, request):
    """
    Function-scoped FastAgent fixture that reuses the module-scoped instance.
    The AsyncEventBus cleanup (autouse fixture) ensures tests don't interfere with each other.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory for this test
    os.chdir(test_dir)

    try:
        # Return the module-scoped agent instance
        yield fast_agent_module
    finally:
        # Restore original directory after test
        os.chdir(original_cwd)


# Module-scoped markup FastAgent fixture for performance
@pytest.fixture(scope="module")
def markup_fast_agent_module(request):
    """
    Module-scoped FastAgent instance with markup config.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory
    os.chdir(test_dir)

    try:
        # Explicitly create absolute path to the config file in the test directory
        config_file = os.path.join(test_dir, "fastagent.config.markup.yaml")

        # Create agent with local config using absolute path
        agent = FastAgent(
            "Test Agent",
            config_path=config_file,
            ignore_unknown_args=True,
        )

        yield agent
    finally:
        # Restore original directory
        os.chdir(original_cwd)


# Function-scoped wrapper for markup agent
@pytest.fixture
def markup_fast_agent(markup_fast_agent_module, request):
    """
    Function-scoped markup FastAgent fixture that reuses the module-scoped instance.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory for this test
    os.chdir(test_dir)

    try:
        yield markup_fast_agent_module
    finally:
        # Restore original directory after test
        os.chdir(original_cwd)


# Module-scoped auto_sampling_off FastAgent fixture for performance
@pytest.fixture(scope="module")
def auto_sampling_off_fast_agent_module(request):
    """
    Module-scoped FastAgent instance with auto_sampling disabled config.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory
    os.chdir(test_dir)

    try:
        # Explicitly create absolute path to the config file in the test directory
        config_file = os.path.join(test_dir, "fastagent.config.auto_sampling_off.yaml")

        # Create agent with local config using absolute path
        agent = FastAgent(
            "Test Agent",
            config_path=config_file,
            ignore_unknown_args=True,
        )

        yield agent
    finally:
        # Restore original directory
        os.chdir(original_cwd)


# Function-scoped wrapper for auto_sampling_off agent
@pytest.fixture
def auto_sampling_off_fast_agent(auto_sampling_off_fast_agent_module, request):
    """
    Function-scoped auto_sampling_off FastAgent fixture that reuses the module-scoped instance.
    """
    # Get the directory where the test file is located
    test_module = request.module.__file__
    test_dir = os.path.dirname(test_module)

    # Save original directory
    original_cwd = os.getcwd()

    # Change to the test file's directory for this test
    os.chdir(test_dir)

    try:
        yield auto_sampling_off_fast_agent_module
    finally:
        # Restore original directory after test
        os.chdir(original_cwd)
