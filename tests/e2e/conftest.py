import importlib
import os
import subprocess
import time
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


# Fixture to set the current working directory for tests
@pytest.fixture
def set_cwd(project_root):
    """Change to the project root directory during tests"""
    # Save the original working directory
    original_cwd = os.getcwd()

    # Change to the project root directory
    os.chdir(project_root)

    # Run the test
    yield

    # Restore the original working directory
    os.chdir(original_cwd)


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


# Fixture to manage TensorZero docker-compose environment
@pytest.fixture(scope="session")
def tensorzero_docker_env(project_root):
    """Ensures the TensorZero docker-compose environment is running."""
    compose_file = project_root / "examples" / "tensorzero" / "docker-compose.yml"
    compose_dir = compose_file.parent
    compose_cmd = ["docker", "compose", "-f", str(compose_file)]

    print(f"\nEnsuring TensorZero Docker environment is up ({compose_file})...")
    try:
        # Use --wait flag if available, otherwise fallback to time.sleep
        check_wait_support_cmd = compose_cmd + ["up", "--help"]
        help_output = subprocess.run(
            check_wait_support_cmd, capture_output=True, text=True, cwd=compose_dir
        )
        use_wait_flag = "--wait" in help_output.stdout or "--wait" in help_output.stderr

        up_command = compose_cmd + ["up", "-d"]
        if use_wait_flag:
            up_command.append("--wait")

        start_result = subprocess.run(
            up_command, check=True, capture_output=True, text=True, cwd=compose_dir
        )
        print("TensorZero Docker 'up -d' completed.")
        print(start_result.stdout)
        if start_result.stderr:
            print(f"Stderr: {start_result.stderr}")

        # If --wait is not supported, add a manual delay
        if not use_wait_flag:
            print("Docker compose '--wait' flag not supported, adding manual delay...")
            time.sleep(20)  # Increased sleep time as fallback

    except subprocess.CalledProcessError as e:
        print(f"Error starting TensorZero Docker services: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        pytest.fail(f"Failed to start docker-compose services from {compose_file}")
        return  # Exit if start failed

    yield  # Run tests

    # Stop services
    print("\nTearing down TensorZero Docker environment...")
    try:
        stop_result = subprocess.run(
            compose_cmd + ["down"], check=True, capture_output=True, text=True, cwd=compose_dir
        )
        print("TensorZero Docker 'down' completed.")
        print(stop_result.stdout)
        if stop_result.stderr:
            print(f"Stderr: {stop_result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error stopping TensorZero Docker services: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        # Don't fail the test run if teardown fails, but log it.


# Fixture to temporarily change CWD to examples/tensorzero
@pytest.fixture
def chdir_to_tensorzero_example(project_root):
    """Change CWD to the tensorzero example directory for a test."""
    original_cwd = Path.cwd()
    example_dir = project_root / "examples" / "tensorzero"
    if not example_dir.is_dir():
        pytest.skip(
            f"TensorZero example directory not found at {example_dir}"
        )  # Use skip instead of fail
        return
    os.chdir(example_dir)
    print(f"\nChanged CWD to: {example_dir}")
    yield
    os.chdir(original_cwd)
    print(f"\nRestored CWD to: {original_cwd}")
