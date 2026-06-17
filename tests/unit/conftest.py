from __future__ import annotations

import os

import pytest

import fast_agent.config as config_module
from fast_agent.constants import FAST_AGENT_RUNTIME_ENVIRONMENT
from fast_agent.session import reset_session_manager


@pytest.fixture(autouse=True)
def isolate_environment_dir(tmp_path):
    """Ensure unit tests never write sessions/skills into a real environment directory.

    Unit tests are sometimes run from within an interactive fast-agent process where
    ``FAST_AGENT_HOME`` or ``ENVIRONMENT_DIR`` may already point at a real user
    environment. Force an isolated temporary environment path per test to avoid
    reading from or writing to developer session storage.
    """

    original_runtime_environment = os.environ.get(FAST_AGENT_RUNTIME_ENVIRONMENT)
    original_fast_agent_home = os.environ.get("FAST_AGENT_HOME")
    original_environment_dir = os.environ.get("ENVIRONMENT_DIR")
    isolated_environment_dir = tmp_path / ".fast-agent-test-env"
    os.environ.pop(FAST_AGENT_RUNTIME_ENVIRONMENT, None)
    os.environ.pop("FAST_AGENT_HOME", None)
    os.environ["ENVIRONMENT_DIR"] = str(isolated_environment_dir)
    # Ensure cached global settings never leak across tests.
    config_module._settings = None
    reset_session_manager()

    try:
        yield
    finally:
        reset_session_manager()
        config_module._settings = None
        if original_runtime_environment is None:
            os.environ.pop(FAST_AGENT_RUNTIME_ENVIRONMENT, None)
        else:
            os.environ[FAST_AGENT_RUNTIME_ENVIRONMENT] = original_runtime_environment
        if original_fast_agent_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = original_fast_agent_home
        if original_environment_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_environment_dir
