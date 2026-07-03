from __future__ import annotations

import os

import pytest

import fast_agent.config as config_module
from fast_agent.constants import FAST_AGENT_RUNTIME_HOME
from fast_agent.session import reset_session_manager


@pytest.fixture(autouse=True)
def isolate_home(tmp_path):
    """Ensure unit tests never write sessions/skills into a real home.

    Unit tests are sometimes run from within an interactive fast-agent process where
    ``FAST_AGENT_HOME`` or ``FAST_AGENT_HOME`` may already point at a real user
    environment. Force an isolated temporary environment path per test to avoid
    reading from or writing to developer session storage.
    """

    original_runtime_environment = os.environ.get(FAST_AGENT_RUNTIME_HOME)
    original_fast_agent_home = os.environ.get("FAST_AGENT_HOME")
    original_home = os.environ.get("FAST_AGENT_HOME")
    isolated_home = tmp_path / ".fast-agent-test-env"
    os.environ.pop(FAST_AGENT_RUNTIME_HOME, None)
    os.environ.pop("FAST_AGENT_HOME", None)
    os.environ["FAST_AGENT_HOME"] = str(isolated_home)
    # Ensure cached global settings never leak across tests.
    config_module._settings = None
    reset_session_manager()

    try:
        yield
    finally:
        reset_session_manager()
        config_module._settings = None
        if original_runtime_environment is None:
            os.environ.pop(FAST_AGENT_RUNTIME_HOME, None)
        else:
            os.environ[FAST_AGENT_RUNTIME_HOME] = original_runtime_environment
        if original_fast_agent_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = original_fast_agent_home
        if original_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = original_home
