from __future__ import annotations

import os

import pytest

import fast_agent.config as config_module
from fast_agent.session import reset_session_manager


@pytest.fixture(autouse=True)
def isolate_environment_dir(tmp_path):
    """Ensure unit tests never write sessions/skills into a real environment directory.

    Unit tests are sometimes run from within an interactive fast-agent process where
    ``ENVIRONMENT_DIR`` may already point at ``.dev``. Force an isolated temporary
    environment path per test to avoid polluting developer session storage.
    """

    original_environment_dir = os.environ.get("ENVIRONMENT_DIR")
    original_settings = getattr(config_module, "_settings", None)
    isolated_environment_dir = tmp_path / ".fast-agent-test-env"
    os.environ["ENVIRONMENT_DIR"] = str(isolated_environment_dir)
    # Ensure cached global settings never leak across tests.
    config_module._settings = None
    reset_session_manager()

    try:
        yield
    finally:
        reset_session_manager()
        config_module._settings = original_settings
        if original_environment_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_environment_dir
