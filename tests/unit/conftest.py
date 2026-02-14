from __future__ import annotations

import os

import pytest

from fast_agent.session import reset_session_manager


@pytest.fixture(autouse=True)
def isolate_environment_dir(tmp_path):
    """Ensure unit tests never write sessions/skills into a real environment directory.

    Unit tests are sometimes run from within an interactive fast-agent process where
    ``ENVIRONMENT_DIR`` may already point at ``.dev``. Force an isolated temporary
    environment path per test to avoid polluting developer session storage.
    """

    original_environment_dir = os.environ.get("ENVIRONMENT_DIR")
    isolated_environment_dir = tmp_path / ".fast-agent-test-env"
    os.environ["ENVIRONMENT_DIR"] = str(isolated_environment_dir)
    reset_session_manager()

    try:
        yield
    finally:
        reset_session_manager()
        if original_environment_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_environment_dir
