from __future__ import annotations

import pytest

from fast_agent.integrations import herdr_lifecycle


@pytest.fixture(autouse=True)
def isolate_parent_agent_runtime(monkeypatch: pytest.MonkeyPatch):
    """Keep child CLIs and lifecycle reporting independent of the invoking agent."""
    herdr_lifecycle._reset_for_tests()
    monkeypatch.delenv("HERDR_ENV", raising=False)
    monkeypatch.delenv("HERDR_SOCKET_PATH", raising=False)
    monkeypatch.delenv("HERDR_PANE_ID", raising=False)
    monkeypatch.delenv("FAST_AGENT_SHELL_CHILD", raising=False)
    yield
    herdr_lifecycle._reset_for_tests()
