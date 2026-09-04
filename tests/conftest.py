from __future__ import annotations

import pytest

from fast_agent.integrations import herdr_lifecycle


@pytest.fixture(autouse=True)
def isolate_herdr_lifecycle(monkeypatch: pytest.MonkeyPatch):
    """Prevent tests run inside Herdr from reporting against the developer's pane."""
    herdr_lifecycle._reset_for_tests()
    monkeypatch.delenv("HERDR_ENV", raising=False)
    monkeypatch.delenv("HERDR_SOCKET_PATH", raising=False)
    monkeypatch.delenv("HERDR_PANE_ID", raising=False)
    yield
    herdr_lifecycle._reset_for_tests()
