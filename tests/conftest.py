from __future__ import annotations

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from fast_agent.integrations import herdr_lifecycle


@pytest.fixture(autouse=True)
def isolate_prompt_toolkit_app_session():
    """Give every test its own prompt_toolkit app session, detached from the terminal.

    Building a prompt_toolkit Application binds the ambient AppSession's output,
    and constructing that output probes the terminal. Under pytest's captured
    stdout on Windows the probe raises NoConsoleScreenBufferError, so every test
    that builds UI fails there — while Linux CI never sees it, because the POSIX
    Vt100 output has no equivalent probe.

    Binding a DummyOutput and a pipe input makes the session explicit rather than
    ambient, which is also what keeps these tests from touching the real terminal
    when they are run locally.
    """
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            yield


@pytest.fixture(autouse=True)
def isolate_herdr_lifecycle(monkeypatch: pytest.MonkeyPatch):
    """Prevent tests run inside Herdr from reporting against the developer's pane."""
    herdr_lifecycle._reset_for_tests()
    monkeypatch.delenv("HERDR_ENV", raising=False)
    monkeypatch.delenv("HERDR_SOCKET_PATH", raising=False)
    monkeypatch.delenv("HERDR_PANE_ID", raising=False)
    yield
    herdr_lifecycle._reset_for_tests()
