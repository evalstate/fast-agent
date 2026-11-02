from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from fast_agent.ui.terminal_images import probe

DA = "\x1b[?62;22c"
CELL = "\x1b[6;25;11t"
KITTY = "\x1b_Gi=42;OK\x1b\\"


class Replies:
    """A virtual clock and byte stream; no sleeps or real terminal input."""

    def __init__(self, arrivals: list[tuple[float, str]]) -> None:
        self.now = 0.0
        self.arrivals = [(at, char) for at, text in arrivals for char in text]

    def read(self, fd: int, length: int, timeout: float | None = None) -> str:
        assert timeout is not None
        end = self.now + timeout
        if not self.arrivals or self.arrivals[0][0] > end:
            self.now = end
            raise TimeoutError
        at, char = self.arrivals.pop(0)
        self.now = max(at, self.now)
        return char


def install_replies(monkeypatch: pytest.MonkeyPatch, replies: Replies) -> None:
    stdin = SimpleNamespace(buffer=SimpleNamespace(fileno=lambda: 0), isatty=lambda: True)
    stdout = SimpleNamespace(isatty=lambda: True, write=lambda text: None, flush=lambda: None)
    streams = SimpleNamespace(__stdin__=stdin, __stdout__=stdout)
    monkeypatch.setattr(probe, "sys", streams)
    monkeypatch.setattr(probe.terminal, "sys", streams)
    monkeypatch.setattr(probe.terminal, "capture_mode", nullcontext)
    monkeypatch.setattr(probe.terminal, "read", replies.read)
    monkeypatch.setattr(probe.time, "monotonic", lambda: replies.now)
    monkeypatch.setattr(probe.terminal, "get_tiocgwinsz", lambda: (60, 120, 0, 0))
    monkeypatch.delattr(probe.terminal.probe_terminal, "_result", raising=False)


@pytest.mark.parametrize(
    "arrivals",
    [
        [(0.001, CELL + DA), (0.002, KITTY)],
        [(0.001, KITTY + CELL + DA)],
        [(0.001, CELL + DA + KITTY)],
        [(0.001, CELL + DA), (0.10, KITTY[:8]), (0.15, KITTY[8:])],
        [(0.001, KITTY + DA), (0.002, CELL)],
    ],
)
def test_probe_detects_kitty_in_either_reply_order(monkeypatch, arrivals) -> None:
    replies = Replies(arrivals)
    install_replies(monkeypatch, replies)
    original = probe.terminal.capture_until_primary_da
    try:
        probe.prepare_terminal_image_probe()
        result = probe.terminal.probe_terminal()
        assert result.tgp is True
        assert result.sixel is False
        assert probe.terminal.capture_until_primary_da is original
        # Query replies must not become stray prompt input.
        assert not replies.arrivals
        assert replies.now < 0.3
    finally:
        monkeypatch.delattr(probe.terminal.probe_terminal, "_result", raising=False)


@pytest.mark.parametrize("da,sixel", [(DA, False), ("\x1b[?62;4;22c", True)])
def test_non_kitty_terminal_keeps_geometry_and_bounds_wait(monkeypatch, da, sixel) -> None:
    replies = Replies([(0.001, CELL + da)])
    install_replies(monkeypatch, replies)
    try:
        probe.prepare_terminal_image_probe()
        result = probe.terminal.probe_terminal()
        assert not result.tgp
        assert result.sixel is sixel
        assert result.cell_size == (11, 25)
        assert replies.now < 0.3
    finally:
        monkeypatch.delattr(probe.terminal.probe_terminal, "_result", raising=False)


def test_no_replies_has_a_total_deadline(monkeypatch) -> None:
    replies = Replies([])
    install_replies(monkeypatch, replies)
    with probe._capture_replies(0.5) as response:
        pass
    assert response.sequence == ""
    assert replies.now == 0.5


def test_negative_kitty_reply_is_consumed_without_grace_wait(monkeypatch) -> None:
    replies = Replies([(0.001, CELL + DA), (0.002, "\x1b_Gi=42;ENOTSUP\x1b\\")])
    install_replies(monkeypatch, replies)
    with probe._capture_replies(2) as response:
        pass
    assert "ENOTSUP" in response.sequence
    assert replies.now == 0.002
    assert not replies.arrivals


def test_probe_restores_library_on_error(monkeypatch) -> None:
    install_replies(monkeypatch, Replies([]))
    original = probe.terminal.capture_until_primary_da

    def fail():
        raise OSError("terminal closed")

    monkeypatch.setattr(probe.terminal, "probe_terminal", fail)
    with pytest.raises(OSError):
        probe.prepare_terminal_image_probe()
    assert probe.terminal.capture_until_primary_da is original


def test_redirected_output_does_not_probe(monkeypatch) -> None:
    monkeypatch.setattr(probe, "sys", SimpleNamespace(__stdout__=None))

    def fail():
        pytest.fail("redirected output must not query stdin")

    monkeypatch.setattr(probe.terminal, "probe_terminal", fail)
    probe.prepare_terminal_image_probe()
