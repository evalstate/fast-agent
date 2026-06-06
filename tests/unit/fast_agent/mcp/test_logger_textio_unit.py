from __future__ import annotations

from fast_agent.mcp import logger_textio
from fast_agent.mcp.logger_textio import LoggerTextIO


def test_logger_textio_reuses_devnull_fd(monkeypatch) -> None:
    opened: list[str] = []

    def open_stub(path: str, flags: int) -> int:
        del flags
        opened.append(path)
        return 123

    monkeypatch.setattr(logger_textio.os, "open", open_stub)

    stream = LoggerTextIO("demo")
    try:
        assert stream.fileno() == 123
        assert stream.fileno() == 123
    finally:
        stream.close()

    assert opened == [logger_textio.os.devnull]


def test_logger_textio_close_closes_devnull_fd_once(monkeypatch) -> None:
    closed: list[int] = []
    monkeypatch.setattr(logger_textio.os, "open", lambda _path, _flags: 456)
    monkeypatch.setattr(logger_textio.os, "close", closed.append)

    stream = LoggerTextIO("demo")
    assert stream.fileno() == 456

    stream.close()
    stream.close()

    assert closed == [456]
