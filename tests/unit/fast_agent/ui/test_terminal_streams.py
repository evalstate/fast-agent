from fast_agent.ui.terminal_streams import is_tty_stream


class TtyStream:
    def __init__(self, value: bool) -> None:
        self.value = value

    def isatty(self) -> bool:
        return self.value

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


class PlainStream:
    pass


class ClosedStream:
    def isatty(self) -> bool:
        raise ValueError("I/O operation on closed file")

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


def test_is_tty_stream_accepts_true_tty_streams() -> None:
    assert is_tty_stream(TtyStream(True)) is True


def test_is_tty_stream_rejects_non_tty_and_plain_streams() -> None:
    assert is_tty_stream(TtyStream(False)) is False
    assert is_tty_stream(PlainStream()) is False


def test_is_tty_stream_rejects_streams_with_unavailable_tty_state() -> None:
    assert is_tty_stream(ClosedStream()) is False
