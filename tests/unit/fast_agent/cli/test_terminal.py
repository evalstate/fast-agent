from rich.console import Console, RenderableType
from rich.status import Status
from rich.style import StyleType
from rich.text import Text

from fast_agent.cli.terminal import Application


class _RecordingConsole(Console):
    def __init__(self) -> None:
        super().__init__(record=True, width=80)
        self.renderable: object | None = None

    def status(
        self,
        status: RenderableType,
        *,
        spinner: str = "dots",
        spinner_style: StyleType = "status.spinner",
        speed: float = 1.0,
        refresh_per_second: float = 12.5,
    ) -> Status:
        self.renderable = status
        return super().status(
            status,
            spinner=spinner,
            spinner_style=spinner_style,
            speed=speed,
            refresh_per_second=refresh_per_second,
        )


def test_terminal_log_prints_bracketed_messages_literally() -> None:
    app = Application()
    app.console = Console(record=True, width=80)

    app.log("use [draft] registry")

    assert "[INFO] use [draft] registry" in app.console.export_text()


def test_terminal_status_uses_literal_text_renderable() -> None:
    app = Application()
    recording_console = _RecordingConsole()
    app.console = recording_console

    app.status("indexing [draft]")

    assert isinstance(recording_console.renderable, Text)
    assert recording_console.renderable.plain == "indexing [draft]"
    assert recording_console.renderable.style == "bold cyan"
