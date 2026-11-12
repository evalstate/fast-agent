from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO, StringIO

import pytest
from PIL import Image
from rich.console import Console
from typer.testing import CliRunner

from fast_agent.cli.checks import terminal
from fast_agent.cli.commands.check_config import app
from fast_agent.config import Settings, TerminalImageSettings

runner = CliRunner()


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(terminal, "get_settings", lambda **kwargs: Settings())
    monkeypatch.delenv("FAST_AGENT_DISABLE_PROGRAM_STATUS", raising=False)


def test_json_is_parseable_and_never_probes_even_with_a_tty(monkeypatch) -> None:
    monkeypatch.setattr(terminal, "_interactive_terminal", lambda: True)

    def fail():
        pytest.fail("JSON must not emit capability queries")

    monkeypatch.setattr(terminal, "prepare_terminal_image_probe", fail)
    result = runner.invoke(app, ["terminal", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["probe"] == "skipped"
    assert data["reported_kitty"] is None
    assert data["selected_renderer"] is None
    assert "\x1b" not in result.stdout


def test_non_tty_plain_output_does_not_probe(monkeypatch) -> None:
    monkeypatch.setattr(terminal, "_interactive_terminal", lambda: False)
    monkeypatch.setattr(terminal, "prepare_terminal_image_probe", lambda: pytest.fail("probe"))
    result = runner.invoke(app, ["terminal"])
    assert result.exit_code == 0, result.output
    assert "skipped" in result.output
    assert "\x1b" not in result.stdout


@pytest.mark.parametrize("args", [["--image", "--json"], ["--image"]])
def test_image_rejects_json_or_non_tty(args, monkeypatch) -> None:
    monkeypatch.setattr(terminal, "_interactive_terminal", lambda: False)
    result = runner.invoke(app, ["terminal", *args])
    assert result.exit_code == 2
    assert "requires interactive" in result.output or "cannot be combined" in result.output


def test_home_is_forwarded_and_disabled_settings_are_reported(monkeypatch, tmp_path) -> None:
    homes = []
    config = Settings()
    config.logger.terminal_images = TerminalImageSettings(enabled=False, backend="halfcell")

    def load(**kwargs):
        homes.append(kwargs["home"])
        return config

    monkeypatch.setattr(terminal, "get_settings", load)
    result = runner.invoke(app, ["--home", str(tmp_path), "terminal", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["images_enabled"] is False
    assert data["configured_backend"] == "halfcell"
    assert homes == [tmp_path]


def test_tty_probe_reports_observed_support_not_terminal_brand(monkeypatch) -> None:
    from textual_image import _terminal

    class FakeImage:
        pass

    monkeypatch.setenv("TERM_PROGRAM", "something-misleading")
    monkeypatch.setenv("HC_TERMINAL_ID", "t42")
    monkeypatch.setattr(terminal, "_interactive_terminal", lambda: True)
    calls = []
    monkeypatch.setattr(terminal, "prepare_terminal_image_probe", lambda: calls.append("probe"))
    monkeypatch.setattr(
        _terminal,
        "probe_terminal",
        lambda: _terminal.TerminalCapabilities(_terminal.CellSize(11, 25), False, True),
    )
    monkeypatch.setattr(terminal, "_resolve_textual_image_class", lambda backend: FakeImage)
    report = terminal.inspect_terminal(TerminalImageSettings(), json_output=False)
    assert calls == ["probe"]
    assert report.probe == "completed"
    assert report.reported_kitty is True
    assert report.reported_sixel is False
    assert (report.cell_width, report.cell_height) == (11, 25)
    assert report.hc_terminal_id == "t42"
    assert report.selected_renderer and report.selected_renderer.endswith(".FakeImage")
    assert "Not checked" in report.model_image_support


def test_failed_probe_is_diagnostic_not_traceback(monkeypatch) -> None:
    monkeypatch.setattr(terminal, "_interactive_terminal", lambda: True)

    def fail():
        raise OSError("closed terminal")

    monkeypatch.setattr(terminal, "prepare_terminal_image_probe", fail)
    result = runner.invoke(app, ["terminal"])
    assert result.exit_code == 1
    assert "closed terminal" in result.output


def test_chart_is_deterministic_and_contains_fine_detail() -> None:
    data = terminal.test_chart_png()
    assert data == terminal.test_chart_png()
    with Image.open(BytesIO(data)) as image:
        assert image.size == (1200, 240)
        assert image.getpixel((600, 20)) != image.getpixel((601, 20))
        assert image.getpixel((10, 150)) != image.getpixel((1190, 150))


def test_image_uses_configured_backend_and_fits_viewport(monkeypatch) -> None:
    report = terminal.inspect_terminal(TerminalImageSettings(), json_output=True)
    report = replace(
        report, probe="completed", selected_renderer="test.Renderer", cell_width=11, cell_height=25
    )
    observed = []

    def render(settings, items):
        observed.append(settings)
        assert items[0].artifact.mime_type == "image/png"
        return "rendered chart"

    monkeypatch.setattr(terminal, "render_image_items", render)
    output = StringIO()
    terminal._show_image(
        Console(file=output, width=40, height=10),
        report,
        TerminalImageSettings(backend="halfcell"),
    )
    assert "rendered chart" in output.getvalue()
    assert observed[0].backend == "halfcell"
    assert observed[0].width <= 39
    assert observed[0].height <= 6
