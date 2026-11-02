"""Local terminal diagnostics: no model calls, clipboard access or lifecycle writes."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import typer
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.table import Table

from fast_agent.config import TerminalImageSettings, get_settings
from fast_agent.integrations.program_status import program_status_enabled
from fast_agent.ui.terminal_images.probe import prepare_terminal_image_probe
from fast_agent.ui.terminal_images.renderer import (
    ImageArtifact,
    ImageRenderItem,
    _resolve_textual_image_class,
    render_image_items,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class TerminalCheck:
    fast_agent_version: str
    textual_image_version: str
    stdin_tty: bool
    stdout_tty: bool
    term: str | None
    term_program: str | None
    hc_terminal_id: str | None
    images_enabled: bool
    configured_backend: str
    probe: Literal["completed", "skipped", "failed"]
    probe_note: str
    reported_kitty: bool | None
    reported_sixel: bool | None
    cell_width: int | None
    cell_height: int | None
    selected_renderer: str | None
    status_reporting_eligible: bool
    model_image_support: str = "Not checked; depends on the model selected in your agent."
    clipboard: str = (
        "Ctrl+V in Herdr uploads an image to the terminal's machine; fast-agent attaches "
        "the pasted path for image-capable models. Alt+V reads this machine's clipboard. "
        "Neither clipboard access nor upload is tested by this command."
    )


def _interactive_terminal() -> bool:
    # textual-image uses original streams; do not send probes around redirection
    # or an embedded host that replaced only sys.stdout.
    return all(
        stream is not None and stream.isatty()
        for stream in (sys.stdin, sys.stdout, sys.__stdin__, sys.__stdout__)
    )


def inspect_terminal(settings: TerminalImageSettings, *, json_output: bool) -> TerminalCheck:
    state: Literal["completed", "skipped", "failed"] = "skipped"
    note = "Active probing requires interactive stdin/stdout and non-JSON output."
    kitty = sixel = None
    width = height = None
    renderer = None
    if not json_output and _interactive_terminal():
        try:
            prepare_terminal_image_probe()
            from textual_image._terminal import probe_terminal

            capabilities = probe_terminal()
            kitty, sixel = capabilities.tgp, capabilities.sixel
            width, height = capabilities.cell_size
            state = "completed"
            note = (
                "Capabilities reported by the terminal, not inferred from TERM_PROGRAM. "
                "Cell size may be the library fallback if the terminal did not supply one."
            )
            if settings.enabled and settings.backend != "none":
                image_cls = _resolve_textual_image_class(settings.backend)
                if image_cls is not None:
                    renderer = f"{image_cls.__module__}.{image_cls.__name__}"
        except (OSError, ValueError, RuntimeError) as exc:
            state = "failed"
            note = f"Terminal probe failed: {exc}"
    return TerminalCheck(
        fast_agent_version=version("fast-agent-mcp"),
        textual_image_version=version("textual-image"),
        stdin_tty=sys.stdin is not None and sys.stdin.isatty(),
        stdout_tty=sys.stdout is not None and sys.stdout.isatty(),
        term=os.environ.get("TERM"),
        term_program=os.environ.get("TERM_PROGRAM"),
        hc_terminal_id=os.environ.get("HC_TERMINAL_ID"),
        images_enabled=settings.enabled,
        configured_backend=settings.backend,
        probe=state,
        probe_note=note,
        reported_kitty=kitty,
        reported_sixel=sixel,
        cell_width=width,
        cell_height=height,
        selected_renderer=renderer,
        status_reporting_eligible=program_status_enabled(),
    )


def test_chart_png() -> bytes:
    """Portable high-frequency chart; Pillow's bundled font needs no system font."""
    image = Image.new("RGB", (1200, 240), "white")
    draw = ImageDraw.Draw(image)
    for y, size in ((2, 12), (22, 18), (50, 26), (86, 36)):
        draw.text(
            (8, y),
            f"{size}px  High resolution: ABC abc 0123456789",
            font=ImageFont.load_default(size=size),
            fill="black",
        )
    for x in range(600, 1200):
        draw.line((x, 0, x, 120), fill="black" if x % 2 else "white")
    for x in range(1200):
        draw.line((x, 135, x, 185), fill=(x * 255 // 1199, 120, 255 - x * 255 // 1199))
    for x in range(0, 1200, 4):
        draw.rectangle((x, 195, x + 1, 239), fill="black")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _show_image(console: Console, report: TerminalCheck, settings: TerminalImageSettings) -> None:
    if not settings.enabled or settings.backend == "none":
        raise typer.BadParameter("Image rendering is disabled in logger.terminal_images.")
    if report.probe != "completed" or report.selected_renderer is None:
        raise typer.BadParameter("No renderer was resolved; inspect the diagnostic above.")
    assert report.cell_width is not None and report.cell_height is not None
    max_rows = min(18, console.size.height - 4)
    if console.width < 10 or max_rows < 1:
        raise typer.BadParameter("Terminal is too small for the test image.")
    # Fit this chart, rather than assuming the configured preview size fits here.
    columns = min(120, console.width - 1, max_rows * report.cell_height * 5 // report.cell_width)
    rows = max(1, round(columns * report.cell_width / (5 * report.cell_height)))
    fitted = settings.model_copy(update={"width": columns, "height": rows})
    artifact = ImageArtifact(test_chart_png(), "image/png", "Terminal resolution test (1200 x 240)")
    image = render_image_items(fitted, [ImageRenderItem(artifact)])
    if image is None:
        raise typer.BadParameter("The image renderer could not load the test chart.")
    console.print(image)
    console.print(
        "Look for readable small text and fine stripes. Half-cell rendering is blocky; "
        "pixel rendering is detailed. Scaling can soften one-pixel lines.",
        markup=False,
    )


def run_terminal_check(*, home: Path | None, image: bool, json_output: bool) -> None:
    if image and json_output:
        raise typer.BadParameter("--image cannot be combined with --json.")
    if image and not _interactive_terminal():
        raise typer.BadParameter("--image requires interactive terminal stdin and stdout.")
    settings = get_settings(home=home).logger.terminal_images
    report = inspect_terminal(settings, json_output=json_output)
    if json_output:
        # Do not use Rich: stdout must be plain JSON, without probe escapes.
        typer.echo(json.dumps(asdict(report), indent=2))
        return
    console = Console()
    table = Table(title="Terminal integration", show_header=False)
    table.add_column("Check", style="cyan")
    table.add_column("Result")
    rows = [
        (
            "fast-agent / textual-image",
            f"{report.fast_agent_version} / {report.textual_image_version}",
        ),
        ("stdin / stdout TTY", f"{report.stdin_tty} / {report.stdout_tty}"),
        (
            "TERM / TERM_PROGRAM (hints)",
            f"{report.term or 'unset'} / {report.term_program or 'unset'}",
        ),
        ("Herdr core terminal", report.hc_terminal_id or "Not supplied"),
        ("Status reporting eligible", f"{report.status_reporting_eligible} (no status emitted)"),
        (
            "Configured images",
            f"enabled={report.images_enabled}, backend={report.configured_backend}",
        ),
        ("Capability probe", report.probe),
        ("Reported Kitty / Sixel", f"{report.reported_kitty} / {report.reported_sixel}"),
        (
            "Cell pixels used",
            f"{report.cell_width} x {report.cell_height}" if report.cell_width else "Not probed",
        ),
        ("Selected renderer", report.selected_renderer or "Not resolved / rendering disabled"),
    ]
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)
    console.print(report.probe_note, markup=False)
    console.print(report.clipboard, markup=False)
    console.print(f"Model image support: {report.model_image_support}", markup=False)
    if image:
        _show_image(console, report, settings)
    elif report.probe == "completed":
        console.print("Add --image for a visual resolution test.", markup=False)
    if report.probe == "failed":
        raise typer.Exit(1)
