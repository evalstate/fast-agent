#!/usr/bin/env python3
"""Capture command output as a terminal-style SVG for documentation."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.ansi import AnsiDecoder
from rich.console import Console
from rich.text import Text


def _run_with_pty(command: str, cwd: Path) -> str:
    script_bin = shutil.which("script")
    if script_bin is None:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return result.stdout

    with tempfile.NamedTemporaryFile(delete=False) as capture:
        capture_path = Path(capture.name)
    try:
        result = subprocess.run(
            [script_bin, "-q", "-e", "-c", command, str(capture_path)],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        output = capture_path.read_text(encoding="utf-8", errors="replace")
        if result.returncode != 0:
            output += f"\n[exit status {result.returncode}]\n"
        return output
    finally:
        capture_path.unlink(missing_ok=True)


def render_svg(command: str, output: str, *, title: str, width: int) -> str:
    console = Console(
        record=True,
        width=88,
        force_terminal=True,
        color_system="truecolor",
        file=open(os.devnull, "w", encoding="utf-8"),
    )
    decoder = AnsiDecoder()
    console.print(Text(f"$ {command}", style="bold #9CDCFE"))
    for segment in decoder.decode(output):
        console.print(segment, end="")
    svg = console.export_svg(title=title, theme=None, clear=True)
    # Rich exports scalable SVGs; setting a stable width keeps docs layout predictable.
    return svg.replace("<svg ", f'<svg width="{width}" ', 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="Command to run and capture.")
    parser.add_argument("--output", required=True, type=Path, help="SVG output path.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory.")
    parser.add_argument("--title", default="terminal capture", help="SVG title.")
    parser.add_argument("--width", type=int, default=960, help="Rendered SVG width.")
    args = parser.parse_args()

    output = _run_with_pty(args.command, args.cwd)
    svg = render_svg(args.command, output, title=args.title, width=args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Captured {shlex.quote(args.command)} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
