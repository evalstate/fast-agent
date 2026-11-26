#!/usr/bin/env python3
"""Visual + copy/paste + split-pane reflow demo for Markdown & code blocks.

Run in an interactive terminal, test drag-copy/paste on code lines, then narrow
the window or open a split pane to compare how each rendering reflows:

    uv run python scripts/syntax_reflow_demo.py
    uv run python scripts/syntax_reflow_demo.py --stream
    uv run python scripts/syntax_reflow_demo.py --wait
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.markdown import Markdown as StockMarkdown
from rich.text import Text

from fast_agent.ui.markdown.renderables import build_markdown_renderable
from fast_agent.ui.streaming.display import (
    STREAM_CURSOR_BLOCK,
    _CursorStyledRenderable,
    _DiffLive,
)

SAMPLE_MARKDOWN = """\
## 2. Unified semantic coloring

All three views share one mapping:

- **Tool name** (`search_models`): bold cobalt
  - Nested note inside bullet list
- **Status**: muted green on success, bold red on failure

> Quoted note that stays unpadded on the right edge.

---

```python
def connect_proxmox(host: str, api_token: str) -> ProxmoxClient:
    # Uses revocable API token stored in OpenBao instead of plaintext password
    endpoint = f"https://{host}:8006/api2/json"
    return ProxmoxClient(endpoint=endpoint, token=api_token, verify_tls=True)
```

Closing paragraph after fenced code block."""


def _stream_live_markdown(console: Console, *, theme: str) -> None:
    live = _DiffLive(console=console, transient=False)
    with live:
        step = 10
        for end in range(step, len(SAMPLE_MARKDOWN) + step, step):
            chunk = SAMPLE_MARKDOWN[:end]
            renderable = build_markdown_renderable(
                chunk,
                code_theme=theme,
                escape_xml=True,
                cursor_suffix=STREAM_CURSOR_BLOCK,
                close_incomplete_fences=True,
                render_fences_with_syntax=True,
                code_word_wrap=True,
            )
            live.update(
                _CursorStyledRenderable(renderable, visible=True, _segment_cache={}),
                refresh=True,
            )
            time.sleep(0.025)

        final_renderable = build_markdown_renderable(
            SAMPLE_MARKDOWN,
            code_theme=theme,
            escape_xml=True,
            render_fences_with_syntax=True,
            code_word_wrap=True,
        )
        live.update(final_renderable, refresh=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theme",
        default="ansi_light",
        help="Code theme for reflow-safe fast-agent render (default: ansi_light)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream section #2 live through _DiffLive (shows open-fence choreography)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for Enter before exiting so you can resize/split the pane",
    )
    args = parser.parse_args()

    console = Console()
    width = console.size.width

    console.print()
    console.print(
        Text(
            f"Rendered at {width} columns — split or narrow this pane after printing to compare reflow",
            style="bold cyan",
        )
    )
    console.print()

    console.print(
        Text(
            "1. STOCK Rich Markdown + monokai (right-pads prose/lists with spaces, full-width '---', padded code bg):",
            style="bold yellow",
        )
    )
    console.print(StockMarkdown(SAMPLE_MARKDOWN, code_theme="monokai"))
    console.print()

    console.print(
        Text(
            f"2. CURRENT dev/0.10.68 build_markdown_renderable (theme={args.theme!r} — unpadded prose/lists, "
            "36-cell rules, open-fence streaming, pure col-0 copy-paste):",
            style="bold green",
        )
    )
    if args.stream:
        _stream_live_markdown(console, theme=args.theme)
    else:
        console.print(
            build_markdown_renderable(
                SAMPLE_MARKDOWN,
                code_theme=args.theme,
                escape_xml=True,
                render_fences_with_syntax=True,
                code_word_wrap=True,
            )
        )
    console.print()

    if args.wait:
        try:
            input("Press Enter to exit…")
        except (EOFError, KeyboardInterrupt):
            console.print()


if __name__ == "__main__":
    main()
