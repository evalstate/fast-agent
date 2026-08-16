"""Compare Rich/Pygments error-token backgrounds used in shell previews.

Run the built-in example:
    uv run examples/syntax_error_background_demo.py

Render a captured shell command:
    uv run examples/syntax_error_background_demo.py captured-command.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Error
from rich.console import Console, Group, RenderableType
from rich.syntax import Syntax

from fast_agent.ui.syntax_highlighting import SyntaxBlock, shell_syntax_blocks

DEMO_COMMAND = """\
python - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
valid = {"root": str(root)}
partial_stream_expression = ???
provider_fragment = $unfinished
print(json.dumps(valid))
PY
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current and transparent Rich Syntax error backgrounds.",
    )
    parser.add_argument(
        "command_file",
        nargs="?",
        type=Path,
        help="file containing a shell command to render instead of the built-in sample",
    )
    parser.add_argument(
        "--theme",
        default="native",
        help="Pygments theme to compare (default: native)",
    )
    return parser.parse_args()


def _render_blocks(
    blocks: list[SyntaxBlock],
    *,
    theme: str,
    transparent: bool,
) -> RenderableType:
    background = {"background_color": "default"} if transparent else {}
    return Group(
        *(
            Syntax(
                block.code,
                block.language,
                theme=theme,
                line_numbers=False,
                word_wrap=True,
                **background,
            )
            for block in blocks
        )
    )


def _error_spans(block: SyntaxBlock) -> list[str]:
    lexer = get_lexer_by_name(block.language, stripnl=False, ensurenl=False)
    return [value for token, value in lex(block.code, lexer) if token in Error]


def main() -> None:
    args = _parse_args()
    command = args.command_file.read_text() if args.command_file is not None else DEMO_COMMAND
    blocks = shell_syntax_blocks(
        command,
        shell_language="bash",
        include_incomplete=True,
    )

    console = Console()
    console.rule("[bold]Before: current Syntax rendering")
    console.print(_render_blocks(blocks, theme=args.theme, transparent=False))
    console.rule('[bold]After: background_color="default"')
    console.print(_render_blocks(blocks, theme=args.theme, transparent=True))

    console.rule("[bold]Pygments error tokens")
    found_errors = False
    for block in blocks:
        errors = _error_spans(block)
        if not errors:
            continue
        found_errors = True
        console.print(f"[cyan]{block.language}[/cyan]: {errors!r}")
    if not found_errors:
        console.print("[green]No Token.Error spans found.[/green]")


if __name__ == "__main__":
    main()
