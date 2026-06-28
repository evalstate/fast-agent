from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.theme import Theme

SAMPLE = """
## Yellow heading soft

Paragraph with `inline code`, `fast-agent demo markdown`, and `--theme-file`.

> Blockquote text should feel like an aside, not a warning.

- bullet with `item-code`
- bullet with `uv run`

""".strip()

BASE_STYLES = {
    "markdown.h2": "bright_yellow underline",
    "markdown.h3": "bold bright_yellow",
    "markdown.block_quote": "italic dim",
    "markdown.link": "bright_cyan",
    "markdown.link_url": "bright_cyan underline",
    "markdown.table.border": "yellow",
    "markdown.table.header": "bright_yellow",
    "markdown.hr": "yellow dim",
}

INLINE_CODE_VARIANTS = {
    "reverse": "reverse",
    "bold-reverse": "bold reverse",
    "yellow": "yellow",
    "bold-yellow": "bold yellow",
    "cyan": "cyan",
}


def main() -> None:
    for name, inline_code_style in INLINE_CODE_VARIANTS.items():
        styles = {**BASE_STYLES, "markdown.code": inline_code_style}
        console = Console(theme=Theme(styles))
        console.print(Rule(name))
        console.print(Markdown(SAMPLE, code_theme="native"))
        console.print()


if __name__ == "__main__":
    main()
