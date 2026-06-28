from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

SAMPLE = """
# High contrast preview

Paragraph with `inline code`, `fast-agent demo markdown`, `--theme-file`, and `uv run`.

> Blockquote text should feel distinct and readable without shouting over the heading.

- bullet with `item-code`
- another bullet with `--wrap-code`

| Flag | Meaning |
| --- | --- |
| `--theme-file` | Use a Rich theme file |
| `--wrap-code` | Wrap syntax-rendered code |
""".strip()

BASE_STYLES = {
    "markdown.h1": "bold bright_white underline",
    "markdown.h2": "bold bright_cyan underline",
    "markdown.h3": "bold bright_cyan",
    "markdown.h4": "bright_cyan",
    "markdown.h5": "bold white",
    "markdown.h6": "white dim",
    "markdown.link": "bright_cyan underline",
    "markdown.link_url": "bright_blue underline",
    "markdown.hr": "bright_white dim",
    "markdown.table.border": "bright_white",
    "markdown.table.header": "bold bright_white",
}

VARIANTS = {
    "current": {
        "markdown.block_quote": "bright_blue",
        "markdown.code": "bold bright_green on black",
    },
    "soft-green": {
        "markdown.block_quote": "blue",
        "markdown.code": "bright_green on black",
    },
    "reverse-green": {
        "markdown.block_quote": "italic bright_blue",
        "markdown.code": "bold bright_green reverse",
    },
    "cool-cyan": {
        "markdown.block_quote": "bright_blue",
        "markdown.code": "bold bright_cyan on black",
    },
    "neutral-white": {
        "markdown.block_quote": "bright_blue",
        "markdown.code": "bold bright_white on black",
    },
}


def main() -> None:
    for name, overrides in VARIANTS.items():
        styles = {**BASE_STYLES, **overrides}
        console = Console(theme=Theme(styles))
        console.print(Rule(name))
        console.print(
            Text(
                f"blockquote={overrides['markdown.block_quote']} | "
                f"inline_code={overrides['markdown.code']}",
                style="dim",
            )
        )
        console.print(Markdown(SAMPLE, code_theme="native"))
        console.print()


if __name__ == "__main__":
    main()
