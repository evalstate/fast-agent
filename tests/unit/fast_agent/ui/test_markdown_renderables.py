import io

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.ui.markdown_renderables import (
    _rewrite_fence_languages,
    build_markdown_renderable,
    extract_single_fenced_code_block,
)


def test_build_markdown_renderable_uses_syntax_for_code_only_fence() -> None:
    renderable = build_markdown_renderable(
        "```bash\necho hi\n```",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Syntax)

    output = io.StringIO()
    Console(file=output, force_terminal=False, width=40).print(renderable)
    rendered = output.getvalue().splitlines()
    assert any(line.startswith("echo hi") for line in rendered)


def test_build_markdown_renderable_normalizes_cmd_fence_language() -> None:
    renderable = build_markdown_renderable(
        "```cmd\ndir\n```",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Syntax)
    assert renderable._lexer == "batch"


def test_build_markdown_renderable_styles_apply_patch_fence() -> None:
    renderable = build_markdown_renderable(
        "```apply_patch\n*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new\n```",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Text)
    span_styles = {str(span.style) for span in renderable.spans}
    assert "cyan" in span_styles
    assert "yellow" in span_styles
    assert "red" in span_styles
    assert "green" in span_styles


def test_rewrite_fence_languages_normalizes_apply_patch_for_markdown() -> None:
    rewritten = _rewrite_fence_languages(
        "Patch:\n\n```apply_patch\n*** Begin Patch\n@@\n-old\n+new\n```"
    )

    assert rewritten == "Patch:\n\n```diff\n*** Begin Patch\n@@\n-old\n+new\n```"


def test_rewrite_fence_languages_does_not_touch_literal_nested_fences() -> None:
    markdown = "Example:\n\n````markdown\n```cmd\ndir\n```\n````"

    assert _rewrite_fence_languages(markdown) == markdown


def test_build_markdown_renderable_keeps_mixed_markdown_as_markdown() -> None:
    renderable = build_markdown_renderable(
        "Run this:\n\n```python\nprint(1)\n```",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Markdown)


def test_extract_single_fenced_code_block_handles_incomplete_stream() -> None:
    block = extract_single_fenced_code_block("```python\nprint('hi')")

    assert block is not None
    assert block.language == "python"
    assert block.code == "print('hi')"
    assert block.complete is False
