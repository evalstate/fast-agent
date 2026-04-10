import io

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.ui.markdown_renderables import (
    _rewrite_fence_languages,
    build_markdown_renderable,
    extract_single_fenced_code_block,
)


def _find_first_link_href(markdown: Markdown) -> str | None:
    def iter_tokens(tokens):
        for token in tokens:
            yield token
            children = getattr(token, "children", None) or ()
            yield from iter_tokens(children)

    for token in iter_tokens(markdown.parsed):
        if token.type == "link_open":
            href = token.attrs.get("href")
            return None if href is None else str(href)
    return None


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


def test_build_markdown_renderable_can_wrap_syntax_code() -> None:
    renderable = build_markdown_renderable(
        "```python\nprint('this is a very long line that should wrap when enabled')\n```",
        code_theme="monokai",
        escape_xml=True,
        code_word_wrap=True,
    )

    assert isinstance(renderable, Syntax)
    assert renderable.word_wrap is True


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

    assert isinstance(renderable, Group)
    assert isinstance(renderable.renderables[0], Markdown)
    assert isinstance(renderable.renderables[1], Syntax)


def test_build_markdown_renderable_keeps_reference_links_across_mixed_fences() -> None:
    renderable = build_markdown_renderable(
        "See [docs][r]\n\n```python\nprint(1)\n```\n\n[r]: https://example.com\n",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Group)
    assert len(renderable.renderables) == 2
    assert isinstance(renderable.renderables[0], Markdown)
    assert isinstance(renderable.renderables[1], Syntax)
    assert _find_first_link_href(renderable.renderables[0]) == "https://example.com"


def test_build_markdown_renderable_keeps_reference_links_when_cursor_appends_to_tail() -> None:
    renderable = build_markdown_renderable(
        "```python\nprint(1)\n```\n\nSee [docs][r]\n\n[r]: https://example.com\n",
        code_theme="monokai",
        escape_xml=True,
        cursor_suffix="|",
    )

    assert isinstance(renderable, Group)
    assert len(renderable.renderables) == 2
    assert isinstance(renderable.renderables[0], Syntax)
    assert isinstance(renderable.renderables[1], Markdown)
    assert _find_first_link_href(renderable.renderables[1]) == "https://example.com"


def test_build_markdown_renderable_can_disable_code_block_padding() -> None:
    renderable = build_markdown_renderable(
        "```bash\necho hi\n```",
        code_theme="monokai",
        escape_xml=True,
        pad_code_blocks=False,
    )

    assert isinstance(renderable, Syntax)
    assert renderable.code == "echo hi"


def test_build_markdown_renderable_can_disable_syntax_split() -> None:
    renderable = build_markdown_renderable(
        "Run this:\n\n```python\nprint(1)\n```",
        code_theme="monokai",
        escape_xml=True,
        render_fences_with_syntax=False,
    )

    assert isinstance(renderable, Markdown)


def test_build_markdown_renderable_mixed_apply_patch_keeps_preview_styling() -> None:
    renderable = build_markdown_renderable(
        "Patch follows:\n\n```apply_patch\n*** Begin Patch\n@@\n-old\n+new\n```\n",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Group)
    assert isinstance(renderable.renderables[0], Markdown)
    assert isinstance(renderable.renderables[1], Text)


def test_build_markdown_renderable_closes_incomplete_mixed_fence_for_streaming() -> None:
    renderable = build_markdown_renderable(
        "Before:\n\n```python\nprint('hi')",
        code_theme="monokai",
        escape_xml=True,
        close_incomplete_fences=True,
    )

    assert isinstance(renderable, Group)
    assert isinstance(renderable.renderables[0], Markdown)
    assert isinstance(renderable.renderables[1], Syntax)


def test_build_markdown_renderable_preserves_empty_fenced_block_in_mixed_content() -> None:
    renderable = build_markdown_renderable(
        "Before\n\n```\n```\n\nAfter",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Group)
    assert len(renderable.renderables) == 3
    assert isinstance(renderable.renderables[0], Markdown)
    assert isinstance(renderable.renderables[1], Syntax)
    assert isinstance(renderable.renderables[2], Markdown)


def test_build_markdown_renderable_skips_nested_fences_without_disabling_top_level_syntax() -> None:
    renderable = build_markdown_renderable(
        "Top level:\n\n```python\nprint(1)\n```\n\n- step:\n  ```bash\n  echo hi\n  ```\n",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Group)
    assert len(renderable.renderables) == 3
    assert isinstance(renderable.renderables[0], Markdown)
    assert isinstance(renderable.renderables[1], Syntax)
    assert isinstance(renderable.renderables[2], Markdown)


def test_extract_single_fenced_code_block_handles_incomplete_stream() -> None:
    block = extract_single_fenced_code_block("```python\nprint('hi')")

    assert block is not None
    assert block.language == "python"
    assert block.code == "print('hi')"
    assert block.complete is False
