import io

import pytest
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.command_actions import MarkdownTextStyle
from fast_agent.ui.markdown.content import prepare_markdown_content
from fast_agent.ui.markdown.renderables import (
    _rewrite_fence_languages,
    build_markdown_renderable,
    extract_single_fenced_code_block,
    style_markdown_renderable,
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
    assert renderable.word_wrap is True

    output = io.StringIO()
    Console(file=output, force_terminal=False, width=40).print(renderable)
    rendered = output.getvalue().splitlines()
    assert any(line.startswith("echo hi") for line in rendered)


def test_build_markdown_renderable_normalizes_cmd_fence_language() -> None:
    renderable = build_markdown_renderable(
        "``` CMD \ndir\n```",
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
        "``` APPLY_PATCH \n*** Begin Patch\n*** Update File: a.txt\n@@\n context\n-old\n+new\n```",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Text)
    span_styles = {str(span.style) for span in renderable.spans}
    assert "dim" in span_styles
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


def test_prepare_markdown_content_preserves_blockquote_markers() -> None:
    prepared = prepare_markdown_content("> quoted <tag>\n\nplain > text")

    assert prepared.startswith("> quoted &lt;tag&gt;")
    assert "plain &gt; text" in prepared


def test_build_markdown_renderable_renders_blockquote_prefix() -> None:
    renderable = build_markdown_renderable(
        "> quoted <tag>",
        code_theme="monokai",
        escape_xml=True,
    )

    output = io.StringIO()
    Console(file=output, force_terminal=False, width=20).print(renderable)

    assert "▌ quoted <tag>" in output.getvalue()


def test_build_markdown_renderable_adds_newline_after_final_table_row() -> None:
    renderable = build_markdown_renderable(
        "| Name | Value |\n| --- | --- |\n| one | two |",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Markdown)
    assert renderable.markup.endswith("| one | two |\n")


def test_style_markdown_renderable_colors_only_selected_table_text() -> None:
    source = "| Cached |\n| ---: |\n| 400,000 (40%) |\n| 500,000 (50%) |\n"
    markdown = build_markdown_renderable(
        source,
        code_theme="monokai",
        escape_xml=True,
    )
    renderable = style_markdown_renderable(
        markdown,
        (MarkdownTextStyle(text="40%", style="red"),),
    )
    console = Console(force_terminal=True, color_system="standard", width=40)
    segments = tuple(console.render(renderable, console.options))

    low_cache = next(segment for segment in segments if segment.text == "40%")
    threshold = next(segment for segment in segments if "50%" in segment.text)
    assert low_cache.style is not None
    assert low_cache.style.color is not None
    assert low_cache.style.color.name == "red"
    assert threshold.style is None or threshold.style.color is None
    assert isinstance(markdown, Markdown)
    assert "[red]" not in markdown.markup
    assert "\x1b" not in markdown.markup


def test_style_markdown_renderable_matches_across_segments_and_preserves_styles() -> None:
    text = Text()
    text.append("4", style="bold")
    text.append("0%", style="italic")
    renderable = style_markdown_renderable(
        text,
        (MarkdownTextStyle(text="40%", style="red"),),
    )
    console = Console(force_terminal=True, color_system="standard", width=40)
    segments = tuple(console.render(renderable, console.options))
    styled = [segment for segment in segments if segment.text in {"4", "0%"}]
    styles = [segment.style for segment in styled if segment.style is not None]

    assert len(styles) == 2
    assert all(style.color is not None and style.color.name == "red" for style in styles)
    assert styles[0].bold is True
    assert styles[1].italic is True


def test_style_markdown_renderable_does_not_match_numeric_suffixes() -> None:
    markdown = build_markdown_renderable(
        "| Cache read |\n| ---: |\n| 4,290,048 (89%) |\n| 0 (0%) |\n| 11,096,064 (90%) |\n",
        code_theme="monokai",
        escape_xml=True,
    )
    renderable = style_markdown_renderable(
        markdown,
        (MarkdownTextStyle(text="0%", style="red"),),
    )
    console = Console(force_terminal=True, color_system="standard", width=40)
    segments = tuple(console.render(renderable, console.options))
    red_text = "".join(
        segment.text
        for segment in segments
        if segment.style is not None
        and segment.style.color is not None
        and segment.style.color.name == "red"
    )

    assert red_text == "0%"


def test_build_markdown_renderable_adds_newline_after_optional_pipe_table_row() -> None:
    renderable = build_markdown_renderable(
        "Name | Value\n--- | ---\none | two |",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Markdown)
    assert renderable.markup.endswith("one | two |\n")


def test_build_markdown_renderable_leaves_non_table_trailing_pipe() -> None:
    renderable = build_markdown_renderable(
        "not a table |",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Markdown)
    assert renderable.markup.endswith("not a table |")


def test_build_markdown_renderable_does_not_treat_cursor_suffix_as_table_pipe() -> None:
    renderable = build_markdown_renderable(
        "| Name | Value |\n| --- | --- |\n| one | two",
        code_theme="monokai",
        escape_xml=True,
        cursor_suffix="|",
    )

    assert isinstance(renderable, Markdown)
    assert renderable.markup.endswith("| one | two|")


def test_extract_single_fenced_code_block_handles_incomplete_stream() -> None:
    block = extract_single_fenced_code_block("```python\nprint('hi')")

    assert block is not None
    assert block.language == "python"
    assert block.code == "print('hi')"
    assert block.complete is False


def test_build_markdown_renderable_frames_ansi_code_blocks_with_capped_rules() -> None:
    renderable = build_markdown_renderable(
        "Before\n\n```python\nx = 1\n```\n\nAfter",
        code_theme="ansi_light",
        escape_xml=True,
    )
    console = Console(width=100, force_terminal=True, color_system="truecolor")
    with console.capture() as capture:
        console.print(renderable)

    lines = [line for line in capture.get().splitlines() if line.strip()]
    header = next(line for line in lines if "python" in line)
    footer = lines[-2]
    assert "──" in header
    assert Text.from_ansi(header).cell_len == 36
    assert Text.from_ansi(footer).cell_len == 36
    assert Text.from_ansi(footer).plain == "─" * 36

    unlabelled = build_markdown_renderable(
        "```\nplain line\n```",
        code_theme="ansi_light",
        escape_xml=True,
    )
    with console.capture() as capture:
        console.print(unlabelled)
    unlabelled_lines = [
        Text.from_ansi(line).plain for line in capture.get().splitlines() if line.strip()
    ]
    assert unlabelled_lines[0] == "─" * 36
    assert "text" not in unlabelled_lines[0]


@pytest.mark.parametrize("code_theme", ["ansi_light", "monokai"])
def test_build_markdown_renderable_omits_trailing_spaces_and_caps_hr_for_reflow(
    code_theme: str,
) -> None:
    sample = (
        "# Top Heading\n\n"
        "## 2. Unified semantic coloring\n\n"
        "All three views share one mapping:\n\n"
        "- **Tool name**: bold cobalt\n"
        "  - Nested item\n\n"
        "    Second paragraph inside bullet.\n"
        "1. First ordered item\n"
        "2. Second ordered item\n\n"
        "> Quoted line one\n"
        ">\n"
        "> Quoted line two\n\n"
        "---\n\n"
        "Closing paragraph."
    )
    renderable = build_markdown_renderable(
        sample,
        code_theme=code_theme,
        escape_xml=True,
    )
    buffer_console = Console(width=100, force_terminal=True, color_system="truecolor")
    lines = buffer_console.render_lines(renderable, buffer_console.options, pad=False)
    plain_lines = ["".join(segment.text for segment in line) for line in lines]

    for plain in plain_lines:
        assert plain == plain.rstrip(" "), f"unexpected trailing space in {plain!r}"

    with buffer_console.capture() as capture:
        buffer_console.print(renderable)
    printed_lines = [Text.from_ansi(line).plain for line in capture.get().splitlines()]
    for plain in printed_lines:
        assert plain == plain.rstrip(" "), f"unexpected trailing space in printed {plain!r}"

    assert "Top Heading" in printed_lines
    assert "─" * 36 in printed_lines
    assert "-" * 100 not in printed_lines


def test_live_incomplete_fences_omit_bottom_rule_and_keep_single_clean_cursor() -> None:
    from fast_agent.ui.markdown.truncation import MarkdownTruncator
    from fast_agent.ui.streaming.display import STREAM_CURSOR_BLOCK, _CursorStyledRenderable

    console = Console(width=80, force_terminal=True, color_system="truecolor")

    def render_plain(markup: str, *, theme: str = "ansi_light", cursor: bool = True) -> list[str]:
        renderable = build_markdown_renderable(
            markup,
            code_theme=theme,
            escape_xml=True,
            cursor_suffix=STREAM_CURSOR_BLOCK if cursor else "",
            close_incomplete_fences=True,
        )
        if cursor:
            renderable = _CursorStyledRenderable(renderable, visible=True, _segment_cache={})
        return [
            "".join(seg.text for seg in line)
            for line in console.render_lines(renderable, console.options, pad=False)
        ]

    # Open top-level fence: header pill present, NO bottom rule, trailing newline moves cursor to next row
    open_no_nl = render_plain("Intro\n\n```python\nx = 1")
    assert any("python" in line and "──" in line for line in open_no_nl)
    assert open_no_nl[-1] == f"x = 1{STREAM_CURSOR_BLOCK}"
    assert "─" * 36 not in open_no_nl

    open_with_nl = render_plain("Intro\n\n```python\nx = 1\n")
    assert open_with_nl[-2:] == ["x = 1", STREAM_CURSOR_BLOCK]
    assert "─" * 36 not in open_with_nl

    # Closed fence: bottom rule appears, live cursor moves OUTSIDE the framed block
    closed = render_plain("Intro\n\n```python\nx = 1\n```")
    assert closed[-2] == "─" * 36
    assert closed[-1] == STREAM_CURSOR_BLOCK

    # Nested list & blockquote open fences: single cursor, no duplicate block or bouncing closer
    nested_list = render_plain("- item\n\n  ```python\n  x = 1")
    assert sum(line.count(STREAM_CURSOR_BLOCK) for line in nested_list) == 1
    assert nested_list[-1].endswith(f"x = 1{STREAM_CURSOR_BLOCK}")
    assert not any(line.strip() == "─" * 33 for line in nested_list)

    nested_quote = render_plain("> ```python\n> a")
    assert sum(line.count(STREAM_CURSOR_BLOCK) for line in nested_quote) == 1
    assert nested_quote[-1] == f"▌ a{STREAM_CURSOR_BLOCK}"

    # Exiting a blockquote without an explicit closing fence closes the block and moves cursor out
    exited_quote_blank = render_plain("> ```python\n> a\n\n")
    assert sum(line.count(STREAM_CURSOR_BLOCK) for line in exited_quote_blank) == 1
    assert any(line == "▌ " + "─" * 36 for line in exited_quote_blank)
    assert exited_quote_blank[-1] == STREAM_CURSOR_BLOCK

    exited_quote_prose = render_plain("> ```python\n> a\n\noutside")
    assert sum(line.count(STREAM_CURSOR_BLOCK) for line in exited_quote_prose) == 1
    assert any(line == "▌ " + "─" * 36 for line in exited_quote_prose)
    assert exited_quote_prose[-1] == f"outside{STREAM_CURSOR_BLOCK}"

    # Under ansi_dark, cursor must not inherit Pygments red/underline error styling
    dark_renderable = _CursorStyledRenderable(
        build_markdown_renderable(
            "```python\nx = 1",
            code_theme="ansi_dark",
            escape_xml=True,
            cursor_suffix=STREAM_CURSOR_BLOCK,
            close_incomplete_fences=True,
        ),
        visible=True,
        _segment_cache={},
    )
    cursor_segments = [
        seg
        for seg in console.render(dark_renderable, console.options)
        if STREAM_CURSOR_BLOCK in seg.text
    ]
    assert len(cursor_segments) == 1
    assert cursor_segments[0].style is not None
    assert not cursor_segments[0].style.underline

    # Truncator height measurement matches open vs closed rule count
    truncator = MarkdownTruncator()
    open_h = truncator.measure_rendered_height("```python\nx = 1", console, code_theme="ansi_light")
    closed_h = truncator.measure_rendered_height(
        "```python\nx = 1\n```", console, code_theme="ansi_light"
    )
    assert closed_h == open_h + 1


def test_soft_wrap_keeps_markdown_blocks_unbroken_and_bypasses_streaming_cache() -> None:
    import io
    import re

    from fast_agent.ui.streaming.display import _CachedStaticRenderable

    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    heading = "A long heading that exceeds sixty columns and should stay on one unbroken line"
    para = (
        "This is a long prose paragraph that exceeds sixty columns and should stay on one "
        "unbroken logical line so copying never injects unwanted newlines."
    )
    quote = "A long blockquote line that exceeds sixty columns and should stay unbroken on copy."
    item = (
        "Item one with plenty of words that wrap cleanly inside a list item without clipping "
        "or injecting newlines."
    )
    code = "echo hello && curl -s https://example.com/very/long/path/that/exceeds/sixty/columns/cleanly"
    nested_code = "export TF_VAR_proxmox_api_token='terraform@pam!homelab=<your-token-uuid>'"
    md = (
        f"## {heading}\n\n"
        f"{para}\n\n"
        f"> {quote}\n\n"
        f"1. {item}\n\n"
        f"   ```bash\n   {nested_code}\n   ```\n\n"
        f"```bash\n{code}\n```\n"
    )

    renderable = build_markdown_renderable(md, code_theme="native", escape_xml=True)
    cached = _CachedStaticRenderable(renderable=renderable, _segment_cache={})

    buf = io.StringIO()
    console = Console(file=buf, width=60, force_terminal=True, color_system="truecolor")

    # Populate streaming cache first (hard-wrapped to width 60 for live cursor math).
    console.render_lines(cached, options=console.options.update(width=60), pad=False)

    # Final print with soft_wrap=True must bypass the wrapped streaming cache.
    console.print(cached, soft_wrap=True)

    plain_lines = ansi_re.sub("", buf.getvalue()).splitlines()
    assert heading in plain_lines
    assert para in plain_lines
    assert f"▌ {quote}" in plain_lines
    assert f" 1 {item}" in plain_lines
    assert f"   {nested_code}" in plain_lines
    assert code in plain_lines


def test_soft_wrap_preserves_markdown_table_layout() -> None:
    import io

    md = (
        "| Col A | Col B |\n"
        "| --- | --- |\n"
        "| short | a long table cell value that should stay wrapped inside its column |\n\n"
        "A long trailing paragraph that exceeds fifty columns and should stay unbroken."
    )
    renderable = build_markdown_renderable(md, code_theme="native", escape_xml=True)
    buf = io.StringIO()
    console = Console(file=buf, width=50, force_terminal=False)
    console.print(renderable, soft_wrap=True)

    lines = buf.getvalue().splitlines()
    assert "A long trailing paragraph that exceeds fifty columns and should stay unbroken." in lines
    # Table rows stay within the console width rather than spilling unbroken past 50 columns.
    table_lines = [line for line in lines if "Col A" in line or "short" in line]
    assert table_lines
    assert all(len(line) <= 50 for line in table_lines)


def test_reflow_safe_syntax_omits_right_edge_padding_and_preserves_theme_and_whitespace() -> None:
    import io
    import re

    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    code = "x = 1  \n\tprint('hi')"
    renderable = build_markdown_renderable(
        f"```python\n{code}\n```\n",
        code_theme="native",
        escape_xml=True,
    )

    # Streaming mode (hard-wrapped to width 60): no right-edge padding, theme bg kept on tokens,
    # intentional trailing spaces and tab expansion preserved.
    stream_buf = io.StringIO()
    stream_console = Console(
        file=stream_buf, width=60, force_terminal=True, color_system="truecolor"
    )
    stream_console.print(renderable)
    stream_raw = stream_buf.getvalue()
    assert "\x1b[" in stream_raw
    # Theme background ANSI SGR (48;...) is preserved on code tokens, never padded to width 60.
    assert "48;" in stream_raw
    stream_plain = ansi_re.sub("", stream_raw).splitlines()
    assert "x = 1  " in stream_plain
    assert "    print('hi')" in stream_plain
    assert all(len(line) < 60 for line in stream_plain)


def test_diff_live_stop_clears_drawn_frame_and_reprints_soft_wrapped() -> None:
    import io
    import re

    from fast_agent.ui.streaming.display import _DiffLive

    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    para = (
        "Streaming output paragraph that is definitely longer than forty columns so live "
        "frames wrap it and final stop reprints it unbroken."
    )
    renderable = build_markdown_renderable(para, code_theme="native", escape_xml=True)

    buf = io.StringIO()
    console = Console(file=buf, width=40, force_terminal=True, color_system="truecolor")
    live = _DiffLive(renderable, console=console, transient=False)
    cleared: list[bool] = []
    live._clear_region = lambda: cleared.append(True)  # type: ignore[method-assign]
    live._stop_drawn_frame()

    assert cleared == [True]
    plain_lines = ansi_re.sub("", buf.getvalue()).splitlines()
    assert para in plain_lines

    # Transient mode clears without reprinting.
    buf_t = io.StringIO()
    console_t = Console(file=buf_t, width=40, force_terminal=True, color_system="truecolor")
    live_t = _DiffLive(renderable, console=console_t, transient=True)
    cleared_t: list[bool] = []
    live_t._clear_region = lambda: cleared_t.append(True)  # type: ignore[method-assign]
    live_t._stop_drawn_frame()

    assert cleared_t == [True]
    assert buf_t.getvalue() == ""
