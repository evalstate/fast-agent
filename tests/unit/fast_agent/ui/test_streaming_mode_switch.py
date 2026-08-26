import asyncio
import io
from typing import Any, cast

import pytest
from rich.console import Console, Group
from rich.live import Live
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.config import Settings
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.types.streaming import StreamingMode
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay, _StreamingMessageHandle
from fast_agent.ui.markdown.renderables import close_incomplete_code_blocks
from fast_agent.ui.streaming import display as streaming_module
from fast_agent.ui.streaming.segments import (
    StreamSegment,
    StreamSegmentAssembler,
    ToolCodePreview,
)


def _set_console_size(width: int = 80, height: int = 24) -> Console:
    original_console = console.console
    console.console = Console(
        file=io.StringIO(),
        force_terminal=True,
        width=width,
        height=height,
    )
    return original_console


def _restore_console_size(original_console: Console) -> None:
    console.console = original_console


def test_stream_rollback_discards_uncommitted_attempt() -> None:
    assembler = StreamSegmentAssembler(base_kind="plain", tool_prefix="")
    assembler.handle_stream_chunk(StreamChunk("partial answer"))
    assembler.handle_tool_event(
        "start",
        {"tool_name": "fetch", "tool_use_id": "call_partial"},
    )

    assert assembler.segments
    assert assembler.handle_stream_chunk(StreamChunk(event="rollback"))
    assert assembler.segments == []
    assert not assembler.has_pending_content()

    assembler.handle_stream_chunk(StreamChunk("recovered answer"))

    assert [segment.text for segment in assembler.segments] == ["recovered answer"]


def test_stream_batch_coalesces_contiguous_deltas_for_the_same_tool_call() -> None:
    first = streaming_module._ToolStreamEvent(
        event_type="delta",
        info={"tool_name": "search", "tool_use_id": "call-1", "chunk": '{"query":"'},
    )
    second = streaming_module._ToolStreamEvent(
        event_type="delta",
        info={"tool_name": "search", "tool_use_id": "call-1", "chunk": 'value"}'},
    )
    stop = streaming_module._ToolStreamEvent(
        event_type="stop",
        info={"tool_name": "search", "tool_use_id": "call-1"},
    )

    coalesced = streaming_module._coalesce_tool_delta_payloads([first, second, stop])

    assert coalesced == [
        streaming_module._ToolStreamEvent(
            event_type="delta",
            info={
                "tool_name": "search",
                "tool_use_id": "call-1",
                "chunk": '{"query":"value"}',
            },
        ),
        stop,
    ]


def test_stream_rollback_restores_committed_text_and_tool_segments() -> None:
    assembler = StreamSegmentAssembler(base_kind="plain", tool_prefix="->")
    assembler.handle_stream_chunk(StreamChunk("before tool"))
    assembler.handle_tool_event(
        "start",
        {
            "tool_name": "fetch",
            "tool_use_id": "call_committed",
            "chunk": '{"query":"status"}',
        },
    )
    assembler.handle_tool_event(
        "stop",
        {"tool_name": "fetch", "tool_use_id": "call_committed"},
    )
    assembler.handle_stream_chunk(StreamChunk(event="commit"))
    committed_segments = [
        (segment.kind, segment.text, segment.tool_completed) for segment in assembler.segments
    ]

    assembler.handle_stream_chunk(StreamChunk("failed follow-up"))
    assembler.handle_tool_event(
        "start",
        {"tool_name": "discard", "tool_use_id": "call_failed"},
    )
    assert assembler.handle_stream_chunk(StreamChunk(event="rollback"))

    assert [
        (segment.kind, segment.text, segment.tool_completed) for segment in assembler.segments
    ] == committed_segments
    assert "before tool" in "".join(segment.text for segment in assembler.segments)
    assert "failed follow-up" not in "".join(segment.text for segment in assembler.segments)
    assert all(segment.tool_use_id != "call_failed" for segment in assembler.segments)


def _make_handle(
    streaming_mode: StreamingMode = "markdown",
) -> _StreamingMessageHandle:
    settings = Settings()
    settings.logger.streaming = streaming_mode
    display = ConsoleDisplay(settings)
    return _StreamingMessageHandle(
        display=display,
        use_plain_text=streaming_mode == "plain",
        header_left="",
        header_right="",
        progress_display=None,
    )


def test_reasoning_stream_switches_back_to_markdown() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Intro"))
    assembler.handle_stream_chunk(StreamChunk("Thinking", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer"))

    text = "".join(segment.text for segment in assembler.segments)
    intro_idx = text.find("Intro")
    thinking_idx = text.find("Thinking")
    answer_idx = text.find("Answer")
    assert intro_idx != -1
    assert thinking_idx != -1
    assert answer_idx != -1
    assert "\n" in text[intro_idx + len("Intro") : thinking_idx]
    assert "\n\n" in text[thinking_idx + len("Thinking") : answer_idx]


def test_reasoning_stream_handles_multiple_blocks() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Think1", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer1"))
    assembler.handle_stream_chunk(StreamChunk("Think2", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer2"))

    text = "".join(segment.text for segment in assembler.segments)
    assert "Think1" in text
    assert "Answer1" in text
    assert "Think2" in text
    assert "Answer2" in text


def test_reasoning_gap_merges_into_markdown_segment() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Intro"))
    assembler.handle_stream_chunk(StreamChunk("Thinking", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer"))

    segments = assembler.segments
    assert [segment.kind for segment in segments] == ["markdown", "reasoning", "markdown"]
    assert segments[-1].text.startswith("\n\n")


def test_reasoning_tag_split_across_stream_chunks_switches_segments() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Intro <thi"))
    assembler.handle_stream_chunk(StreamChunk("nk>Thinking</thi"))
    assembler.handle_stream_chunk(StreamChunk("nk>Answer"))

    assert [(segment.kind, segment.text) for segment in assembler.segments] == [
        ("markdown", "Intro \n"),
        ("reasoning", "Thinking"),
        ("markdown", "\n\nAnswer"),
    ]


def test_incomplete_reasoning_tag_flushes_as_content() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Intro <thi"))
    assert assembler.flush()

    assert [(segment.kind, segment.text) for segment in assembler.segments] == [
        ("markdown", "Intro <thi"),
    ]


def test_streaming_live_starts_without_initial_renderable() -> None:
    handle = _make_handle("markdown")
    assert handle._live is not None
    assert getattr(handle._live, "_renderable", "missing") is None


def test_alt_screen_streaming_uses_rich_live(monkeypatch) -> None:
    monkeypatch.setenv("FAST_AGENT_STREAM_ALT_SCREEN", "1")
    handle = _make_handle("markdown")

    assert type(handle._live).__name__ == "Live"
    assert handle._alt_screen_streaming is True


def test_alt_screen_streaming_disables_preserve_final_frame(monkeypatch) -> None:
    monkeypatch.setenv("FAST_AGENT_STREAM_ALT_SCREEN", "1")
    handle = _make_handle("markdown")
    handle.update("hello")

    assert handle.preserve_final_frame() is False


def test_markdown_stream_uses_one_column_safety_gutter() -> None:
    handle = _make_handle("markdown")
    original_console = _set_console_size(width=80, height=24)
    try:
        assert handle._effective_stream_width() == 79
    finally:
        _restore_console_size(original_console)


def test_plain_stream_does_not_use_width_gutter() -> None:
    handle = _make_handle("plain")
    original_console = _set_console_size(width=80, height=24)
    try:
        assert handle._effective_stream_width() is None
    finally:
        _restore_console_size(original_console)


def test_console_width_override_restores_existing_width() -> None:
    class TargetConsole:
        _width = 100

    missing = object()
    target = TargetConsole()
    original = streaming_module._apply_console_width_override(target, 79)

    assert getattr(target, "_width", missing) == 79

    streaming_module._restore_console_width_override(
        target,
        original_width=original,
        width_override=79,
    )

    assert target._width == 100


def test_console_width_override_removes_temporary_width() -> None:
    class TargetConsole:
        pass

    missing = object()
    target = TargetConsole()
    original = streaming_module._apply_console_width_override(target, 79)

    assert getattr(target, "_width", missing) == 79

    streaming_module._restore_console_width_override(
        target,
        original_width=original,
        width_override=79,
    )

    assert getattr(target, "_width", missing) is missing


def test_console_width_restore_noops_without_override() -> None:
    class TargetConsole:
        pass

    missing = object()
    target = TargetConsole()
    original = streaming_module._apply_console_width_override(target, None)

    streaming_module._restore_console_width_override(
        target,
        original_width=original,
        width_override=None,
    )

    assert getattr(target, "_width", missing) is missing


def test_diff_live_updates_only_changed_tail_line() -> None:
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=40),
        transient=True,
    )

    live.__enter__()
    output.truncate(0)
    output.seek(0)
    live.update(Text("alpha\nbeta"), refresh=True)

    output.truncate(0)
    output.seek(0)
    live.update(Text("alpha\ngamma"), refresh=True)

    rendered = output.getvalue()
    assert "\x1b[2K\x1b[1A\x1b[2K" not in rendered
    assert "\x1b[1Ggamma" in rendered


def test_diff_live_appends_new_lines_with_newline_scroll() -> None:
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=40),
        transient=True,
    )

    live.__enter__()
    output.truncate(0)
    output.seek(0)
    live.update(Text("alpha\nbeta"), refresh=True)

    output.truncate(0)
    output.seek(0)
    live.update(Text("alpha\nbeta\ngamma"), refresh=True)

    rendered = output.getvalue()
    assert rendered.startswith("\x1b[1G\ngamma")
    assert "\x1b[1B" not in rendered


def test_diff_live_clears_only_shortened_line_tail() -> None:
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=40),
        transient=True,
    )

    live.__enter__()
    output.truncate(0)
    output.seek(0)
    live.update(Text("abcdefgh"), refresh=True)

    output.truncate(0)
    output.seek(0)
    live.update(Text("abc"), refresh=True)

    rendered = output.getvalue()
    assert rendered.startswith("\x1b[1Gabc")
    assert "\x1b[0K" in rendered


def test_diff_live_non_terminal_prints_only_final_frame() -> None:
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=False, color_system=None, width=40),
        transient=True,
    )

    live.__enter__()
    live.update(Text("alpha"), refresh=True)

    assert output.getvalue() == ""

    live.stop()

    rendered = output.getvalue()
    assert "alpha" in rendered
    assert "\x1b[" not in rendered


def test_render_tool_segment_uses_syntax_preview_for_code_tools() -> None:
    handle = _make_handle("markdown")
    output = io.StringIO()
    renderer = Console(file=output, force_terminal=False, color_system=None, width=80)
    segment = StreamSegment(
        kind="tool",
        text="",
        tool_name="hf_hub_query_raw",
        code_preview=ToolCodePreview(
            code="resp = await hf_trending()\nprint(resp)",
            language="python",
            complete=False,
        ),
    )

    renderer.print(handle._render_tool_segment(segment, cursor_suffix=""))

    rendered = output.getvalue()
    assert "hf_hub_query_raw" in rendered
    assert "resp = await hf_trending()" in rendered
    assert "print(resp)" in rendered


def test_render_tool_segment_splits_completed_shell_heredoc_by_language() -> None:
    handle = _make_handle("markdown")
    command = "cat > example.py <<'PY'\nprint('hello')\nPY\npython example.py"
    segment = StreamSegment(
        kind="tool",
        text="",
        tool_name="execute",
        code_preview=ToolCodePreview(
            code=command,
            language="bash",
            complete=True,
            variant="shell",
        ),
    )

    renderable = handle._render_tool_segment(segment, cursor_suffix="")
    assert isinstance(renderable, Group)
    syntax_blocks = [child for child in renderable.renderables if isinstance(child, Syntax)]

    assert [block._lexer for block in syntax_blocks] == ["bash", "python", "bash"]
    assert [block.code for block in syntax_blocks] == [
        "cat > example.py <<'PY'",
        "print('hello')",
        "PY\npython example.py",
    ]


def test_render_tool_segment_splits_direct_interpreter_heredoc_by_language() -> None:
    handle = _make_handle("markdown")
    command = "python - <<'PY'\nprint('hello')\nPY\nrm -rf /tmp/example"
    segment = StreamSegment(
        kind="tool",
        text="",
        tool_name="execute",
        code_preview=ToolCodePreview(
            code=command,
            language="bash",
            complete=True,
            variant="shell",
        ),
    )

    renderable = handle._render_tool_segment(segment, cursor_suffix="")
    assert isinstance(renderable, Group)
    syntax_blocks = [child for child in renderable.renderables if isinstance(child, Syntax)]

    assert [block._lexer for block in syntax_blocks] == ["bash", "python", "bash"]


def test_render_tool_segment_splits_streaming_interpreter_heredoc_before_delimiter() -> None:
    handle = _make_handle("markdown")
    command = "python - <<'PY'\nprint('hello')"
    segment = StreamSegment(
        kind="tool",
        text="",
        tool_name="execute",
        code_preview=ToolCodePreview(
            code=command,
            language="bash",
            complete=False,
            variant="shell",
        ),
    )

    renderable = handle._render_tool_segment(segment, cursor_suffix="")
    assert isinstance(renderable, Group)
    syntax_blocks = [child for child in renderable.renderables if isinstance(child, Syntax)]

    assert [block._lexer for block in syntax_blocks] == ["bash", "python"]


def test_render_tool_segment_splits_streaming_uv_run_python_heredoc() -> None:
    handle = _make_handle("markdown")
    command = (
        "uv run python - <<'PY'\n"
        "from importlib.metadata import metadata, version\n"
        "for name in ('mcp','mcp-types'):\n"
        " print(name, version(name))"
    )
    segment = StreamSegment(
        kind="tool",
        text="",
        tool_name="execute",
        code_preview=ToolCodePreview(
            code=command,
            language="bash",
            complete=False,
            variant="shell",
        ),
    )

    renderable = handle._render_tool_segment(segment, cursor_suffix="")
    assert isinstance(renderable, Group)
    syntax_blocks = [child for child in renderable.renderables if isinstance(child, Syntax)]

    assert [block._lexer for block in syntax_blocks] == ["bash", "python"]
    assert syntax_blocks[1].code.startswith("from importlib.metadata import")


def test_render_tool_segment_splits_streaming_pnpm_exec_tsx_heredoc() -> None:
    handle = _make_handle("markdown")
    command = (
        "pnpm -C packages/app exec tsx - <<'TS'\n"
        "import { Client, StreamableHTTPClientTransport } "
        "from '@modelcontextprotocol/client';\n"
        "import { z } from 'zod';\n"
        "const client = new Client({ name: 'smoke', version: '1.0.0' });"
    )
    segment = StreamSegment(
        kind="tool",
        text="",
        tool_name="execute",
        code_preview=ToolCodePreview(
            code=command,
            language="bash",
            complete=False,
            variant="shell",
        ),
    )

    renderable = handle._render_tool_segment(segment, cursor_suffix="")
    assert isinstance(renderable, Group)
    syntax_blocks = [child for child in renderable.renderables if isinstance(child, Syntax)]

    assert [block._lexer for block in syntax_blocks] == ["bash", "typescript"]
    assert syntax_blocks[1].code.startswith("import { Client")


def test_render_tool_segment_keeps_final_unterminated_interpreter_heredoc_as_shell() -> None:
    handle = _make_handle("markdown")
    command = "python - <<'PY'\nprint('hello')"
    segment = StreamSegment(
        kind="tool",
        text="",
        tool_name="execute",
        code_preview=ToolCodePreview(
            code=command,
            language="bash",
            complete=True,
            variant="shell",
        ),
    )

    renderable = handle._render_tool_segment(segment, cursor_suffix="")
    assert isinstance(renderable, Group)
    syntax_blocks = [child for child in renderable.renderables if isinstance(child, Syntax)]

    assert [block._lexer for block in syntax_blocks] == ["bash"]


def test_render_tool_segment_styles_apply_patch_preview_lines() -> None:
    handle = _make_handle("markdown")
    segment = StreamSegment(
        kind="tool",
        text=(
            "execute\n"
            "apply_patch preview: streaming patch (partial)\n"
            "*** Begin Patch\n"
            "*** Update File: a.txt\n"
            "@@\n"
            " context\n"
            "-old\n"
            "+new\n"
        ),
        tool_name="execute",
    )

    renderable = handle._render_tool_segment(segment, cursor_suffix="")

    assert isinstance(renderable, Text)
    span_styles = {str(span.style) for span in renderable.spans}
    assert "dim" in span_styles
    assert "cyan" in span_styles
    assert "yellow" in span_styles
    assert "red" in span_styles
    assert "green" in span_styles


def test_render_tool_segment_keeps_apply_patch_styling_after_truncation() -> None:
    handle = _make_handle("markdown")
    segment = StreamSegment(
        kind="tool",
        text="execute\n*** Update File: a.txt\n@@\n-old\n+new\n",
        tool_name="execute",
        apply_patch_preview=True,
    )

    renderable = handle._render_tool_segment(segment, cursor_suffix="")

    assert isinstance(renderable, Text)
    span_styles = {str(span.style) for span in renderable.spans}
    assert "cyan" in span_styles
    assert "yellow" in span_styles
    assert "red" in span_styles
    assert "green" in span_styles


def test_diff_live_stop_reprints_full_truncated_frame_when_preserved() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=40)
    local_console._height = 2
    live = streaming_module._DiffLive(
        console=local_console,
        transient=False,
    )

    live.__enter__()
    live.update(Text("one\ntwo\nthree"), refresh=True)

    output.truncate(0)
    output.seek(0)
    live.stop()

    rendered = output.getvalue()
    assert "one" in rendered
    assert "two" in rendered
    assert "three" in rendered


def test_diff_live_stop_reprint_preserves_dim_apply_patch_context() -> None:
    handle = _make_handle("markdown")
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system="standard", width=40)
    local_console._height = 2
    live = streaming_module._DiffLive(
        console=local_console,
        transient=False,
    )
    renderable = handle._render_tool_segment(
        StreamSegment(
            kind="tool",
            text=(
                "execute\n"
                "apply_patch preview: streaming patch (partial)\n"
                "*** Begin Patch\n"
                "@@\n"
                " context\n"
                "*** End Patch\n"
            ),
            tool_name="execute",
            apply_patch_preview=True,
        ),
        cursor_suffix="",
    )

    live.__enter__()
    live.update(renderable, refresh=True)

    output.truncate(0)
    output.seek(0)
    live.stop()

    rendered = output.getvalue()
    assert "\x1b[2m context" in rendered


def test_diff_live_reprints_frame_after_console_print() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=40)
    live = streaming_module._DiffLive(
        console=local_console,
        transient=True,
    )

    live.__enter__()
    live.update(Text("frame"), refresh=True)

    output.truncate(0)
    output.seek(0)
    local_console.print("notice")

    rendered = output.getvalue()
    assert "notice" in rendered
    assert "frame" in rendered


def test_nested_diff_live_does_not_write_cursor_diffs() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=40)

    with Live(Text("monitor"), console=local_console, auto_refresh=False):
        live = streaming_module._DiffLive(
            console=local_console,
            transient=True,
        )
        live.__enter__()
        assert live._nested

        output.seek(0)
        output.truncate(0)
        live.update(Text("streaming patch"), refresh=True)

        assert output.getvalue() == ""
        assert live._lines == []
        live.stop()


def test_diff_live_cleans_frame_before_releasing_live_ownership() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=40)

    class OwnershipCheckingDiffLive(streaming_module._DiffLive):
        def _clear_region(self) -> None:
            assert self in self.console._live_stack
            super()._clear_region()

        def _restore_console_state(self) -> None:
            if self._console_state_active:
                assert self in self.console._live_stack
            super()._restore_console_state()

    live = OwnershipCheckingDiffLive(console=local_console, transient=True)
    live.__enter__()
    live.update(Text("streaming patch"), refresh=True)
    live.stop()

    assert local_console._live_stack == []


def test_diff_live_releases_live_ownership_when_state_restoration_fails() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=40)

    class FailingRestoreDiffLive(streaming_module._DiffLive):
        restore_calls = 0

        def _restore_console_state(self) -> None:
            super()._restore_console_state()
            self.restore_calls += 1
            if self.restore_calls == 1:
                raise RuntimeError("restore failed")

    live = FailingRestoreDiffLive(console=local_console, transient=True)
    live.__enter__()
    live.update(Text("streaming patch"), refresh=True)

    with pytest.raises(RuntimeError, match="restore failed"):
        live.stop()

    assert local_console._live_stack == []


def test_diff_live_stop_does_not_add_extra_newline_when_cursor_below_frame() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=40)
    live = streaming_module._DiffLive(
        console=local_console,
        transient=False,
    )

    live.__enter__()
    live.update(Text("frame"), refresh=True)
    local_console.print("notice")

    output.truncate(0)
    output.seek(0)
    live.stop()

    rendered = output.getvalue()
    assert "\n" not in rendered


def test_diff_live_resyncs_cursor_after_console_print_before_next_update() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=40)
    live = streaming_module._DiffLive(
        console=local_console,
        transient=True,
    )

    live.__enter__()
    live.update(Text("frame1\nline2"), refresh=True)

    output.truncate(0)
    output.seek(0)
    local_console.print("notice")

    output.truncate(0)
    output.seek(0)
    live.update(Text("frameX\nline2"), refresh=True)

    rendered = output.getvalue()
    assert rendered.startswith("\x1b[1G\x1b[2AframeX")
    assert "\x1b[1G\x1b[1AframeX" not in rendered


def test_diff_live_appends_without_extra_scroll_after_console_print() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=40)
    live = streaming_module._DiffLive(
        console=local_console,
        transient=True,
    )

    live.__enter__()
    live.update(Text("a\nb"), refresh=True)

    output.truncate(0)
    output.seek(0)
    local_console.print("notice")

    output.truncate(0)
    output.seek(0)
    live.update(Text("a\nb\nc"), refresh=True)

    rendered = output.getvalue()
    assert rendered.startswith("\x1b[1Gc")
    assert rendered != "\x1b[1G\nc\x1b[1G"


def test_diff_live_growing_last_line_rewinds_after_console_print() -> None:
    output = io.StringIO()
    local_console = Console(file=output, force_terminal=True, color_system=None, width=4)
    live = streaming_module._DiffLive(
        console=local_console,
        transient=True,
    )

    live.__enter__()
    live.update(Text("abcd\nefg"), refresh=True)

    output.truncate(0)
    output.seek(0)
    local_console.print("notice")

    output.truncate(0)
    output.seek(0)
    live.update(Text("abcd\nefgh\ni"), refresh=True)

    rendered = output.getvalue()
    assert rendered.startswith("\x1b[1G\x1b[1Aefgh")
    assert not rendered.startswith("\x1b[1Gefgh")


def test_render_coalesces_contiguous_markdown_segments() -> None:
    handle = _make_handle("markdown")

    merged = handle._coalesce_display_segments(
        [
            StreamSegment(kind="markdown", text="See [foo]\n\n", frozen=True),
            StreamSegment(kind="markdown", text="[foo]: https://example.com\n"),
            StreamSegment(kind="tool", text="tool\n"),
        ]
    )

    assert [segment.kind for segment in merged] == ["markdown", "tool"]
    assert merged[0].text == "See [foo]\n\n[foo]: https://example.com\n"


def test_diff_live_uses_newlines_when_last_line_wraps_and_grows() -> None:
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=4),
        transient=True,
    )

    live.__enter__()
    live._lines = [
        streaming_module._RenderedLine("abcd", 4),
        streaming_module._RenderedLine("efg", 3),
    ]
    output.truncate(0)
    output.seek(0)
    live._write_diff(
        [
            streaming_module._RenderedLine("abcd", 4),
            streaming_module._RenderedLine("efgh", 4),
            streaming_module._RenderedLine("i", 1),
        ]
    )

    rendered = output.getvalue()
    assert rendered.startswith("\x1b[1Gefgh\x1b[1G\ni")
    assert "\x1b[1B" not in rendered
    assert "\x1b[1G\n" in rendered


def test_diff_live_uses_newline_when_frame_grows_past_old_bottom() -> None:
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=40),
        transient=True,
    )

    live.__enter__()
    live._lines = [
        streaming_module._RenderedLine("alpha", 5),
        streaming_module._RenderedLine("beta", 4),
    ]
    output.truncate(0)
    output.seek(0)
    live._write_diff(
        [
            streaming_module._RenderedLine("ALPHA", 5),
            streaming_module._RenderedLine("BETA", 4),
            streaming_module._RenderedLine("gamma", 5),
        ]
    )

    rendered = output.getvalue()
    assert "\x1b[1G\n" in rendered
    assert "\x1b[1B\x1b[1Ggamma" not in rendered


def test_diff_live_uses_newline_for_tail_full_width_line_updates() -> None:
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=4),
        transient=True,
    )

    live.__enter__()
    live._lines = [
        streaming_module._RenderedLine("one", 3),
        streaming_module._RenderedLine("abcd", 4),
    ]
    output.truncate(0)
    output.seek(0)
    live._write_diff(
        [
            streaming_module._RenderedLine("one", 3),
            streaming_module._RenderedLine("wxyz", 4),
            streaming_module._RenderedLine("tail", 4),
        ]
    )

    rendered = output.getvalue()
    assert "\x1b[1G\n" in rendered
    assert "\x1b[1B" not in rendered


def test_diff_live_does_not_scroll_for_full_width_mid_frame_updates() -> None:
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=4),
        transient=True,
    )

    live.__enter__()
    live._lines = [
        streaming_module._RenderedLine("one", 3),
        streaming_module._RenderedLine("abcd", 4),
        streaming_module._RenderedLine("tail", 4),
    ]
    output.truncate(0)
    output.seek(0)
    live._write_diff(
        [
            streaming_module._RenderedLine("one", 3),
            streaming_module._RenderedLine("wxyz", 4),
            streaming_module._RenderedLine("TAIL", 4),
        ]
    )

    rendered = output.getvalue()
    assert "\x1b[1G\n" not in rendered
    assert "\x1b[1G\x1b[1B" in rendered


def test_sync_streaming_markdown_unthrottled_before_scroll(monkeypatch) -> None:
    handle = _make_handle("markdown")
    assert handle._async_mode is False

    render_calls: list[None] = []
    monkeypatch.setattr(handle, "_render_current_buffer", lambda: render_calls.append(None))

    handle.update("first")
    handle.update("second")
    handle.update("third")

    assert len(render_calls) == 3


def test_markdown_pre_scroll_throttle_activates_for_tall_content() -> None:
    handle = _make_handle("markdown")

    assert handle._pre_scroll_throttle_started is False

    handle._update_pre_scroll_throttle(content_height=16, max_allowed_height=20)

    assert handle._pre_scroll_throttle_started is True
    assert handle._render_throttle_active is True


def test_markdown_pre_scroll_throttle_stays_off_for_short_content() -> None:
    handle = _make_handle("markdown")

    handle._update_pre_scroll_throttle(content_height=5, max_allowed_height=20)

    assert handle._pre_scroll_throttle_started is False
    assert handle._render_throttle_active is False


def test_sync_streaming_respects_render_interval_after_scroll(monkeypatch) -> None:
    handle = _make_handle("markdown")
    assert handle._async_mode is False
    assert handle._min_render_interval is not None
    handle._scrolling_started = True

    render_calls: list[None] = []
    monkeypatch.setattr(handle, "_render_current_buffer", lambda: render_calls.append(None))

    interval = handle._min_render_interval or 0.25
    monotonic_values = [0.0, 0.0, interval / 2, interval + 0.01, interval + 0.01]

    def _fake_monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return interval + 0.01

    monkeypatch.setattr(streaming_module.time, "monotonic", _fake_monotonic)

    handle.update("first")
    handle.update("second")
    handle.update("third")

    assert len(render_calls) == 2


def test_sync_streaming_respects_render_interval_after_pre_scroll_throttle(monkeypatch) -> None:
    handle = _make_handle("markdown")
    assert handle._async_mode is False
    assert handle._min_render_interval is not None
    handle._pre_scroll_throttle_started = True

    render_calls: list[None] = []
    monkeypatch.setattr(handle, "_render_current_buffer", lambda: render_calls.append(None))

    interval = handle._min_render_interval or 0.125
    monotonic_values = [0.0, 0.0, interval / 2, interval + 0.01, interval + 0.01]

    def _fake_monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return interval + 0.01

    monkeypatch.setattr(streaming_module.time, "monotonic", _fake_monotonic)

    handle.update("first")
    handle.update("second")
    handle.update("third")

    assert len(render_calls) == 2


def test_sync_streaming_plain_respects_render_interval(monkeypatch) -> None:
    handle = _make_handle("plain")
    assert handle._async_mode is False
    assert handle._min_render_interval is not None

    render_calls: list[None] = []
    monkeypatch.setattr(handle, "_render_current_buffer", lambda: render_calls.append(None))

    interval = handle._min_render_interval or 0.05
    monotonic_values = [0.0, 0.0, interval / 2, interval + 0.01, interval + 0.01]

    def _fake_monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return interval + 0.01

    monkeypatch.setattr(streaming_module.time, "monotonic", _fake_monotonic)

    handle.update("first")
    handle.update("second")
    handle.update("third")

    assert len(render_calls) == 2


def test_resolve_progress_resume_debounce_seconds_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FAST_AGENT_PROGRESS_RESUME_DEBOUNCE_SECONDS", "0.05")
    assert streaming_module._resolve_progress_resume_debounce_seconds() == 0.05

    monkeypatch.setenv("FAST_AGENT_PROGRESS_RESUME_DEBOUNCE_SECONDS", "-1")
    assert streaming_module._resolve_progress_resume_debounce_seconds() == 0.0

    monkeypatch.setenv("FAST_AGENT_PROGRESS_RESUME_DEBOUNCE_SECONDS", "invalid")
    assert streaming_module._resolve_progress_resume_debounce_seconds() == 0.12


def test_stream_cursor_suffix_only_targets_last_segment() -> None:
    handle = _make_handle("plain")

    assert handle._cursor_suffix(segment_index=0, total_segments=2) == ""
    assert (
        handle._cursor_suffix(segment_index=1, total_segments=2)
        == streaming_module.STREAM_CURSOR_BLOCK
    )

    handle._stream_cursor_visible = False
    assert handle._cursor_suffix(segment_index=1, total_segments=2) == " "

    handle._show_stream_cursor = False
    assert handle._cursor_suffix(segment_index=1, total_segments=2) == ""


def test_stream_cursor_blinks_after_idle_delay_and_resets() -> None:
    blink = streaming_module._StreamCursorBlink(
        last_activity_at=10.0,
        idle_delay=0.5,
        half_cycle=0.5,
    )

    assert blink.visible_at(10.499) is True
    assert blink.visible_at(10.5) is False
    assert blink.visible_at(10.999) is False
    assert blink.visible_at(11.0) is True
    assert blink.visible_at(11.5) is False
    assert blink.next_transition_at(11.6) == 12.0

    blink.reset(20.0)
    assert blink.visible_at(20.499) is True
    assert blink.visible_at(20.5) is False


def test_stream_cursor_is_bright_green_without_recoloring_content() -> None:
    handle = _make_handle("plain")
    renderable = handle._render_display_segment(
        StreamSegment(kind="plain", text="prior ● content"),
        cursor_suffix=streaming_module.STREAM_CURSOR_BLOCK,
    )
    local_console = Console(force_terminal=True, color_system="standard", width=40)
    segments = tuple(local_console.render(renderable, local_console.options))
    green_segments = [
        segment
        for segment in segments
        if segment.style is not None
        and segment.style.color is not None
        and segment.style.color.name == "bright_green"
    ]

    assert (
        "".join(segment.text for segment in green_segments) == streaming_module.STREAM_CURSOR_BLOCK
    )
    assert green_segments[0].style is not None
    assert green_segments[0].style.bold is True
    assert green_segments[0].style.dim is False


def test_stream_cursor_is_inserted_after_syntax_highlighting() -> None:
    handle = _make_handle("markdown")
    segment = StreamSegment(
        kind="tool",
        text="",
        tool_name="execute",
        code_preview=ToolCodePreview(
            code="python - <<'PY'\nprint('hello')",
            language="bash",
            complete=False,
            variant="shell",
        ),
    )

    renderable = handle._render_display_segment(
        segment,
        cursor_suffix=streaming_module.STREAM_CURSOR_BLOCK,
    )

    assert isinstance(renderable, streaming_module._CursorStyledRenderable)
    assert isinstance(renderable.renderable, Group)
    syntax_blocks = [
        child for child in renderable.renderable.renderables if isinstance(child, Syntax)
    ]
    assert syntax_blocks
    assert all(streaming_module.STREAM_CURSOR_BLOCK not in block.code for block in syntax_blocks)

    output = io.StringIO()
    local_console = Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=40,
    )
    local_console.print(renderable)

    rendered = output.getvalue()
    assert streaming_module.STREAM_CURSOR_BLOCK in rendered
    assert "48;2;227;210;210" not in rendered

    hidden = handle._render_display_segment(segment, cursor_suffix=" ")
    plain_console = Console(force_terminal=True, color_system=None, width=40)
    visible_lines = [
        "".join(rendered.text for rendered in line)
        for line in plain_console.render_lines(renderable, plain_console.options, pad=False)
    ]
    hidden_lines = [
        "".join(rendered.text for rendered in line)
        for line in plain_console.render_lines(hidden, plain_console.options, pad=False)
    ]
    assert any(
        line.rstrip().endswith(f"print('hello'){streaming_module.STREAM_CURSOR_BLOCK}")
        for line in visible_lines
    )
    assert [
        line.replace(streaming_module.STREAM_CURSOR_BLOCK, " ") for line in visible_lines
    ] == hidden_lines


def test_stream_cursor_remains_visible_after_markdown_table() -> None:
    handle = _make_handle("markdown")
    renderable = handle._render_display_segment(
        StreamSegment(kind="markdown", text="| A | B |\n| - | - |\n| 1 | 2 |"),
        cursor_suffix=streaming_module.STREAM_CURSOR_BLOCK,
    )
    local_console = Console(force_terminal=True, color_system="standard", width=40)
    segments = tuple(local_console.render(renderable, local_console.options))
    green_text = "".join(
        segment.text
        for segment in segments
        if segment.style is not None
        and segment.style.color is not None
        and segment.style.color.name == "bright_green"
    )

    assert green_text == streaming_module.STREAM_CURSOR_BLOCK


def test_stream_cursor_replaces_reasoning_markdown_suffix_at_segment_boundary() -> None:
    handle = _make_handle("markdown")
    renderable = handle._render_display_segment(
        StreamSegment(kind="reasoning", text="*Thinking*"),
        cursor_suffix=streaming_module.STREAM_CURSOR_BLOCK,
    )
    local_console = Console(force_terminal=True, color_system="standard", width=40)
    segments = tuple(local_console.render(renderable, local_console.options))
    visible = "".join(segment.text for segment in segments if not segment.control)
    cursor_segments = [
        segment for segment in segments if streaming_module.STREAM_CURSOR_BLOCK in segment.text
    ]

    assert visible.count(streaming_module.STREAM_CURSOR_BLOCK) == 1
    assert visible.rstrip().endswith(f"Thinking{streaming_module.STREAM_CURSOR_BLOCK}")
    assert len(cursor_segments) == 1
    cursor_style = cursor_segments[0].style
    assert cursor_style is not None
    assert cursor_style.color is not None
    assert cursor_style.color.name == "bright_green"


@pytest.mark.asyncio
async def test_async_stream_cursor_blinks_and_activity_makes_it_visible() -> None:
    original_console = _set_console_size()
    handle = _make_handle("plain")
    worker = handle._worker_task
    try:
        assert handle._async_mode is True
        handle._stream_cursor_blink = streaming_module._StreamCursorBlink(
            idle_delay=0.01,
            half_cycle=0.01,
        )

        handle.update("first")
        await handle.wait_for_drain()
        assert handle._stream_cursor_visible is True

        async with asyncio.timeout(0.25):
            while handle._stream_cursor_visible:
                await asyncio.sleep(0.001)

        handle.update(" second")
        await handle.wait_for_drain()
        assert handle._stream_cursor_visible is True
    finally:
        handle.close()
        if worker is not None:
            await asyncio.gather(worker, return_exceptions=True)
        _restore_console_size(original_console)


def test_preserve_final_frame_requires_rendered_content() -> None:
    handle = _make_handle("markdown")
    assert handle.preserve_final_frame() is False


def test_streaming_model_update_invalidates_header_cache() -> None:
    handle = _make_handle("markdown")

    assert "model-before" not in handle._build_header().plain

    handle.update_model("model-before ↔ ↗")

    assert "model-before ↔ ↗" in handle._build_header().plain


def test_preserve_final_frame_allows_pending_reasoning_content() -> None:
    handle = _make_handle("markdown")
    handle.update_chunk(StreamChunk("<think>"))
    assert handle.preserve_final_frame() is True


def test_preserve_final_frame_sets_live_non_transient() -> None:
    class _FakeLive:
        def __init__(self) -> None:
            self.transient = True
            self.exited = False

        def __exit__(self, *_args: object) -> None:
            self.exited = True

    handle = _make_handle("markdown")
    fake_live = _FakeLive()
    handle._live = cast("Any", fake_live)
    handle._live_started = True

    assert handle.preserve_final_frame() is True
    handle._shutdown_live_resources()

    assert fake_live.transient is False
    assert fake_live.exited is True


def test_preserve_final_frame_finalize_keeps_padding_stable(monkeypatch) -> None:
    handle = _make_handle("markdown")
    handle._preserve_final_frame = True
    handle._max_render_height = 99
    handle._height_fudge = 3
    handle._handle_chunk("short response")

    monkeypatch.setattr(handle._segment_assembler, "flush", lambda: False)

    captured: list[tuple[int, int]] = []

    def _capture_render() -> None:
        captured.append((handle._max_render_height, handle._height_fudge))

    monkeypatch.setattr(handle, "_render_current_buffer", _capture_render)

    handle.finalize("short response")

    assert captured == [(99, 3)]


def test_preserve_final_frame_finalize_omits_tail_padding_from_last_frame(monkeypatch) -> None:
    class _FakeLive:
        def __init__(self) -> None:
            self.renderable: Any | None = None
            self.transient = True

        def update(self, renderable: Any, *, refresh: bool = False) -> None:
            del refresh
            self.renderable = renderable

        def __exit__(self, *_args: object) -> None:
            return None

    handle = _make_handle("markdown")
    handle._preserve_final_frame = True
    handle._max_render_height = 6
    handle._handle_chunk("short response")
    monkeypatch.setattr(handle._segment_assembler, "flush", lambda: False)

    fake_live = _FakeLive()
    handle._live = cast("Any", fake_live)
    handle._live_started = True

    original_console = _set_console_size(width=40, height=12)
    try:
        handle.finalize("short response")
        assert fake_live.renderable is not None

        options = console.console.options.update(width=40)
        rendered_lines = console.console.render_lines(
            fake_live.renderable, options=options, pad=False
        )
        rendered_text = [console.console._render_buffer(line).rstrip() for line in rendered_lines]

        assert rendered_text[-1] == "short response"
    finally:
        _restore_console_size(original_console)


def test_scrolling_indicator_is_debounced_and_sticky() -> None:
    handle = _make_handle("markdown")

    handle._update_scroll_status(is_truncated=True, now=0.0)
    assert handle._scrolling_started is True
    assert handle._scroll_indicator_visible is False
    assert "scrolling" not in handle._build_header().plain

    handle._update_scroll_status(is_truncated=False, now=0.05)
    assert handle._scroll_indicator_visible is False

    handle._update_scroll_status(is_truncated=True, now=0.1)
    handle._update_scroll_status(
        is_truncated=True,
        now=0.1 + streaming_module.SCROLL_INDICATOR_DEBOUNCE_SECONDS + 0.01,
    )
    assert handle._scroll_indicator_visible is True
    assert "scrolling" in handle._build_header().plain

    handle._update_scroll_status(is_truncated=False, now=1.0)
    assert handle._scroll_indicator_visible is True
    assert "scrolling" in handle._build_header().plain


def test_finalize_hides_scrolling_indicator_from_last_live_frame(monkeypatch) -> None:
    handle = _make_handle("markdown")
    handle._scroll_indicator_visible = True
    handle._handle_chunk("final response")

    monkeypatch.setattr(handle._segment_assembler, "flush", lambda: False)

    captured: list[bool] = []

    def _capture_render() -> None:
        captured.append(handle._scroll_indicator_visible)

    monkeypatch.setattr(handle, "_render_current_buffer", _capture_render)

    handle.finalize("final response")

    assert captured == [False]


def test_close_incomplete_code_blocks_keeps_closed_fence_with_trailing_text() -> None:
    text = """Here is code:\n```python\nprint(1)\n```\nAnd more text."""

    assert close_incomplete_code_blocks(text) == text


def test_close_incomplete_code_blocks_closes_unterminated_fence() -> None:
    text = """Here is code:\n```python\nprint(1)"""

    assert close_incomplete_code_blocks(text) == text + "\n```\n"


def test_close_incomplete_code_blocks_supports_tilde_fences() -> None:
    text = """Here is code:\n~~~json\n{\"a\": 1}"""

    assert close_incomplete_code_blocks(text) == text + "\n~~~\n"


# -- _DiffLive height-clamp tests -------------------------------------------


def test_diff_live_clamps_new_frame_to_terminal_height() -> None:
    """When the rendered frame exceeds terminal height, only the tail is kept."""
    output = io.StringIO()
    # Terminal height = 4
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=40, height=4),
        transient=True,
    )

    live.__enter__()

    # Build a 6-line renderable – taller than the 4-row terminal.
    live.update(Text("L1\nL2\nL3\nL4\nL5\nL6"), refresh=True)

    # The stored frame should be clamped to 4 lines (the terminal height).
    assert len(live._lines) == 4
    # The kept lines are the tail: L3, L4, L5, L6.
    stored_texts = [line.text for line in live._lines]
    assert "L3" in stored_texts[0]
    assert "L6" in stored_texts[-1]


def test_diff_live_clamps_old_frame_for_safe_diff() -> None:
    """If a previous frame exceeded height, old lines are clamped before diff."""
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=40, height=4),
        transient=True,
    )

    live.__enter__()

    # Manually set _lines to something oversized (simulates a pre-fix state or
    # a terminal that was resized smaller between frames).
    live._lines = [streaming_module._RenderedLine(f"old{i}", len(f"old{i}")) for i in range(8)]

    # Now refresh with a small frame – the old lines should be clamped first
    # so the diff can compute correct cursor-up distances.
    live.update(Text("A\nB"), refresh=True)

    assert len(live._lines) == 2
    assert "A" in live._lines[0].text


def test_diff_live_height_clamp_preserves_cursor_tracking() -> None:
    """After clamping, subsequent diffs position correctly."""
    output = io.StringIO()
    live = streaming_module._DiffLive(
        console=Console(file=output, force_terminal=True, color_system=None, width=40, height=5),
        transient=True,
    )

    live.__enter__()
    # First frame: exactly 5 lines (fits the terminal).
    live.update(Text("A\nB\nC\nD\nE"), refresh=True)
    assert len(live._lines) == 5

    output.truncate(0)
    output.seek(0)

    # Second frame: 7 lines – will be clamped to 5.
    live.update(Text("A\nB\nC\nD\nE\nF\nG"), refresh=True)
    assert len(live._lines) == 5
    assert "G" in live._lines[-1].text

    output.truncate(0)
    output.seek(0)

    # Third frame: change the last line only – diff should emit minimal output.
    live.update(Text("A\nB\nC\nD\nE\nF\nZ"), refresh=True)
    assert len(live._lines) == 5
    rendered = output.getvalue()
    assert "Z" in rendered
    # Should NOT contain a large cursor-up (max 4 for a 5-line frame).
    assert "\x1b[5A" not in rendered
    assert "\x1b[6A" not in rendered


def test_stream_header_margin_constant() -> None:
    """The header/margin constant should reserve 3 lines (header + safety)."""
    assert streaming_module._STREAM_HEADER_AND_MARGIN_LINES == 3
