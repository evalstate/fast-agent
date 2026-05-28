from fast_agent.ui.apply_patch_preview import (
    build_apply_patch_preview,
    extract_apply_patch_text,
    extract_non_command_args,
    extract_partial_apply_patch_text,
    format_partial_apply_patch_preview,
    is_shell_execution_tool,
    render_patch_preview,
    style_apply_patch_preview_text,
    summarize_patch,
)


def test_extract_apply_patch_text_from_direct_argument() -> None:
    command = "apply_patch '*** Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch'"

    patch_text = extract_apply_patch_text(command)

    assert patch_text is not None
    assert patch_text.startswith("*** Begin Patch")
    assert patch_text.endswith("*** End Patch")


def test_extract_apply_patch_text_from_heredoc() -> None:
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Delete File: old.txt\n"
        "*** End Patch\n"
        "PATCH"
    )

    patch_text = extract_apply_patch_text(command)

    assert patch_text is not None
    assert "*** Delete File: old.txt" in patch_text


def test_extract_apply_patch_text_with_shell_wrapper() -> None:
    command = (
        "bash -lc \"set -e\n"
        "apply_patch <<'EOF'\n"
        "*** Begin Patch\n"
        "*** Update File: foo.py\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
        "EOF\""
    )

    patch_text = extract_apply_patch_text(command)

    assert patch_text is not None
    assert "*** Update File: foo.py" in patch_text


def test_extract_apply_patch_text_returns_none_when_markers_missing() -> None:
    assert extract_apply_patch_text("apply_patch 'not a patch'") is None


def test_extract_partial_apply_patch_text_returns_tail_without_end_marker() -> None:
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Update File: foo.py\n"
        "@@\n"
        "-old\n"
        "+new"
    )

    patch_text = extract_partial_apply_patch_text(command)

    assert patch_text is not None
    assert patch_text.startswith("*** Begin Patch")
    assert patch_text.endswith("+new")


def test_summarize_patch_counts_operations() -> None:
    patch_text = (
        "*** Begin Patch\n"
        "*** Add File: added.txt\n"
        "+hello\n"
        "*** Update File: foo.py\n"
        "@@\n"
        "-a\n"
        "+b\n"
        "*** Delete File: gone.txt\n"
        "*** End Patch"
    )

    summary = summarize_patch(patch_text)

    assert summary is not None
    assert summary.file_count == 3
    assert summary.operation_counts == {"add": 1, "update": 1, "delete": 1}
    assert "3 files" in summary.summary


def test_summarize_patch_returns_none_for_invalid_patch() -> None:
    patch_text = (
        "*** Begin Patch\n"
        "*** Frobnicate File: bad.txt\n"
        "*** End Patch"
    )

    assert summarize_patch(patch_text) is None
    assert build_apply_patch_preview(f"apply_patch '{patch_text}'") is None


def test_render_patch_preview_truncates_large_patches() -> None:
    patch_text = "\n".join(
        [
            "*** Begin Patch",
            "*** Add File: long.txt",
            "+line-1",
            "+line-2",
            "+line-3",
            "*** End Patch",
        ]
    )

    rendered = render_patch_preview(patch_text, max_lines=4)

    assert "+line-1" in rendered
    assert "+line-2" in rendered
    assert "+line-3" not in rendered
    assert "(+2 more lines)" in rendered


def test_format_partial_apply_patch_preview_marks_streaming_patch() -> None:
    text = format_partial_apply_patch_preview(
        "*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new"
    )

    assert text.startswith("apply_patch preview:")
    assert "streaming patch (partial)" in text
    assert "*** Update File: a.txt" in text


def test_is_shell_execution_tool_supports_aliases_and_namespaces() -> None:
    assert is_shell_execution_tool("execute")
    assert is_shell_execution_tool("bash")
    assert is_shell_execution_tool("shell")
    assert is_shell_execution_tool("server.execute")
    assert is_shell_execution_tool("agent:shell")
    assert not is_shell_execution_tool("read_text_file")


def test_extract_non_command_args_keeps_other_fields() -> None:
    tool_args = {"command": "echo hi", "cwd": "/tmp", "timeout": 30}

    remaining = extract_non_command_args(tool_args)

    assert remaining == {"cwd": "/tmp", "timeout": 30}


def test_style_apply_patch_preview_text_applies_diff_line_styles() -> None:
    text = (
        "apply_patch preview: 1 file (1 update)\n"
        "*** Begin Patch\n"
        "*** Update File: a.txt\n"
        "@@\n"
        " context\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )

    styled = style_apply_patch_preview_text(text, default_style="dim")
    span_styles = {str(span.style) for span in styled.spans}

    assert styled.plain == text
    assert "dim" in span_styles
    assert "cyan" in span_styles
    assert "yellow" in span_styles
    assert "red" in span_styles
    assert "green" in span_styles


def test_style_apply_patch_preview_text_handles_leading_spaces() -> None:
    text = (
        "  apply_patch preview: streaming patch (partial)\n"
        "  *** Update File: a.txt\n"
        "  @@\n"
        "  -old\n"
        "  +new\n"
    )

    styled = style_apply_patch_preview_text(text, default_style="dim")
    span_styles = {str(span.style) for span in styled.spans}

    assert "bold white" in span_styles
    assert "cyan" in span_styles
    assert "yellow" in span_styles
    assert "red" in span_styles
    assert "green" in span_styles


def test_style_apply_patch_preview_text_dims_context_lines_by_default() -> None:
    text = (
        "apply_patch preview: 1 file (1 update)\n"
        "*** Begin Patch\n"
        "@@\n"
        " context\n"
        "*** End Patch\n"
    )

    styled = style_apply_patch_preview_text(text)
    context_start = text.index(" context")
    context_end = context_start + len(" context")

    assert any(
        span.start <= context_start and span.end >= context_end and str(span.style) == "dim"
        for span in styled.spans
    )
