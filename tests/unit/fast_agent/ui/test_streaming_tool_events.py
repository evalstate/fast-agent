import json

from fast_agent.ui.stream_segments import StreamSegmentAssembler, extract_partial_json_string_field


def _make_assembler(
    *,
    tool_metadata_resolver=None,
    apply_patch_preview_max_lines=None,
) -> StreamSegmentAssembler:
    return StreamSegmentAssembler(
        base_kind="markdown",
        tool_prefix="->",
        tool_metadata_resolver=tool_metadata_resolver,
        apply_patch_preview_max_lines=apply_patch_preview_max_lines,
    )


def test_tool_stream_delta_bootstraps_mode() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "delta", {"tool_name": "search", "tool_use_id": "tool-1", "chunk": "{\"q\":1}"}
    )

    text = "".join(segment.text for segment in assembler.segments)
    assert "-> search" in text
    assert "{\"q\":1}" in text

    assembler.handle_tool_event("stop", {"tool_name": "search", "tool_use_id": "tool-1"})
    text = "".join(segment.text for segment in assembler.segments)
    assert "\"q\": 1" in text


def test_tool_stream_status_updates_visible_text() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "start",
        {
            "tool_name": "web_search",
            "tool_use_id": "ws-1",
            "chunk": "starting search...",
        },
    )
    assembler.handle_tool_event(
        "status",
        {
            "tool_name": "web_search",
            "tool_use_id": "ws-1",
            "chunk": "searching...",
            "status": "searching",
        },
    )
    assembler.handle_tool_event(
        "status",
        {
            "tool_name": "web_search",
            "tool_use_id": "ws-1",
            "chunk": "search complete",
            "status": "completed",
        },
    )
    assembler.handle_tool_event("stop", {"tool_name": "web_search", "tool_use_id": "ws-1"})

    text = "".join(segment.text for segment in assembler.segments)
    assert "Searching the web" in text
    assert "search complete" in text
    assert "starting search..." not in text


def test_tool_stream_status_uses_fallback_chunk_when_missing() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "status",
        {
            "tool_name": "web_search_call",
            "tool_use_id": "ws-2",
            "status": "searching",
        },
    )
    assembler.handle_tool_event("stop", {"tool_name": "web_search_call", "tool_use_id": "ws-2"})

    text = "".join(segment.text for segment in assembler.segments)
    assert "Searching the web" in text
    assert "searching..." in text


def test_tool_stream_status_for_mcp_does_not_use_search_copy() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "status",
        {
            "tool_name": "mcp_list_tools",
            "presentation_family": "remote_tool_listing",
            "tool_display_name": "Loading remote tools",
            "tool_use_id": "mcp-2",
            "status": "completed",
        },
    )
    assembler.handle_tool_event(
        "stop",
        {
            "tool_name": "mcp_list_tools",
            "presentation_family": "remote_tool_listing",
            "tool_display_name": "Loading remote tools",
            "tool_use_id": "mcp-2",
        },
    )

    text = "".join(segment.text for segment in assembler.segments)
    assert "Loading remote tools" in text
    assert "remote tools loaded" in text
    assert "search complete" not in text


def test_tool_stream_replace_resets_snapshot_content() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "start",
        {
            "tool_name": "search",
            "tool_use_id": "mcp-1",
            "chunk": "{}",
        },
    )
    assembler.handle_tool_event(
        "replace",
        {
            "tool_name": "search",
            "tool_use_id": "mcp-1",
            "chunk": "{}",
        },
    )
    assembler.handle_tool_event(
        "stop",
        {"tool_name": "search", "tool_use_id": "mcp-1"},
    )

    text = "".join(segment.text for segment in assembler.segments)
    assert "{}{}" not in text
    assert text.count("{}") == 1


def test_tool_stream_remote_labels_are_explicit() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "start",
        {
            "tool_name": "huggingface_mcp/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
            "chunk": "{}",
        },
    )
    assembler.handle_tool_event(
        "replace",
        {
            "tool_name": "huggingface_mcp/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1:result",
            "chunk": "evalstate",
        },
    )
    assembler.handle_tool_event(
        "stop",
        {
            "tool_name": "huggingface_mcp/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1:result",
        },
    )

    text = "\n".join(segment.text for segment in assembler.segments)
    assert "remote tool: hf_whoami" in text


def test_remote_tool_stream_preserves_args_status_and_result() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
            "chunk": "{}",
        },
    )
    assembler.handle_tool_event(
        "status",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
            "chunk": "calling remote tool...",
        },
    )
    assembler.handle_tool_event(
        "replace",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
            "chunk": "You are authenticated as evalstate.",
        },
    )

    in_progress_text = "".join(segment.text for segment in assembler.segments)
    assert "remote tool: hf_whoami" in in_progress_text
    assert "status: calling remote tool..." in in_progress_text
    assert "result: You are authenticated as evalstate." in in_progress_text
    assert "args: {}" in in_progress_text

    assembler.handle_tool_event(
        "stop",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
        },
    )

    text = "".join(segment.text for segment in assembler.segments)
    assert "remote tool: hf_whoami" in text
    assert "{}" in text
    assert "{}\n\nYou are authenticated as evalstate." in text
    assert "You are authenticated as evalstate." in text
    assert "status:" not in text
    assert "result:" not in text
    assert "args:" not in text


def test_remote_tool_stream_result_only_does_not_duplicate_completed_output() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "start",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
        },
    )
    assembler.handle_tool_event(
        "replace",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
            "chunk": "You are authenticated as evalstate.",
        },
    )
    assembler.handle_tool_event(
        "stop",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
        },
    )

    text = "".join(segment.text for segment in assembler.segments)
    assert text.count("You are authenticated as evalstate.") == 1


def test_remote_tool_stream_failed_blob_does_not_duplicate_completed_output() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "replace",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
            "chunk": "status: failed\nresult: forbidden",
        },
    )
    assembler.handle_tool_event(
        "stop",
        {
            "tool_name": "huggingface/hf_whoami",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_whoami",
            "tool_use_id": "mcp-1",
        },
    )

    text = "".join(segment.text for segment in assembler.segments)
    assert text.count("status: failed\nresult: forbidden") == 1


def test_remote_tool_search_collapses_to_compact_completed_status() -> None:
    assembler = _make_assembler()

    assembler.handle_tool_event(
        "start",
        {
            "tool_name": "tool_search",
            "presentation_family": "remote_tool_search",
            "preserve_details": True,
            "tool_display_name": "Deferred tool search",
            "tool_use_id": "search-1",
            "chunk": "searching deferred tools...",
        },
    )
    assembler.handle_tool_event(
        "replace",
        {
            "tool_name": "tool_search",
            "presentation_family": "remote_tool_search",
            "preserve_details": True,
            "tool_display_name": "Deferred tool search",
            "tool_use_id": "search-1",
            "chunk": "deferred tool search complete",
        },
    )
    assembler.handle_tool_event(
        "stop",
        {
            "tool_name": "tool_search",
            "presentation_family": "remote_tool_search",
            "preserve_details": True,
            "tool_display_name": "Deferred tool search",
            "tool_use_id": "search-1",
        },
    )

    text = "".join(segment.text for segment in assembler.segments)
    assert "Deferred tool search" in text
    assert "deferred tool search complete" in text
    assert "args:" not in text
    assert "result:" not in text


def test_tool_stream_apply_patch_preview_keeps_other_args() -> None:
    assembler = _make_assembler()
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Add File: a.txt\n"
        "+hello\n"
        "*** End Patch\n"
        "PATCH"
    )
    args_chunk = json.dumps(
        {"command": command, "cwd": "/tmp/work", "timeout_seconds": 90},
    )

    assembler.handle_tool_event(
        "delta",
        {"tool_name": "execute", "tool_use_id": "tool-apply-1", "chunk": args_chunk},
    )
    assembler.handle_tool_event("stop", {"tool_name": "execute", "tool_use_id": "tool-apply-1"})

    text = "".join(segment.text for segment in assembler.segments)
    assert "apply_patch preview:" in text
    assert "*** Begin Patch" in text
    assert "other args:" in text
    assert '"cwd": "/tmp/work"' in text
    assert '"timeout_seconds": 90' in text


def test_tool_stream_apply_patch_preview_supports_shell_aliases() -> None:
    assembler = _make_assembler()
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Delete File: a.txt\n"
        "*** End Patch\n"
        "PATCH"
    )
    args_chunk = json.dumps(
        {"command": command},
    )

    assembler.handle_tool_event(
        "delta",
        {"tool_name": "bash", "tool_use_id": "tool-apply-2", "chunk": args_chunk},
    )
    assembler.handle_tool_event("stop", {"tool_name": "bash", "tool_use_id": "tool-apply-2"})

    text = "".join(segment.text for segment in assembler.segments)
    assert "apply_patch preview:" in text
    assert "*** Delete File: a.txt" in text


def test_tool_stream_direct_apply_patch_preview_appears_before_stop() -> None:
    assembler = _make_assembler()
    partial_chunk = "*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new"

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "apply_patch",
            "tool_use_id": "tool-apply-direct-1",
            "chunk": partial_chunk,
        },
    )

    segment = assembler.segments[0]
    assert segment.code_preview is None
    assert "apply_patch preview: streaming patch" in segment.text
    assert "*** Update File: a.txt" in segment.text
    assert "-old" in segment.text
    assert "+new" in segment.text


def test_extract_partial_json_string_field_decodes_incomplete_code_value() -> None:
    extracted = extract_partial_json_string_field(
        '{"query":"count","code":"resp = await hf_trending()\\nprin',
        field_name="code",
    )

    assert extracted is not None
    assert extracted.key == "code"
    assert extracted.value == "resp = await hf_trending()\nprin"
    assert extracted.complete is False


def test_tool_stream_code_preview_tracks_partial_code() -> None:
    metadata = {
        "variant": "code",
        "code_arg": "code",
        "language": "python",
    }
    assembler = _make_assembler(
        tool_metadata_resolver=lambda tool_name: metadata if tool_name == "hf_hub_query_raw" else None
    )

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "hf_hub_query_raw",
            "tool_use_id": "tool-code-1",
            "chunk": '{"query":"count","code":"resp = await hf_trending()\\nprin',
        },
    )

    assert len(assembler.segments) == 1
    preview = assembler.segments[0].code_preview
    assert preview is not None
    assert preview.language == "python"
    assert preview.code == "resp = await hf_trending()\nprin"
    assert preview.complete is False


def test_tool_stream_shell_preview_tracks_partial_command() -> None:
    metadata = {
        "variant": "shell",
        "shell_name": "bash",
    }
    assembler = _make_assembler(
        tool_metadata_resolver=lambda tool_name: metadata if tool_name == "execute" else None
    )

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "execute",
            "tool_use_id": "tool-shell-1",
            "chunk": '{"command":"uv run scripts/lint.py && uv run scr',
        },
    )

    assert len(assembler.segments) == 1
    preview = assembler.segments[0].code_preview
    assert preview is not None
    assert preview.language == "bash"
    assert preview.code == "uv run scripts/lint.py && uv run scr"
    assert preview.complete is False


def test_tool_stream_shell_preview_skips_apply_patch_commands() -> None:
    metadata = {
        "variant": "shell",
        "shell_name": "bash",
    }
    assembler = _make_assembler(
        tool_metadata_resolver=lambda tool_name: metadata if tool_name == "execute" else None
    )
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Add File: a.txt\n"
        "+hello\n"
        "*** End Patch\n"
        "PATCH"
    )

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "execute",
            "tool_use_id": "tool-shell-2",
            "chunk": json.dumps({"command": command}),
        },
    )
    assembler.handle_tool_event("stop", {"tool_name": "execute", "tool_use_id": "tool-shell-2"})

    assert assembler.segments[0].code_preview is None
    assert "apply_patch preview:" in assembler.segments[0].text


def test_tool_stream_apply_patch_preview_appears_before_stop() -> None:
    metadata = {
        "variant": "shell",
        "shell_name": "bash",
    }
    assembler = _make_assembler(
        tool_metadata_resolver=lambda tool_name: metadata if tool_name == "execute" else None
    )
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Add File: a.txt\n"
        "+hello\n"
        "*** End Patch\n"
        "PATCH"
    )
    partial_chunk = (
        '{"command":"'
        + command.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "execute",
            "tool_use_id": "tool-shell-3",
            "chunk": partial_chunk,
        },
    )

    assert assembler.segments[0].code_preview is None
    assert "apply_patch preview:" in assembler.segments[0].text
    assert "*** Begin Patch" in assembler.segments[0].text


def test_tool_stream_apply_patch_preview_colours_partial_patch_lines() -> None:
    metadata = {
        "variant": "shell",
        "shell_name": "bash",
    }
    assembler = _make_assembler(
        tool_metadata_resolver=lambda tool_name: metadata if tool_name == "execute" else None
    )
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Update File: a.txt\n"
        "@@\n"
        "-old\n"
        "+new"
    )
    partial_chunk = (
        '{"command":"'
        + command.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "execute",
            "tool_use_id": "tool-shell-4",
            "chunk": partial_chunk,
        },
    )

    segment = assembler.segments[0]
    assert segment.code_preview is None
    assert "apply_patch preview: streaming patch" in segment.text
    assert "*** Update File: a.txt" in segment.text
    assert "-old" in segment.text
    assert "+new" in segment.text


def test_tool_stream_apply_patch_preview_respects_line_limit() -> None:
    metadata = {
        "variant": "shell",
        "shell_name": "bash",
    }
    assembler = _make_assembler(
        tool_metadata_resolver=lambda tool_name: metadata if tool_name == "execute" else None,
        apply_patch_preview_max_lines=4,
    )
    command = (
        "apply_patch <<'PATCH'\n"
        "*** Begin Patch\n"
        "*** Add File: a.txt\n"
        "+line-1\n"
        "+line-2\n"
        "+line-3\n"
        "*** End Patch\n"
        "PATCH"
    )

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "execute",
            "tool_use_id": "tool-shell-5",
            "chunk": json.dumps({"command": command}),
        },
    )

    assert "(+2 more lines)" in assembler.segments[0].text


def test_tool_stream_code_preview_uses_namespaced_tool_metadata() -> None:
    metadata = {
        "variant": "code",
        "code_arg": "code",
        "language": "python",
    }
    assembler = _make_assembler(
        tool_metadata_resolver=lambda tool_name: (
            metadata if tool_name == "huggingface_mcp/hf_hub_query_raw" else None
        )
    )

    assembler.handle_tool_event(
        "delta",
        {
            "tool_name": "huggingface_mcp/hf_hub_query_raw",
            "presentation_family": "remote_tool",
            "preserve_details": True,
            "tool_display_name": "remote tool: hf_hub_query_raw",
            "tool_use_id": "tool-code-2",
            "chunk": '{"query":"count","code":"resp = await hf_trending()\\nprin',
        },
    )

    assert len(assembler.segments) == 1
    preview = assembler.segments[0].code_preview
    assert preview is not None
    assert preview.language == "python"
    assert preview.code == "resp = await hf_trending()\nprin"
    assert preview.complete is False
