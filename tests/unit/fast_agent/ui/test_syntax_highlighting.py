from fast_agent.ui.syntax_highlighting import shell_syntax_blocks


def test_shell_syntax_blocks_highlights_static_heredoc_body_by_target_extension() -> None:
    command = "cat > example.py <<'PY'\nprint('hello')\nPY\npython example.py"

    blocks = shell_syntax_blocks(command, shell_language="bash")

    assert [block.language for block in blocks] == ["bash", "python", "bash"]
    assert [block.code for block in blocks] == [
        "cat > example.py <<'PY'",
        "print('hello')",
        "PY\npython example.py",
    ]


def test_shell_syntax_blocks_keeps_incomplete_or_dynamic_heredocs_as_shell() -> None:
    incomplete = "cat > example.py <<'PY'\nprint('hello')"
    dynamic = "cat > \"$OUTPUT.py\" <<'PY'\nprint('hello')\nPY"

    incomplete_blocks = shell_syntax_blocks(incomplete, shell_language="bash")
    dynamic_blocks = shell_syntax_blocks(dynamic, shell_language="bash")

    assert len(incomplete_blocks) == 1
    assert incomplete_blocks[0].language == "bash"
    assert incomplete_blocks[0].code == incomplete
    assert len(dynamic_blocks) == 1
    assert dynamic_blocks[0].language == "bash"
    assert dynamic_blocks[0].code == dynamic
