import pytest

from fast_agent.ui.syntax_highlighting import (
    SyntaxBlock,
    python_syntax_blocks,
    shell_syntax_blocks,
)


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


def test_shell_syntax_blocks_highlights_direct_interpreter_heredoc() -> None:
    command = "python - <<'PY'\nprint('hello')\nPY\nrm -rf /tmp/example"

    blocks = shell_syntax_blocks(command, shell_language="bash")

    assert blocks == [
        SyntaxBlock(code="python - <<'PY'", language="bash"),
        SyntaxBlock(code="print('hello')", language="python"),
        SyntaxBlock(code="PY\nrm -rf /tmp/example", language="bash"),
    ]


def test_shell_syntax_blocks_highlights_incomplete_direct_interpreter_body_when_enabled() -> None:
    command = "python - <<'PY'\nprint('hello')"

    assert shell_syntax_blocks(
        command,
        shell_language="bash",
        include_incomplete=True,
    ) == [
        SyntaxBlock(code="python - <<'PY'", language="bash"),
        SyntaxBlock(code="print('hello')", language="python"),
    ]


def test_shell_syntax_blocks_highlights_multiline_python_c_argument() -> None:
    command = (
        'cd /tmp/plane && /usr/bin/python3 -c "\n'
        "from PIL import Image\n"
        "import numpy as np\n"
        "for y in range(780, 1010, 6):\n"
        "    # also sample a pixel inside the fuselage\n"
        "    print(f'y={y}')\n"
        '" | head -30'
    )

    assert shell_syntax_blocks(command, shell_language="bash") == [
        SyntaxBlock(code='cd /tmp/plane && /usr/bin/python3 -c "', language="bash"),
        SyntaxBlock(
            code=(
                "from PIL import Image\n"
                "import numpy as np\n"
                "for y in range(780, 1010, 6):\n"
                "    # also sample a pixel inside the fuselage\n"
                "    print(f'y={y}')"
            ),
            language="python",
        ),
        SyntaxBlock(code='" | head -30', language="bash"),
    ]


def test_shell_syntax_blocks_highlights_uv_run_python_heredoc() -> None:
    command = (
        "uv run --with pyarrow python - <<'PY'\n"
        "from importlib.metadata import metadata, version\n"
        "for name in ('mcp','mcp-types'):\n"
        " print(name, version(name))\n"
        "PY"
    )

    blocks = shell_syntax_blocks(command, shell_language="bash")

    assert [block.language for block in blocks] == ["bash", "python", "bash"]
    assert blocks[1].code.startswith("from importlib.metadata import")


def test_shell_syntax_blocks_highlights_incomplete_pnpm_exec_tsx_heredoc() -> None:
    command = (
        "pnpm -C packages/app exec tsx - <<'TS'\n"
        "import { Client } from '@modelcontextprotocol/client';\n"
        "const client = new Client({ name: 'smoke', version: '1.0.0' });"
    )

    blocks = shell_syntax_blocks(
        command,
        shell_language="bash",
        include_incomplete=True,
    )

    assert [block.language for block in blocks] == ["bash", "typescript"]
    assert blocks[1].code.startswith("import { Client }")


@pytest.mark.parametrize(
    ("command", "language"),
    [
        ("node - <<'JS'\nconsole.log('hello')\nJS", "javascript"),
        ("ruby - <<'RB'\nputs 'hello'\nRB", "ruby"),
    ],
)
def test_shell_syntax_blocks_highlights_allowlisted_stdin_interpreter(
    command: str,
    language: str,
) -> None:
    blocks = shell_syntax_blocks(command, shell_language="bash")

    assert [block.language for block in blocks] == ["bash", language, "bash"]


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'PY'\nprint('hello')\nPY",
        "python -c 'print(1)' - <<'PY'\nprint('hello')\nPY",
        "unknown - <<'PY'\nprint('hello')\nPY",
        "python - <<'PY'\nprint('hello')",
    ],
)
def test_shell_syntax_blocks_keeps_ambiguous_interpreter_heredoc_as_shell(
    command: str,
) -> None:
    assert shell_syntax_blocks(command, shell_language="bash") == [
        SyntaxBlock(code=command, language="bash")
    ]


def test_shell_syntax_blocks_highlights_node_e_inline_payload() -> None:
    command = (
        'cd /tmp/example && node -e "\n'
        "import puppeteer from 'puppeteer';\n"
        "const browser = await puppeteer.launch();\n"
        "console.log(browser);\n"
        '" 2>&1'
    )

    blocks = shell_syntax_blocks(command, shell_language="bash")

    assert [block.language for block in blocks] == ["bash", "javascript", "bash"]
    assert blocks[0].code == 'cd /tmp/example && node -e "'
    assert "import puppeteer from 'puppeteer';" in blocks[1].code
    assert blocks[2].code == '" 2>&1'


@pytest.mark.parametrize(
    ("command", "language", "snippet"),
    [
        ("python -c 'print(1)'", "python", "print(1)"),
        ("python3.14 -c \"print('hi')\"", "python", "print('hi')"),
        ("uv run --no-sync python -c 'print(2)'", "python", "print(2)"),
        ("node -e 'console.log(1)'", "javascript", "console.log(1)"),
        ("ruby -e 'puts 1'", "ruby", "puts 1"),
        ("perl -e 'print 1'", "perl", "print 1"),
        ("php -r 'echo 1;'", "php", "echo 1;"),
        ("lua -e 'print(1)'", "lua", "print(1)"),
        ("osascript -e 'display dialog \"hi\"'", "applescript", 'display dialog "hi"'),
    ],
)
def test_shell_syntax_blocks_highlights_inline_interpreter_payload(
    command: str,
    language: str,
    snippet: str,
) -> None:
    blocks = shell_syntax_blocks(command, shell_language="bash")

    assert any(block.language == language and snippet in block.code for block in blocks)


@pytest.mark.parametrize(
    "command",
    [
        "python script.py",
        "node server.js",
        "python -m pytest",
        "grep -e pattern file.txt",
        "bash -c 'echo hi'",
        "python -c",
        "node -e",
    ],
)
def test_shell_syntax_blocks_does_not_split_non_inline_commands(command: str) -> None:
    assert shell_syntax_blocks(command, shell_language="bash") == [
        SyntaxBlock(code=command, language="bash")
    ]


def test_shell_syntax_blocks_prefers_heredoc_over_inline_inside_body() -> None:
    command = "python - <<'PY'\nprint('python -c ignore')\nPY\npython -c 'print(1)'"

    blocks = shell_syntax_blocks(command, shell_language="bash")

    assert [block.language for block in blocks] == ["bash", "python", "bash", "python", "bash"]
    assert blocks[1].code == "print('python -c ignore')"
    assert blocks[3].code == "print(1)"
    assert blocks[4].code == "'"


def test_python_syntax_blocks_highlights_shell_heredoc_inside_triple_quoted_string() -> None:
    code = (
        'r = await shell(command=r"""uv run python - <<\'PY\'\n'
        "import asyncio\n"
        'print("hi", 42)\n'
        'PY""")\n'
        "print(show(r))"
    )

    blocks = python_syntax_blocks(code)

    assert blocks == [
        SyntaxBlock(code='r = await shell(command=r"""uv run python - <<\'PY\'', language="python"),
        SyntaxBlock(code='import asyncio\nprint("hi", 42)', language="python"),
        SyntaxBlock(code='PY""")', language="text"),
        SyntaxBlock(code="print(show(r))", language="python"),
    ]


def test_python_syntax_blocks_highlights_file_target_heredoc_and_skips_fstrings() -> None:
    json_heredoc = (
        'await shell(command="""cat <<\'EOF\' > config.json\n{"enabled": true}\nEOF""")\n'
    )
    blocks = python_syntax_blocks(json_heredoc)
    assert [block.language for block in blocks] == ["python", "json", "text"]
    assert blocks[1].code == '{"enabled": true}'

    fstring_heredoc = 'await shell(command=f"""python - <<\'PY\'\nprint({value})\nPY""")\n'
    assert python_syntax_blocks(fstring_heredoc) == [
        SyntaxBlock(code=fstring_heredoc.rstrip(), language="python")
    ]


def test_python_syntax_blocks_highlights_multiline_bash_in_shell_command() -> None:
    code = (
        'r = await shell(command="""git status -s && \\\n'
        "git add \\\n"
        "  src/fast_agent/ui/syntax_highlighting.py && \\\n"
        'git commit -m "feat(ui): test" && \\\n'
        'git push""")\n'
        "print(show(r))"
    )

    blocks = python_syntax_blocks(code)

    assert blocks == [
        SyntaxBlock(code='r = await shell(command="""git status -s && \\', language="python"),
        SyntaxBlock(
            code=(
                "git add \\\n"
                "  src/fast_agent/ui/syntax_highlighting.py && \\\n"
                'git commit -m "feat(ui): test" && \\'
            ),
            language="bash",
        ),
        SyntaxBlock(code='git push""")', language="text"),
        SyntaxBlock(code="print(show(r))", language="python"),
    ]


def test_python_syntax_blocks_mixes_bash_and_heredoc_inside_shell_command() -> None:
    code = (
        'r = await shell(command=r"""\n'
        "set -u\n"
        "uv run python - <<'PY'\n"
        "print(42)\n"
        "PY\n"
        'echo "done"\n'
        '""")\n'
    )

    blocks = python_syntax_blocks(code)

    assert [block.language for block in blocks] == ["python", "bash", "python", "bash", "text"]
    assert blocks[1].code == "set -u\nuv run python - <<'PY'"
    assert blocks[2].code == "print(42)"
    assert blocks[3].code == 'PY\necho "done"'


def test_python_syntax_blocks_highlights_path_write_text_from_line_two() -> None:
    code = (
        "pathlib.Path(\"/tmp/probe_ep.py\").write_text('''import json, urllib.request\n"
        "\n"
        'base = "https://example.invalid"\n'
        "print(base)\n"
        "''')\n"
        "r = await asyncio.gather(shell(command='python3 /tmp/probe_ep.py'))"
    )

    blocks = python_syntax_blocks(code)

    assert blocks == [
        SyntaxBlock(
            code="pathlib.Path(\"/tmp/probe_ep.py\").write_text('''import json, urllib.request",
            language="python",
        ),
        SyntaxBlock(code='\nbase = "https://example.invalid"\nprint(base)', language="python"),
        SyntaxBlock(code="''')", language="text"),
        SyntaxBlock(
            code="r = await asyncio.gather(shell(command='python3 /tmp/probe_ep.py'))",
            language="python",
        ),
    ]


def test_python_syntax_blocks_highlights_edit_file_old_and_new_strings() -> None:
    code = (
        "await edit_file(\n"
        '    path="src/app.ts",\n'
        "    old_string='''\n"
        "const count = 1;\n"
        "console.log(count);\n"
        "''',\n"
        "    new_string='''\n"
        "const count = 2;\n"
        "console.log(count);\n"
        "''',\n"
        ")"
    )

    blocks = python_syntax_blocks(code)

    assert [block.language for block in blocks] == [
        "python",
        "typescript",
        "text",
        "python",
        "typescript",
        "text",
        "python",
    ]
    assert blocks[1].code == "const count = 1;\nconsole.log(count);"
    assert blocks[4].code == "const count = 2;\nconsole.log(count);"


def test_python_syntax_blocks_highlights_assigned_payloads_to_write_and_shell_sinks() -> None:
    code = (
        "script = r'''\n"
        "export const answer = 42;\n"
        "console.log(answer);\n"
        "'''\n"
        'open("example.mjs", "w").write(script)\n'
        'dockerfile = """\n'
        "FROM python:3.14-slim\n"
        'RUN echo "ok"\n'
        '"""\n'
        'await write_text_file(path="/tmp/Dockerfile", content=dockerfile)\n'
        'cmd = """\n'
        "set -e\n"
        "echo hi\n"
        '"""\n'
        "await shell(command=cmd)\n"
    )

    blocks = python_syntax_blocks(code)

    assert [
        (block.language, block.code) for block in blocks if block.language not in {"python", "text"}
    ] == [
        ("javascript", "export const answer = 42;\nconsole.log(answer);"),
        ("dockerfile", 'FROM python:3.14-slim\nRUN echo "ok"'),
        ("bash", "set -e\necho hi"),
    ]


def test_python_syntax_blocks_highlights_with_open_and_path_variable_writes() -> None:
    code = (
        'target = Path("/tmp") / "config.json"\n'
        'payload = """\n'
        '{"ok": true}\n'
        '"""\n'
        "target.write_text(data=payload)\n"
        'with open("/tmp/query.sql", "w") as handle:\n'
        '    handle.write("""\n'
        "SELECT 1;\n"
        '""")\n'
    )

    blocks = python_syntax_blocks(code)

    assert [
        (block.language, block.code) for block in blocks if block.language not in {"python", "text"}
    ] == [
        ("json", '{"ok": true}'),
        ("sql", "SELECT 1;"),
    ]


def test_python_syntax_blocks_skips_plain_text_unconnected_or_two_line_writes() -> None:
    plain = 'Path("/tmp/notes.txt").write_text("""\nhello\nworld\n""")'
    two_line = 'Path("/tmp/a.py").write_text("""x = 1\ny = 2""")'
    unassigned = 'report = """\nhello\nworld\n"""\nprint(report)'

    for code in (plain, two_line, unassigned):
        assert python_syntax_blocks(code) == [SyntaxBlock(code=code, language="python")]


def test_python_syntax_blocks_handles_utf8_prefix_before_write_literal() -> None:
    code = 'label = "—"; Path("/tmp/a.json").write_text("""\n{"ok": true}\n""")'

    blocks = python_syntax_blocks(code)

    assert [block.language for block in blocks] == ["python", "json", "text"]
    assert blocks[1].code == '{"ok": true}'
