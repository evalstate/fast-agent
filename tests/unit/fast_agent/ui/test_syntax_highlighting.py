import pytest

from fast_agent.ui.syntax_highlighting import SyntaxBlock, shell_syntax_blocks


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


def test_shell_syntax_blocks_highlights_incomplete_multiline_python_c_argument() -> None:
    command = 'cd /tmp && python -c "\nfrom pathlib import Path\nprint(Path.cwd())'

    assert shell_syntax_blocks(command, shell_language="bash") == [
        SyntaxBlock(code=command, language="bash")
    ]
    assert shell_syntax_blocks(
        command,
        shell_language="bash",
        include_incomplete=True,
    ) == [
        SyntaxBlock(code='cd /tmp && python -c "', language="bash"),
        SyntaxBlock(code="from pathlib import Path\nprint(Path.cwd())", language="python"),
    ]


def test_shell_syntax_blocks_highlights_uv_run_python_heredoc() -> None:
    command = (
        "uv run python - <<'PY'\n"
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
