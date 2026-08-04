import pytest

from fast_agent.tools.shell_command import (
    classify_shell_detachment,
    shell_heredoc_bodies,
    shell_inline_code_bodies,
)


@pytest.mark.parametrize(
    ("command", "run_in_background", "expected"),
    [
        ("nohup server >server.log 2>&1 &", False, "service_detach"),
        ("/usr/bin/nohup server >server.log 2>&1 &", False, "service_detach"),
        ("command nohup server &", False, "service_detach"),
        ("FOO=bar /usr/bin/nohup server &", False, "service_detach"),
        ("env FOO=bar nohup server &", False, "service_detach"),
        ("env -u OLD FOO=bar nohup server &", False, "service_detach"),
        ("exec -a service /usr/bin/nohup server &", False, "service_detach"),
        (
            "env -i PATH=/usr/bin /usr/bin/nohup server &",
            False,
            "service_detach",
        ),
        ("server & disown", False, "service_detach"),
        ("server &", True, "service_detach"),
        ("server &", False, "ambiguous"),
        ("(sleep 100 &)", False, "ambiguous"),
        ("(sleep 100 &)", True, "service_detach"),
        ("echo $((3 & 1))", False, "none"),
        ("((flags & mask))", False, "none"),
        ("echo $(((3 & 1) | 4))", False, "none"),
        ("echo $(( $(sleep 100 &) + 1 ))", False, "ambiguous"),
        ('echo "$(server >/dev/null 2>&1 &)"', False, "ambiguous"),
        ('echo "value: $((3 & 1))"', False, "none"),
        ("echo one && echo two", True, "none"),
        ("echo 'A&B' 2>&1", True, "none"),
        ("curl 'https://example.test/?a=1&b=2'", True, "none"),
        ("pytest &>results.log", False, "none"),
        ("pytest &>>results.log", False, "none"),
        ("build |& tee build.log", False, "none"),
        ("echo ok # nohup server &", True, "none"),
        ("cat <<'EOF'\nnohup server &\nEOF\n", True, "none"),
        ("cat <<\\EOF\nnohup server &\nEOF\n", True, "none"),
        ("echo '<<EOF'\nnohup server &", False, "service_detach"),
        ('echo "<<EOF"\nnohup server &', False, "service_detach"),
        ("echo 'text\n<<EOF'\nnohup server &", False, "service_detach"),
        ("echo \\<<EOF\nnohup server &", False, "service_detach"),
    ],
)
def test_shell_detachment_classifier(
    command: str,
    run_in_background: bool,
    expected: str,
) -> None:
    assert (
        classify_shell_detachment(
            command,
            run_in_background=run_in_background,
        )
        == expected
    )


def test_shell_heredoc_bodies_match_static_redirect_targets() -> None:
    command = (
        "cat <<'PY' > src/example.py\n"
        "print('hello')\n"
        "PY\n"
        'cat > "web/example.ts" <<-TS\n'
        "\texport const answer = 42;\n"
        "\tTS\n"
    )

    bodies = shell_heredoc_bodies(command)

    assert [body.target_path for body in bodies] == ["src/example.py", "web/example.ts"]
    assert [command[body.start : body.end] for body in bodies] == [
        "print('hello')\n",
        "\texport const answer = 42;\n",
    ]


def test_shell_heredoc_bodies_match_direct_stdin_interpreter() -> None:
    command = "/usr/bin/python3.14 - <<'PY'\nprint('hello')\nPY\n"

    body = shell_heredoc_bodies(command)[0]

    assert body.target_path is None
    assert body.stdin_interpreter == "python3.14"


def test_shell_heredoc_bodies_can_include_incomplete_direct_interpreter_body() -> None:
    command = "python - <<'PY'\nprint('hello')"

    assert shell_heredoc_bodies(command) == []

    body = shell_heredoc_bodies(command, include_incomplete=True)[0]
    assert body.stdin_interpreter == "python"
    assert command[body.start : body.end] == "print('hello')"


@pytest.mark.parametrize(
    "command",
    [
        "uv run python - <<'PY'\nprint('hello')\nPY\n",
        "uv run --no-sync python3.14 - <<'PY'\nprint('hello')\nPY\n",
    ],
)
def test_shell_heredoc_bodies_match_uv_run_stdin_interpreter(command: str) -> None:
    body = shell_heredoc_bodies(command)[0]

    assert body.stdin_interpreter in {"python", "python3.14"}
    assert command[body.start : body.end] == "print('hello')\n"


@pytest.mark.parametrize(
    "command",
    [
        "pnpm exec tsx - <<'TS'\nconst answer = 42;\nTS\n",
        "pnpm -C packages/app exec tsx - <<'TS'\nconst answer = 42;\nTS\n",
        "pnpm --dir=packages/app exec tsx - <<'TS'\nconst answer = 42;\nTS\n",
    ],
)
def test_shell_heredoc_bodies_match_pnpm_exec_stdin_interpreter(command: str) -> None:
    body = shell_heredoc_bodies(command)[0]

    assert body.stdin_interpreter == "tsx"
    assert command[body.start : body.end] == "const answer = 42;\n"


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'PY'\nprint('hello')\nPY\n",
        "python -c 'print(1)' - <<'PY'\nprint('hello')\nPY\n",
        "uv run --python 3.14 python - <<'PY'\nprint('hello')\nPY\n",
        "uv run echo python - <<'PY'\nprint('hello')\nPY\n",
        "pnpm --filter app exec tsx - <<'TS'\nconst answer = 42;\nTS\n",
        "pnpm echo exec tsx - <<'TS'\nconst answer = 42;\nTS\n",
        "cat <<A <<B\nfirst\nA\nsecond\nB\n",
    ],
)
def test_shell_heredoc_bodies_do_not_guess_stdin_interpreter(command: str) -> None:
    bodies = shell_heredoc_bodies(command)

    assert bodies
    assert all(body.stdin_interpreter is None for body in bodies)


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'EOF'\ntext\nEOF\n",
        "cat > \"$OUTPUT.py\" <<'EOF'\ntext\nEOF\n",
        "cat > one.py <<A <<B\nfirst\nA\nsecond\nB\n",
        "cat > incomplete.py <<'EOF'\ntext\n",
        "echo ignored > unrelated.py; cat <<'EOF'\ntext\nEOF\n",
        "# cat > commented.py <<'EOF'\ntext\nEOF\n",
        "((value<<EOF))\ntext\nEOF\n",
    ],
)
def test_shell_heredoc_bodies_leave_ambiguous_or_incomplete_targets_unsplit(command: str) -> None:
    assert not [body for body in shell_heredoc_bodies(command) if body.target_path is not None]


def test_shell_heredoc_body_preserves_trailing_whitespace() -> None:
    command = "cat 1> example.md <<'EOF'\ntrailing spaces  \nEOF\n"

    body = shell_heredoc_bodies(command)[0]

    assert body.target_path == "example.md"
    assert command[body.start : body.end] == "trailing spaces  \n"


def test_shell_heredoc_single_quoted_dollar_path_is_static() -> None:
    command = "cat > '$OUTPUT.py' <<'EOF'\nprint('literal path')\nEOF\n"

    body = shell_heredoc_bodies(command)[0]

    assert body.target_path == "$OUTPUT.py"


def test_shell_heredoc_uses_final_stdout_redirect() -> None:
    command = "cat > ignored.py > actual.txt <<'EOF'\nplain text\nEOF\n"

    body = shell_heredoc_bodies(command)[0]

    assert body.target_path == "actual.txt"


def test_shell_inline_code_bodies_match_multiline_python_c_argument() -> None:
    command = (
        'cd /tmp/plane && /usr/bin/python3 -c "\n'
        "from PIL import Image\n"
        "for y in range(780, 1010, 6):\n"
        "    # also sample a pixel inside the fuselage\n"
        "    print(f'y={y}')\n"
        '" | head -30'
    )

    body = shell_inline_code_bodies(command)[0]

    assert body.interpreter == "python3"
    assert command[body.start : body.end] == (
        "from PIL import Image\n"
        "for y in range(780, 1010, 6):\n"
        "    # also sample a pixel inside the fuselage\n"
        "    print(f'y={y}')\n"
    )


def test_shell_inline_code_bodies_can_include_incomplete_python_c_argument() -> None:
    command = 'python -c "\nprint('

    assert shell_inline_code_bodies(command) == []

    body = shell_inline_code_bodies(command, include_incomplete=True)[0]
    assert body.interpreter == "python"
    assert command[body.start : body.end] == "print("


@pytest.mark.parametrize(
    "command",
    [
        'python -c "\nprint(1)\n">result.txt',
        'python \\\n-c "\nprint(1)\n"',
    ],
)
def test_shell_inline_code_bodies_accept_shell_argument_boundaries(command: str) -> None:
    body = shell_inline_code_bodies(command)[0]

    assert command[body.start : body.end] == "print(1)\n"


@pytest.mark.parametrize(
    "command",
    [
        'python -c "print(1)"',
        '# python -c "\nprint(1)\n"',
        'node -c "\nconsole.log(1)\n"',
        'python -c "\nprint(\\"$HOME\\")\n"',
        'python -c "\nprint(1)\n"suffix',
    ],
)
def test_shell_inline_code_bodies_do_not_guess_ambiguous_arguments(command: str) -> None:
    assert shell_inline_code_bodies(command, include_incomplete=True) == []


def test_shell_inline_code_bodies_do_not_overlap_heredoc_bodies() -> None:
    command = "cat > out.js <<'EOF'; python -c '\nprint(1)\n'\nconst answer = 42;\nEOF"

    assert shell_inline_code_bodies(command, include_incomplete=True) == []
