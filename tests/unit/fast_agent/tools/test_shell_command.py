import pytest

from fast_agent.tools.shell_command import classify_shell_detachment, shell_heredoc_bodies


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
        "cat <<'PY'\nprint('hello')\nPY\n",
        "python -c 'print(1)' - <<'PY'\nprint('hello')\nPY\n",
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
