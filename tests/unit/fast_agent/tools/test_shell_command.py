import pytest

from fast_agent.tools.shell_command import classify_shell_detachment


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
        ("echo one && echo two", True, "none"),
        ("echo 'A&B' 2>&1", True, "none"),
        ("curl 'https://example.test/?a=1&b=2'", True, "none"),
        ("pytest &>results.log", False, "none"),
        ("pytest &>>results.log", False, "none"),
        ("build |& tee build.log", False, "none"),
        ("echo ok # nohup server &", True, "none"),
        ("cat <<'EOF'\nnohup server &\nEOF\n", True, "none"),
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
