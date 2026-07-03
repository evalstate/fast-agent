from __future__ import annotations

import subprocess
import sys

import pytest
from click.utils import strip_ansi

from fast_agent.cli import __main__ as cli_main
from fast_agent.cli.__main__ import _first_positional_argument


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--home", "demo", "demo", "--help"], "demo"),
        (["--home", "demo", "--message", "hi"], None),
        (["-m", "serve", "--help"], None),
        (["--home=demo", "cards", "--help"], "cards"),
        (["--", "demo"], "demo"),
    ],
)
def test_first_positional_argument_skips_option_values(
    arguments: list[str],
    expected: str | None,
) -> None:
    assert _first_positional_argument(arguments) == expected


def _run_fast_agent_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fast_agent.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_routed_help(output: str, command: str) -> None:
    assert f"Usage: python -m fast_agent.cli {command} [OPTIONS]" in output
    assert "COMMAND" in output


def test_auto_routes_to_go_when_env_value_matches_subcommand() -> None:
    result = _run_fast_agent_cli("--home", "demo", "--message", "hi", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    _assert_routed_help(output, "go")
    assert "--message" in output


def test_auto_routes_to_go_when_message_matches_subcommand() -> None:
    result = _run_fast_agent_cli("-m", "serve", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    _assert_routed_help(output, "go")


def test_auto_routes_to_go_with_trailing_quiet_option() -> None:
    result = _run_fast_agent_cli("-m", "hello", "-q", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    _assert_routed_help(output, "go")
    assert "--quiet" in output


def test_auto_routes_to_go_with_json_schema_option() -> None:
    result = _run_fast_agent_cli("--json-schema", "schema.json", "--message", "hello", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    _assert_routed_help(output, "go")
    assert "--json-schema" in output


def test_auto_routes_to_go_when_pack_flag_used_at_root() -> None:
    result = _run_fast_agent_cli("--pack", "alpha", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    _assert_routed_help(output, "go")
    assert "--pack" in output
    assert "--pack-registry" in output


def test_auto_routes_to_go_when_no_shell_used_at_root() -> None:
    result = _run_fast_agent_cli("--no-shell", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    _assert_routed_help(output, "go")
    assert "--no-shell" in output


def test_resume_sentinel_is_not_added_for_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "batch", "run", "--resume"])
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == ["fast-agent", "batch", "run", "--resume"]


def test_resume_sentinel_is_added_for_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "go", "--resume"])
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == ["fast-agent", "go", "--resume", "__latest__"]


def test_root_resume_auto_routes_to_go_and_adds_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "--resume"])
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == ["fast-agent", "go", "--resume", "__latest__"]


def test_root_serve_mcp_routes_to_serve_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "--serve", "mcp", "--model", "haiku"])
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == ["fast-agent", "serve", "--model", "haiku"]


def test_root_serve_transport_routes_to_serve_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "--home", "demo", "--serve=stdio"])
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == ["fast-agent", "--home", "demo", "serve", "--transport", "stdio"]


def test_demo_subcommand_still_detected_after_env_option_value() -> None:
    result = _run_fast_agent_cli("--home", "demo", "demo", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    _assert_routed_help(output, "demo")
    assert "Demo commands for UI features." in output


def test_main_converts_keyboard_interrupt_to_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "app", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main()

    assert exc_info.value.code == 130
