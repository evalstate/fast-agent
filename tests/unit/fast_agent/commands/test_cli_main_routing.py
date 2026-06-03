from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import pytest
from click.utils import strip_ansi

from fast_agent.cli import __main__ as cli_main
from fast_agent.cli import main as cli_main_module
from fast_agent.cli.__main__ import _first_positional_argument
from fast_agent.cli.main import _installed_package_version, _resolve_root_verbosity


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--env", "demo", "demo", "--help"], "demo"),
        (["--env", "demo", "--message", "hi"], None),
        (["-m", "serve", "--help"], None),
        (["--env=demo", "cards", "--help"], "cards"),
        (["--", "demo"], "demo"),
    ],
)
def test_first_positional_argument_skips_option_values(
    arguments: list[str],
    expected: str | None,
) -> None:
    assert _first_positional_argument(arguments) == expected


@pytest.mark.parametrize(
    ("verbose", "quiet", "expected"),
    [
        (True, False, 1),
        (True, True, 1),
        (False, False, 0),
        (False, True, -1),
    ],
)
def test_resolve_root_verbosity_prefers_verbose_over_quiet(
    verbose: bool,
    quiet: bool,
    expected: int,
) -> None:
    assert _resolve_root_verbosity(verbose=verbose, quiet=quiet) == expected


def test_installed_package_version_returns_metadata_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_main_module, "package_version", lambda _name: "1.2.3")

    assert _installed_package_version("fast-agent-mcp") == "1.2.3"


def test_installed_package_version_returns_unknown_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_package(_name: str) -> str:
        raise PackageNotFoundError("fast-agent-mcp")

    monkeypatch.setattr(cli_main_module, "package_version", missing_package)

    assert _installed_package_version("fast-agent-mcp") == "unknown"


def _run_fast_agent_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fast_agent.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_auto_routes_to_go_when_env_value_matches_subcommand() -> None:
    result = _run_fast_agent_cli("--env", "demo", "--message", "hi", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
    assert "--message" in output


def test_auto_routes_to_go_when_message_matches_subcommand() -> None:
    result = _run_fast_agent_cli("-m", "serve", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output


def test_auto_routes_to_go_with_trailing_quiet_option() -> None:
    result = _run_fast_agent_cli("-m", "hello", "-q", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
    assert "--quiet" in output


def test_auto_routes_to_go_with_json_schema_option() -> None:
    result = _run_fast_agent_cli("--json-schema", "schema.json", "--message", "hello", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
    assert "--json-schema" in output


def test_auto_routes_to_go_with_attach_option() -> None:
    result = _run_fast_agent_cli("--attach", "image.png", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
    assert "--attach" in output


def test_auto_routes_to_go_with_structured_tool_policy_option() -> None:
    result = _run_fast_agent_cli("--structured-tool-policy", "auto", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
    assert "--structured-tool-policy" in output


def test_auto_routes_to_go_when_pack_flag_used_at_root() -> None:
    result = _run_fast_agent_cli("--pack", "alpha", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
    assert "--pack" in output
    assert "--pack-registry" in output


def test_auto_routes_to_go_when_card_tool_used_at_root() -> None:
    result = _run_fast_agent_cli("--card-tool", "tools", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
    assert "--card-tool" in output


def test_auto_routes_to_go_when_no_shell_used_at_root() -> None:
    result = _run_fast_agent_cli("--no-shell", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
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


def test_empty_resume_equals_form_is_normalized_for_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "go", "--resume="])
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


def test_root_empty_resume_equals_form_auto_routes_to_go_and_adds_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "--resume="])
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == ["fast-agent", "go", "--resume", "__latest__"]


def test_root_resume_equals_value_auto_routes_without_rewriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "--resume=session-123"])
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == ["fast-agent", "go", "--resume=session-123"]


def test_demo_subcommand_still_detected_after_env_option_value() -> None:
    result = _run_fast_agent_cli("--env", "demo", "demo", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "demo [OPTIONS] COMMAND" in output
    assert "Demo commands for UI features." in output


def test_main_converts_keyboard_interrupt_to_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "app", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main()

    assert exc_info.value.code == 130
