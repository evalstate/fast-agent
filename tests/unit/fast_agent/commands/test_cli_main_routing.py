from __future__ import annotations

import sys

import pytest

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


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["fast-agent", "--home", "demo", "--message", "hi", "--help"],
            ["fast-agent", "go", "--home", "demo", "--message", "hi", "--help"],
        ),
        (
            ["fast-agent", "-m", "serve", "--help"],
            ["fast-agent", "go", "-m", "serve", "--help"],
        ),
        (
            ["fast-agent", "-m", "hello", "-q", "--help"],
            ["fast-agent", "go", "-m", "hello", "-q", "--help"],
        ),
        (
            ["fast-agent", "--json-schema", "schema.json", "--message", "hello", "--help"],
            ["fast-agent", "go", "--json-schema", "schema.json", "--message", "hello", "--help"],
        ),
        (
            ["fast-agent", "--pack", "alpha", "--help"],
            ["fast-agent", "go", "--pack", "alpha", "--help"],
        ),
        (
            ["fast-agent", "--environment", "hf-gpu", "--help"],
            ["fast-agent", "go", "--environment", "hf-gpu", "--help"],
        ),
        (
            ["fast-agent", "-E", "hf-gpu", "--help"],
            ["fast-agent", "go", "-E", "hf-gpu", "--help"],
        ),
        (
            ["fast-agent", "--no-shell", "--help"],
            ["fast-agent", "go", "--no-shell", "--help"],
        ),
    ],
)
def test_root_go_options_auto_route_to_go(
    argv: list[str],
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == expected


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


def test_demo_subcommand_still_detected_after_env_option_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture_app() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(sys, "argv", ["fast-agent", "--home", "demo", "demo", "--help"])
    monkeypatch.setattr(cli_main, "app", capture_app)

    cli_main.main()

    assert captured == ["fast-agent", "--home", "demo", "demo", "--help"]


def test_main_converts_keyboard_interrupt_to_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main, "app", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main()

    assert exc_info.value.code == 130
