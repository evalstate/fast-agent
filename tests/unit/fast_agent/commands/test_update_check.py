import os
from pathlib import Path

from typer.testing import CliRunner

from fast_agent.cli import main as cli_main
from fast_agent.cli.commands import check_config as check_config_command
from fast_agent.cli.update_check import (
    DEFAULT_INTERVAL_SECONDS,
    check_for_update_notice,
    is_newer_version,
    resolve_update_check_marker_path,
    should_check_now,
)
from fast_agent.config import Settings, get_settings, update_global_settings
from fast_agent.core.exceptions import ConfigFileError


def test_resolve_update_check_marker_path_uses_environment_root(tmp_path: Path) -> None:
    environment_dir = tmp_path / "custom-env"
    environment_dir.mkdir()

    marker_path = resolve_update_check_marker_path(environment_dir)

    assert marker_path == environment_dir / ".check_for_update_done"


def test_resolve_update_check_marker_path_uses_environment_dir_env_var(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment_dir = tmp_path / "env-from-var"
    environment_dir.mkdir()
    monkeypatch.setenv("ENVIRONMENT_DIR", str(environment_dir))

    marker_path = resolve_update_check_marker_path(None, cwd=tmp_path)

    assert marker_path == environment_dir / ".check_for_update_done"


def test_resolve_update_check_marker_path_uses_configured_environment_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous_settings = get_settings()
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    update_global_settings(Settings(environment_dir=".dev"))
    (tmp_path / ".dev").mkdir()

    try:
        marker_path = resolve_update_check_marker_path(None, cwd=tmp_path)
    finally:
        update_global_settings(previous_settings)

    assert marker_path == tmp_path / ".dev" / ".check_for_update_done"


def test_resolve_update_check_marker_path_returns_none_without_existing_environment_dir(
    tmp_path: Path,
) -> None:
    marker_path = resolve_update_check_marker_path(tmp_path / "missing-env")

    assert marker_path is None


def test_is_newer_version_compares_release_segments() -> None:
    assert is_newer_version("0.6.19", "0.6.18")
    assert is_newer_version("0.10.0", "0.9.9")
    assert not is_newer_version("0.6.18", "0.6.18")
    assert not is_newer_version("0.6.17", "0.6.18")


def test_check_for_update_notice_uses_marker_to_rate_limit(tmp_path: Path) -> None:
    environment_dir = tmp_path / "env"
    environment_dir.mkdir()
    calls = 0

    def fetch_latest_version() -> str:
        nonlocal calls
        calls += 1
        return "0.6.19"

    notice = check_for_update_notice(
        environment_dir=environment_dir,
        current_version="0.6.18",
        now=200000.0,
        fetch_latest_version=fetch_latest_version,
    )

    assert notice is not None
    assert "0.6.19" in notice
    assert calls == 1

    second_notice = check_for_update_notice(
        environment_dir=environment_dir,
        current_version="0.6.18",
        now=200100.0,
        fetch_latest_version=fetch_latest_version,
    )

    assert second_notice is None
    assert calls == 1


def test_should_check_now_respects_marker_age(tmp_path: Path) -> None:
    marker_path = tmp_path / ".check_for_update_done"
    marker_path.touch()
    recent_time = 5000.0
    os.utime(marker_path, (recent_time, recent_time))

    assert not should_check_now(
        marker_path,
        now=recent_time + DEFAULT_INTERVAL_SECONDS - 1,
    )
    assert should_check_now(
        marker_path,
        now=recent_time + DEFAULT_INTERVAL_SECONDS + 1,
    )


def test_check_for_update_notice_skips_dev_versions(tmp_path: Path) -> None:
    environment_dir = tmp_path / "env"

    notice = check_for_update_notice(
        environment_dir=environment_dir,
        current_version="0.6.19.dev0",
        fetch_latest_version=lambda: "0.6.19",
    )

    assert notice is None
    marker_path = resolve_update_check_marker_path(environment_dir)
    assert marker_path is None or not marker_path.exists()


def test_check_for_update_notice_does_not_rate_limit_failed_fetch(tmp_path: Path) -> None:
    environment_dir = tmp_path / "env"
    environment_dir.mkdir()
    calls = 0

    def failing_fetch_latest_version() -> str:
        nonlocal calls
        calls += 1
        raise TimeoutError("temporary failure")

    first_notice = check_for_update_notice(
        environment_dir=environment_dir,
        current_version="0.6.18",
        now=200000.0,
        fetch_latest_version=failing_fetch_latest_version,
    )

    assert first_notice is None
    assert calls == 1
    marker_path = resolve_update_check_marker_path(environment_dir)
    assert marker_path is not None
    assert not marker_path.exists()

    def fetch_latest_version() -> str:
        nonlocal calls
        calls += 1
        return "0.6.19"

    second_notice = check_for_update_notice(
        environment_dir=environment_dir,
        current_version="0.6.18",
        now=200100.0,
        fetch_latest_version=fetch_latest_version,
    )

    assert second_notice is not None
    assert "0.6.19" in second_notice
    assert calls == 2


def test_check_for_update_notice_does_not_create_marker_without_existing_environment_dir(
    tmp_path: Path,
) -> None:
    environment_dir = tmp_path / "missing-env"
    calls = 0

    def fetch_latest_version() -> str:
        nonlocal calls
        calls += 1
        return "0.6.19"

    first_notice = check_for_update_notice(
        environment_dir=environment_dir,
        current_version="0.6.18",
        now=200000.0,
        fetch_latest_version=fetch_latest_version,
    )
    second_notice = check_for_update_notice(
        environment_dir=environment_dir,
        current_version="0.6.18",
        now=200100.0,
        fetch_latest_version=fetch_latest_version,
    )

    assert first_notice is not None
    assert second_notice is not None
    assert calls == 2
    assert resolve_update_check_marker_path(environment_dir) is None


def test_check_for_update_notice_skips_config_resolution_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.cli.update_check.resolve_update_check_marker_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ConfigFileError("Failed to parse YAML file: fastagent.config.yaml", "bad yaml")
        ),
    )

    notice = check_for_update_notice(
        environment_dir=None,
        current_version="0.6.18",
        fetch_latest_version=lambda: "0.6.19",
    )

    assert notice is None


def test_version_flag_skips_update_checks(monkeypatch) -> None:
    runner = CliRunner()

    def _unexpected_check(**_kwargs: object) -> str | None:
        raise AssertionError("version should not check for updates")

    monkeypatch.setattr(cli_main, "check_for_update_notice", _unexpected_check)

    result = runner.invoke(cli_main.app, ["--version"])

    assert result.exit_code == 0
    assert "fast-agent-mcp v" in result.output


def test_check_subcommands_skip_update_checks_for_json_output(monkeypatch) -> None:
    runner = CliRunner()

    def _unexpected_check(**_kwargs: object) -> str | None:
        raise AssertionError("nested check subcommands should not check for updates")

    monkeypatch.setattr(check_config_command, "check_for_update_notice", _unexpected_check)
    monkeypatch.setattr(
        check_config_command,
        "show_model_secret_requirements",
        lambda *_args, **_kwargs: print('{"ok": true}'),
    )

    result = runner.invoke(
        cli_main.app,
        ["check", "models", "--for-model", "demo:model", "--json"],
    )

    assert result.exit_code == 0
    assert result.output.strip() == '{"ok": true}'


def test_bare_check_summary_prints_update_notice(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        check_config_command,
        "check_for_update_notice",
        lambda **_kwargs: "update available",
    )
    monkeypatch.setattr(
        check_config_command,
        "show_check_summary",
        lambda *_args, **_kwargs: print("summary"),
    )

    result = runner.invoke(cli_main.app, ["check"])

    assert result.exit_code == 0
    assert "update available" in result.output
    assert "summary" in result.output
