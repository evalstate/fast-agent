from __future__ import annotations

import os
from pathlib import Path

from click.utils import strip_ansi
from typer.testing import CliRunner

import fast_agent.config as config_module
from fast_agent.cli.commands import auth as auth_command
from fast_agent.config import get_settings, update_global_settings
from fast_agent.core.keyring_utils import KeyringStatus


def test_auth_status_reports_invalid_settings_yaml_without_traceback(tmp_path: Path) -> None:
    env_root = tmp_path / ".fast-agent"
    env_root.mkdir(parents=True)
    config_path = env_root / "fastagent.config.yaml"
    config_path.write_text(
        "mcp:\n"
        "  targets:\n"
        "    - name: openai\n"
        "        target: https://developers.openai.com/mcp\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    old_cwd = Path.cwd()
    old_env_dir = os.environ.get("ENVIRONMENT_DIR")
    try:
        os.chdir(tmp_path)
        os.environ.pop("ENVIRONMENT_DIR", None)
        config_module._settings = None

        runner = CliRunner()
        result = runner.invoke(auth_command.app, ["status"])
        output = strip_ansi(result.output)

        assert result.exit_code == 1, output
        assert "Error loading fast-agent settings:" in output
        assert f"Failed to parse YAML file: {config_path}" in output
        assert "mapping values are not allowed here" in output
        assert "Traceback" not in output
    finally:
        os.chdir(old_cwd)
        if old_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = old_env_dir
        update_global_settings(old_settings)


def test_auth_status_shows_codex_source(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.cli.commands.auth.get_settings_or_exit",
        lambda _config_path=None: get_settings(),
    )
    runner = CliRunner()
    monkeypatch.setattr(
        "fast_agent.cli.commands.auth.get_keyring_status",
        lambda: KeyringStatus(name="SecretService Keyring", available=True, writable=True),
    )
    monkeypatch.setattr(
        "fast_agent.cli.commands.auth.list_keyring_tokens",
        lambda: [],
    )
    monkeypatch.setattr(
        "fast_agent.llm.provider.openai.codex_oauth.get_codex_token_status",
        lambda: {
            "present": True,
            "expires_at": None,
            "expired": False,
            "source": "auth.json",
        },
    )

    result = runner.invoke(auth_command.app, ["status"])

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "Codex OAuth" in output
    assert "Source" in output
    assert "Codex auth.json" in output
