from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import typer
from click.utils import strip_ansi
from typer.testing import CliRunner

import fast_agent.config as config_module
from fast_agent.auth.credentials import OAuthCredential, save_oauth_credential
from fast_agent.cli.commands import auth as auth_command
from fast_agent.config import get_settings, update_global_settings
from fast_agent.core.keyring_utils import KeyringStatus


def test_auth_status_reports_invalid_settings_yaml_without_traceback(tmp_path: Path) -> None:
    home_root = tmp_path / ".fast-agent"
    home_root.mkdir(parents=True)
    config_path = home_root / "fastagent.config.yaml"
    config_path.write_text(
        "mcp:\n  targets:\n    - name: openai\n        target: https://developers.openai.com/mcp\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    old_cwd = Path.cwd()
    old_home = os.environ.get("FAST_AGENT_HOME")
    try:
        os.chdir(tmp_path)
        os.environ.pop("FAST_AGENT_HOME", None)
        config_module._settings = None

        runner = CliRunner()
        result = runner.invoke(auth_command.app, ["mcp", "status"])
        output = strip_ansi(result.output)

        assert result.exit_code == 1, output
        assert "Error loading fast-agent settings:" in output
        assert f"Failed to parse YAML file: {config_path}" in output
        assert "mapping values are not allowed here" in output
        assert "Traceback" not in output
    finally:
        os.chdir(old_cwd)
        if old_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = old_home
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
    assert "Codex" in output
    assert "Source" in output
    assert "auth.json" in output


def test_default_auth_view_includes_provider_and_mcp_status(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        auth_command,
        "provider_status",
        lambda provider=None: calls.append("provider"),
    )
    monkeypatch.setattr(
        auth_command,
        "mcp_status",
        lambda target=None, config_path=None: calls.append("mcp"),
    )

    result = CliRunner().invoke(auth_command.app)

    assert result.exit_code == 0, result.output
    assert calls == ["provider", "mcp"]


def test_auth_status_rejects_unknown_provider_without_traceback() -> None:
    result = CliRunner().invoke(auth_command.app, ["status", "unknown"])

    assert result.exit_code == 1
    output = strip_ansi(result.output)
    assert "Unsupported OAuth provider" in output
    assert "Traceback" not in output


def test_xai_token_export_and_logout_workflow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    export_path = tmp_path / "xai.auth.json"
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(auth_path))
    save_oauth_credential(
        "xai",
        OAuthCredential(
            access_token="xai-access",
            refresh_token="xai-refresh",
        ),
    )
    runner = CliRunner()

    token_result = runner.invoke(auth_command.app, ["token", "xai"])
    export_result = runner.invoke(
        auth_command.app,
        ["export", "xai", str(export_path)],
    )
    logout_result = runner.invoke(auth_command.app, ["logout", "xai", "--yes"])
    status_result = runner.invoke(auth_command.app, ["status", "xai"])

    assert token_result.exit_code == 0
    assert token_result.output.strip() == "xai-access"
    assert export_result.exit_code == 0
    exported = json.loads(export_path.read_text())
    assert set(exported["providers"]) == {"xai"}
    assert exported["providers"]["xai"]["refresh_token"] == "xai-refresh"
    assert logout_result.exit_code == 0
    assert "removed" in logout_result.output
    assert "not configured" in strip_ansi(status_result.output)


def test_codex_export_refuses_to_overwrite_cli_auth_file(monkeypatch, tmp_path: Path) -> None:
    cli_auth_path = tmp_path / "codex-profile" / "auth.json"
    cli_auth_path.parent.mkdir()
    original = json.dumps({"tokens": {"access_token": "cli-token"}})
    cli_auth_path.write_text(original)
    monkeypatch.delenv("FAST_AGENT_AUTH_FILE", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(cli_auth_path.parent))

    result = CliRunner().invoke(
        auth_command.app,
        ["export", "codex", str(cli_auth_path), "--force"],
    )

    assert result.exit_code == 1
    assert "Codex CLI auth file is read-only" in strip_ansi(result.output)
    assert cli_auth_path.read_text() == original


def test_validated_identity_transport_normalizes_values() -> None:
    assert auth_command._validated_identity_transport(None) == "http"
    assert auth_command._validated_identity_transport(" SSE ") == "sse"


def test_validated_identity_transport_rejects_non_remote_transport() -> None:
    with pytest.raises(typer.Exit):
        auth_command._validated_identity_transport("stdio")
