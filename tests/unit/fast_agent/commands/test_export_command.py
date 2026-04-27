from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from click.utils import strip_ansi
from mcp.types import TextContent
from typer.testing import CliRunner

import fast_agent.cli.commands.export as export_command
from fast_agent.cli.main import app
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import save_json
from fast_agent.session import (
    SessionAgentSnapshot,
    SessionContinuationSnapshot,
    SessionRequestSettingsSnapshot,
    SessionSnapshot,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_session_fixture(root: Path, *, session_id: str) -> None:
    session_dir = root / "sessions" / session_id
    session_dir.mkdir(parents=True)
    save_json(
        [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="hello")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="done")],
            ),
        ],
        str(session_dir / "history_dev.json"),
    )
    snapshot = SessionSnapshot(
        session_id=session_id,
        created_at=datetime(2026, 4, 20, 13, 3, 0),
        last_activity=datetime(2026, 4, 20, 13, 8, 0),
        continuation=SessionContinuationSnapshot(
            active_agent="dev",
            agents={
                "dev": SessionAgentSnapshot(
                    history_file="history_dev.json",
                    resolved_prompt="You are dev.",
                    model="gpt-5.4",
                    provider="codexresponses",
                    request_settings=SessionRequestSettingsSnapshot(use_history=True),
                )
            },
        ),
    )
    (session_dir / "session.json").write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def test_export_command_exports_latest_session(tmp_path: Path, monkeypatch) -> None:
    env_dir = tmp_path / "env"
    session_id = "2604201303-x5MNlH"
    _write_session_fixture(env_dir, session_id=session_id)
    output_path = tmp_path / "cli-trace.jsonl"
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--env",
            str(env_dir),
            "export",
            "latest",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Exported codex trace" in result.output
    assert output_path.is_file()
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["type"] == "session_meta"
    assert records[0]["payload"]["id"] == session_id


def test_export_command_implicit_target_uses_latest_session(tmp_path: Path, monkeypatch) -> None:
    env_dir = tmp_path / "env"
    session_id = "2604201303-x5MNlH"
    _write_session_fixture(env_dir, session_id=session_id)
    output_path = tmp_path / "implicit-trace.jsonl"
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--env",
            str(env_dir),
            "export",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Exported codex trace" in result.output
    assert output_path.is_file()
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records[0]["type"] == "session_meta"
    assert records[0]["payload"]["id"] == session_id


def test_export_command_lists_sessions(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    _write_session_fixture(env_dir, session_id="2604201303-x5MNlH")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--env",
            str(env_dir),
            "export",
            "--list",
        ],
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "Sessions:" in output
    assert "x5MNlH" in output


def test_export_help_hides_completion_and_format_options() -> None:
    runner = CliRunner()
    result = runner.invoke(export_command.app, ["--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "--list" in output
    assert "--hf-dataset" in output
    assert "--hf-dataset-path" in output
    assert "--privacy-filter" in output
    assert "--privacy-filter-variant" in output
    assert "--show-redactions" in output
    assert "--format" not in output
    assert "--install-completion" not in output
    assert "--show-completion" not in output
