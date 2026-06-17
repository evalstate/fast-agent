from __future__ import annotations

import subprocess
from pathlib import Path

from fast_agent.ui.prompt import editor


def test_get_text_from_editor_runs_configured_editor_and_cleans_temp_file(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "fake-editor --wait")
    temp_paths: list[Path] = []

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
        assert command[:2] == ["fake-editor", "--wait"]
        temp_path = Path(command[-1])
        temp_paths.append(temp_path)
        assert temp_path.read_text(encoding="utf-8") == "original"
        temp_path.write_text("edited\n", encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = editor.get_text_from_editor("original")

    assert result == "edited"
    assert temp_paths
    assert not temp_paths[0].exists()


def test_get_text_from_editor_returns_initial_text_when_editor_fails(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "missing-editor")

    def fake_run(command: list[str], check: bool) -> None:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert editor.get_text_from_editor("original") == "original"


def test_get_text_from_editor_prints_bracketed_editor_command_literally(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "[missing-editor]")

    def fake_run(command: list[str], check: bool) -> None:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert editor.get_text_from_editor("original") == "original"
    assert "Editor command '[missing-editor]' not found" in capsys.readouterr().out
