from pathlib import Path

from typer.testing import CliRunner

from fast_agent.cli.commands import session as session_command
from fast_agent.session import SessionManager


def test_prune_empty_removes_only_disposable_sessions(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    manager = SessionManager(home_override=home, respect_env_override=False)
    empty = manager.create_session()
    titled = manager.create_session(metadata={"title": "Keep me"})
    pinned = manager.create_session()
    pinned.set_pinned(True)
    content = manager.create_session()
    (content.directory / "history_agent.json").write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(
        session_command.app,
        ["prune", "--empty"],
        obj={"home": home},
    )

    assert result.exit_code == 0
    assert result.output == "Removed 1 empty session.\n"
    assert not empty.directory.exists()
    assert titled.directory.exists()
    assert pinned.directory.exists()
    assert content.directory.exists()


def test_prune_requires_explicit_mode() -> None:
    result = CliRunner().invoke(session_command.app, ["prune"])

    assert result.exit_code == 2
    assert "Specify what to prune with --empty." in result.output
