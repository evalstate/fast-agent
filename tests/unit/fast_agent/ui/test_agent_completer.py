"""Tests for AgentCompleter sub-completion functionality."""

import os
import tempfile
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import fast_agent.config as config_module
from fast_agent.config import Settings, SkillsSettings, get_settings, update_global_settings
from fast_agent.session import get_session_manager, reset_session_manager
from fast_agent.ui.enhanced_prompt import AgentCompleter


def test_complete_history_files_finds_json_and_md():
    """Test that _complete_history_files finds .json and .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "history.json").touch()
        (Path(tmpdir) / "notes.md").touch()
        (Path(tmpdir) / "other.txt").touch()
        (Path(tmpdir) / "data.py").touch()

        completer = AgentCompleter(agents=["agent1"])

        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files(""))
            names = [c.text for c in completions]

            # Should find .json and .md files, not .txt or .py
            assert "history.json" in names
            assert "notes.md" in names
            assert "other.txt" not in names
            assert "data.py" not in names
        finally:
            os.chdir(original_cwd)


def test_complete_history_files_includes_directories():
    """Test that directories are included in completions for navigation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a subdirectory
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.json").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files(""))
            names = [c.text for c in completions]

            # Should include directory with trailing slash
            assert "subdir/" in names
        finally:
            os.chdir(original_cwd)


def test_complete_history_files_filters_by_prefix():
    """Test that completions are filtered by prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "history.json").touch()
        (Path(tmpdir) / "other.md").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files("his"))
            names = [c.text for c in completions]

            assert "history.json" in names
            assert "other.md" not in names
        finally:
            os.chdir(original_cwd)


def test_complete_history_files_handles_subdirectory():
    """Test completion works in subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = Path(tmpdir) / "data"
        subdir.mkdir()
        (subdir / "history.json").touch()
        (subdir / "notes.md").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files("data/"))
            names = [c.text for c in completions]

            assert "data/history.json" in names
            assert "data/notes.md" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_history_load_command():
    """Test get_completions provides file completions after /history load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.json").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Simulate typing "/history load "
            doc = Document("/history load ", cursor_position=14)
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "test.json" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_shell_path_prefix():
    """Ensure shell completions treat path-like tokens as paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "script.sh").touch()
        subdir = Path(tmpdir) / "data"
        subdir.mkdir()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            doc = Document("!./", cursor_position=len("!./"))
            event = CompleteEvent(completion_requested=True)
            completions = list(completer.get_completions(doc, event))
            names = [c.text for c in completions]

            assert "./script.sh" in names
            assert "./data/" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_history_subcommands():
    """Test get_completions suggests /history subcommands."""
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/history ", cursor_position=9)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "show" in names
    assert "save" in names
    assert "load" in names


def test_get_completions_for_session_pin(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(update={"environment_dir": str(env_dir)})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        session = manager.create_session()

        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/session pin ", cursor_position=len("/session pin "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "on" in names
        assert "off" in names
        assert session.info.name in names

        doc = Document("/session pin on ", cursor_position=len("/session pin on "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]
        assert session.info.name in names
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def _write_skill(skill_root: Path, name: str) -> None:
    skill_dir = skill_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: {name}\ndescription: Test skill\n---\n".format(name=name),
        encoding="utf-8",
    )


def test_get_completions_for_skills_subcommands():
    """Test get_completions suggests /skills subcommands."""
    completer = AgentCompleter(agents=["agent1"])

    doc = Document("/skills ", cursor_position=8)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "list" in names
    assert "add" in names
    assert "remove" in names
    assert "registry" in names


def test_get_completions_for_skills_remove(monkeypatch):
    """Test get_completions suggests local skills for /skills remove."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_root = Path(tmpdir) / "skills"
        _write_skill(skills_root, "alpha")
        _write_skill(skills_root, "beta")

        settings = Settings(skills=SkillsSettings(directories=[str(skills_root)]))
        monkeypatch.setattr(config_module, "_settings", settings)

        completer = AgentCompleter(agents=["agent1"])
        doc = Document("/skills remove ", cursor_position=15)
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]

        assert "alpha" in names
        assert "beta" in names


def test_get_completions_for_skills_registry(monkeypatch):
    """Test get_completions suggests registry choices for /skills registry."""
    settings = Settings(
        skills=SkillsSettings(
            marketplace_urls=[
                "https://example.com/registry-one.json",
                "https://example.com/registry-two.json",
            ]
        )
    )
    monkeypatch.setattr(config_module, "_settings", settings)

    completer = AgentCompleter(agents=["agent1"])
    doc = Document("/skills registry ", cursor_position=17)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "1" in names
    assert "2" in names


def test_complete_agent_card_files_finds_md_and_yaml():
    """Test that _complete_agent_card_files finds AgentCard files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "agent.md").touch()
        (Path(tmpdir) / "agent.yaml").touch()
        (Path(tmpdir) / "agent.yml").touch()
        (Path(tmpdir) / "agent.txt").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_agent_card_files(""))
            names = [c.text for c in completions]

            assert "agent.md" in names
            assert "agent.yaml" in names
            assert "agent.yml" in names
            assert "agent.txt" not in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_card_command():
    """Test get_completions provides file completions after /card."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "agent.md").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            doc = Document("/card ", cursor_position=6)
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "agent.md" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_skips_hidden_files():
    """Test that hidden files are not included in completions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / ".hidden.json").touch()
        (Path(tmpdir) / "visible.json").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files(""))
            names = [c.text for c in completions]

            assert "visible.json" in names
            assert ".hidden.json" not in names
        finally:
            os.chdir(original_cwd)


def test_completion_metadata():
    """Test that completions have correct metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.json").touch()
        (Path(tmpdir) / "test.md").touch()
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._complete_history_files(""))

            # Check metadata for each completion type
            for c in completions:
                # display_meta can be string or FormattedText, convert to string
                meta = str(c.display_meta) if c.display_meta else ""
                if c.text == "test.json":
                    assert "JSON history" in meta
                elif c.text == "test.md":
                    assert "Markdown" in meta
                elif c.text == "subdir/":
                    assert "directory" in meta
        finally:
            os.chdir(original_cwd)


def test_command_completions_still_work():
    """Test that regular command completions still work."""
    completer = AgentCompleter(agents=["agent1"])

    # Simulate typing "/hist"
    doc = Document("/hist", cursor_position=5)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    # Should complete to history command
    assert "history" in names


def test_agent_completions_still_work():
    """Test that agent completions still work."""
    completer = AgentCompleter(agents=["test_agent", "other_agent"])

    # Simulate typing "@test"
    doc = Document("@test", cursor_position=5)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "test_agent" in names
    assert "other_agent" not in names
