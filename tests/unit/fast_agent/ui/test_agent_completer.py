"""Tests for AgentCompleter sub-completion functionality."""

import os
import tempfile
from pathlib import Path

from prompt_toolkit.document import Document

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


def test_get_completions_for_load_history_command():
    """Test get_completions provides file completions after /load_history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.json").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Simulate typing "/load_history "
            doc = Document("/load_history ", cursor_position=14)
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "test.json" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_for_load_shortcut():
    """Test get_completions works with /load shortcut."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.md").touch()

        completer = AgentCompleter(agents=["agent1"])

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Simulate typing "/load "
            doc = Document("/load ", cursor_position=6)
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "test.md" in names
        finally:
            os.chdir(original_cwd)


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

    # Simulate typing "/load"
    doc = Document("/load", cursor_position=5)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    # Should complete to load_history command
    assert "load_history" in names


def test_agent_completions_still_work():
    """Test that agent completions still work."""
    completer = AgentCompleter(agents=["test_agent", "other_agent"])

    # Simulate typing "@test"
    doc = Document("@test", cursor_position=5)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "test_agent" in names
    assert "other_agent" not in names
