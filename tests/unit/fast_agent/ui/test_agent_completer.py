"""Tests for AgentCompleter file completion functionality."""

import os
import tempfile
from pathlib import Path

from prompt_toolkit.document import Document

from fast_agent.ui.enhanced_prompt import AgentCompleter


def test_file_completions_finds_json_files() -> None:
    """Test that _get_file_completions finds .json files in directory."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, "history1.json").touch()
        Path(tmpdir, "history2.json").touch()
        Path(tmpdir, "notes.txt").touch()  # Should not be included

        # Change to temp directory for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._get_file_completions("", [".json", ".md"]))
            names = [c.text for c in completions]

            assert "history1.json" in names
            assert "history2.json" in names
            assert "notes.txt" not in names
        finally:
            os.chdir(original_cwd)


def test_file_completions_finds_md_files() -> None:
    """Test that _get_file_completions finds .md files in directory."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, "conversation.md").touch()
        Path(tmpdir, "data.json").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._get_file_completions("", [".json", ".md"]))
            names = [c.text for c in completions]

            assert "conversation.md" in names
            assert "data.json" in names
        finally:
            os.chdir(original_cwd)


def test_file_completions_filters_by_prefix() -> None:
    """Test that completions are filtered by the typed prefix."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, "history1.json").touch()
        Path(tmpdir, "history2.json").touch()
        Path(tmpdir, "backup.json").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._get_file_completions("his", [".json", ".md"]))
            names = [c.text for c in completions]

            assert "history1.json" in names
            assert "history2.json" in names
            assert "backup.json" not in names
        finally:
            os.chdir(original_cwd)


def test_file_completions_includes_directories() -> None:
    """Test that directories are included with trailing slash."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test directory and file
        subdir = Path(tmpdir, "histories")
        subdir.mkdir()
        Path(tmpdir, "history.json").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._get_file_completions("", [".json", ".md"]))
            names = [c.text for c in completions]

            assert "histories/" in names
            assert "history.json" in names
        finally:
            os.chdir(original_cwd)


def test_file_completions_skips_hidden_files() -> None:
    """Test that hidden files are skipped unless prefix starts with dot."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, ".hidden.json").touch()
        Path(tmpdir, "visible.json").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Without dot prefix - hidden file excluded
            completions = list(completer._get_file_completions("", [".json", ".md"]))
            names = [c.text for c in completions]
            assert ".hidden.json" not in names
            assert "visible.json" in names

            # With dot prefix - hidden file included
            completions = list(completer._get_file_completions(".", [".json", ".md"]))
            names = [c.text for c in completions]
            assert ".hidden.json" in names
        finally:
            os.chdir(original_cwd)


def test_file_completions_handles_subdirectory_path() -> None:
    """Test completion works when typing a subdirectory path."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create subdirectory with files
        subdir = Path(tmpdir, "subdir")
        subdir.mkdir()
        Path(subdir, "nested.json").touch()
        Path(subdir, "nested.md").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._get_file_completions("subdir/", [".json", ".md"]))
            names = [c.text for c in completions]

            assert "nested.json" in names
            assert "nested.md" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_triggers_file_completion_for_load_history() -> None:
    """Test that get_completions triggers file completion for /load_history command."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        Path(tmpdir, "test.json").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Simulate typing "/load_history "
            doc = Document("/load_history ")
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "test.json" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_triggers_file_completion_for_load_alias() -> None:
    """Test that get_completions triggers file completion for /load command alias."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        Path(tmpdir, "test.md").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Simulate typing "/load "
            doc = Document("/load ")
            completions = list(completer.get_completions(doc, None))
            names = [c.text for c in completions]

            assert "test.md" in names
        finally:
            os.chdir(original_cwd)


def test_get_completions_shows_command_completions() -> None:
    """Test that standard command completions still work."""
    completer = AgentCompleter(agents=["agent1"])

    # Test command completion
    doc = Document("/load")
    completions = list(completer.get_completions(doc, None))
    commands = [c.text for c in completions]

    assert "load_history" in commands


def test_get_completions_shows_agent_completions() -> None:
    """Test that agent completions still work."""
    completer = AgentCompleter(agents=["myagent", "other"])

    # Test agent completion
    doc = Document("@my")
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]

    assert "myagent" in names
    assert "other" not in names


def test_file_completions_handles_nonexistent_directory() -> None:
    """Test that completions handle non-existent directories gracefully."""
    completer = AgentCompleter(agents=[])

    completions = list(completer._get_file_completions("nonexistent/", [".json", ".md"]))

    assert completions == []


def test_file_completions_display_meta_shows_extension() -> None:
    """Test that completion display_meta shows file extension type."""
    completer = AgentCompleter(agents=[])

    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "data.json").touch()
        Path(tmpdir, "notes.md").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            completions = list(completer._get_file_completions("", [".json", ".md"]))
            # display_meta can be FormattedText or string, convert to string for comparison
            meta_map = {c.text: str(c.display_meta) if hasattr(c.display_meta, '__iter__') and not isinstance(c.display_meta, str) else c.display_meta for c in completions}

            # Check that the meta contains the extension type
            assert "JSON" in str(meta_map["data.json"])
            assert "MD" in str(meta_map["notes.md"])
        finally:
            os.chdir(original_cwd)
