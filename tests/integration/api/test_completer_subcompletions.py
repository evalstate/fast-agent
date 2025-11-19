"""
Test the AgentCompleter sub-completion functionality for file-based commands.
"""

import os

from prompt_toolkit.document import Document

from fast_agent.ui.enhanced_prompt import AgentCompleter


class TestAgentCompleterFileCompletions:
    """Test file completion for /load_history and /save_history commands."""

    def test_load_history_shows_json_and_md_files(self, tmp_path):
        """Test that /load_history shows .json and .md files for completion."""
        # Create test files in the temp directory
        (tmp_path / "history1.json").touch()
        (tmp_path / "history2.json").touch()
        (tmp_path / "notes.md").touch()
        (tmp_path / "readme.txt").touch()  # Should not appear

        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            completer = AgentCompleter(agents=["agent1"])
            document = Document("/load_history ")

            completions = list(completer.get_completions(document, None))

            # Extract completion text
            completion_texts = [c.text for c in completions]

            # Should include .json and .md files
            assert "history1.json" in completion_texts
            assert "history2.json" in completion_texts
            assert "notes.md" in completion_texts

            # Should not include .txt files
            assert "readme.txt" not in completion_texts
        finally:
            os.chdir(original_cwd)

    def test_load_history_filters_by_partial_filename(self, tmp_path):
        """Test that partial filename filters completions."""
        (tmp_path / "history1.json").touch()
        (tmp_path / "history2.json").touch()
        (tmp_path / "other.json").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            completer = AgentCompleter(agents=["agent1"])
            document = Document("/load_history his")

            completions = list(completer.get_completions(document, None))
            completion_texts = [c.text for c in completions]

            # Should only include files starting with "his"
            assert "history1.json" in completion_texts
            assert "history2.json" in completion_texts
            assert "other.json" not in completion_texts
        finally:
            os.chdir(original_cwd)

    def test_save_history_shows_files(self, tmp_path):
        """Test that /save_history also provides file completions."""
        (tmp_path / "existing.json").touch()
        (tmp_path / "existing.md").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            completer = AgentCompleter(agents=["agent1"])
            document = Document("/save_history ")

            completions = list(completer.get_completions(document, None))
            completion_texts = [c.text for c in completions]

            assert "existing.json" in completion_texts
            assert "existing.md" in completion_texts
        finally:
            os.chdir(original_cwd)

    def test_load_alias_works(self, tmp_path):
        """Test that /load (short alias) also works."""
        (tmp_path / "test.json").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            completer = AgentCompleter(agents=["agent1"])
            document = Document("/load ")

            completions = list(completer.get_completions(document, None))
            completion_texts = [c.text for c in completions]

            assert "test.json" in completion_texts
        finally:
            os.chdir(original_cwd)

    def test_case_insensitive_matching(self, tmp_path):
        """Test that filename matching is case-insensitive."""
        (tmp_path / "History.json").touch()
        (tmp_path / "HISTORY.md").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            completer = AgentCompleter(agents=["agent1"])
            document = Document("/load_history his")

            completions = list(completer.get_completions(document, None))
            completion_texts = [c.text for c in completions]

            # Both should match "his" case-insensitively
            assert "History.json" in completion_texts
            assert "HISTORY.md" in completion_texts
        finally:
            os.chdir(original_cwd)

    def test_completion_metadata(self, tmp_path):
        """Test that completions have appropriate metadata."""
        (tmp_path / "test.json").touch()
        (tmp_path / "test.md").touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            completer = AgentCompleter(agents=["agent1"])
            document = Document("/load_history ")

            completions = list(completer.get_completions(document, None))

            # Check metadata
            for completion in completions:
                meta_str = str(completion.display_meta)
                if completion.text.endswith(".json"):
                    assert "(.json)" in meta_str
                elif completion.text.endswith(".md"):
                    assert "(.md)" in meta_str
        finally:
            os.chdir(original_cwd)

    def test_empty_directory(self, tmp_path):
        """Test behavior with no matching files."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            completer = AgentCompleter(agents=["agent1"])
            document = Document("/load_history ")

            completions = list(completer.get_completions(document, None))

            assert completions == []
        finally:
            os.chdir(original_cwd)

    def test_regular_command_completion_still_works(self):
        """Test that regular command completion is not affected."""
        completer = AgentCompleter(agents=["agent1", "agent2"])
        document = Document("/hel")

        completions = list(completer.get_completions(document, None))
        completion_texts = [c.text for c in completions]

        assert "help" in completion_texts

    def test_agent_completion_still_works(self):
        """Test that agent completion is not affected."""
        completer = AgentCompleter(
            agents=["myagent", "other"],
            agent_types={}
        )
        document = Document("@my")

        completions = list(completer.get_completions(document, None))
        completion_texts = [c.text for c in completions]

        assert "myagent" in completion_texts
        assert "other" not in completion_texts
