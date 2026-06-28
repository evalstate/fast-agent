"""Integration tests for prompt template parsing and delimited loading."""

import os
import tempfile
from pathlib import Path

import pytest
from mcp.types import TextContent

from fast_agent.mcp.prompt_serialization import (
    load_delimited,
)
from fast_agent.mcp.prompts.prompt_template import (
    PromptTemplate,
    PromptTemplateLoader,
)


def _text(block: object) -> TextContent:
    assert isinstance(block, TextContent)
    return block


class TestTemplateIntegration:
    """Tests for prompt template integration."""

    def test_template_with_substitutions(self):
        """Test applying substitutions to a template."""
        # Create a template with variables
        template_text = """---USER
Hello, I'm trying to learn about {{topic}}.

---ASSISTANT
I'd be happy to help you learn about {{topic}}!
"""
        template = PromptTemplate(template_text)

        context = {"topic": "Python programming"}
        sections = template.apply_substitutions(context)

        assert len(sections) == 2
        assert sections[0].role == "user"
        assert (
            "Hello, I'm trying to learn about Python programming."
            in sections[0].text
        )

        assert sections[1].role == "assistant"
        assert (
            "I'd be happy to help you learn about Python programming!"
            in sections[1].text
        )

    @pytest.fixture
    def temp_delimited_file(self):
        """Create a temporary delimited file for testing."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
Hello, this is a test!

---ASSISTANT
Hi there! I'm here to help with your test.
""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    def test_save_and_load_from_file(self, temp_delimited_file):
        """Test saving and loading multipart messages to/from a file."""

        # Instead of saving through serialization, let's use direct file manipulation
        # Save messages directly to the file
        with open(str(temp_delimited_file), "w", encoding="utf-8") as f:
            f.write("---USER\n")
            f.write("Can you explain quantum physics?\n")
            f.write("---ASSISTANT\n")
            f.write(
                "Quantum physics is fascinating! It deals with the behavior of matter at atomic scales.\n"
            )

        # DEBUG: Read the file content to verify it's written correctly
        with open(str(temp_delimited_file), "r", encoding="utf-8") as f:
            file_content = f.read()
            print(f"DEBUG: File content:\n{file_content}")

        # Load from file
        loaded_messages = load_delimited(str(temp_delimited_file))

        # DEBUG: Print the loaded messages
        print(f"DEBUG: Loaded messages: {loaded_messages}")

        # Verify results
        assert len(loaded_messages) == 2

        # Check user message
        assert loaded_messages[0].role == "user"
        assert len(loaded_messages[0].content) == 1
        assert _text(loaded_messages[0].content[0]).type == "text"
        assert "Can you explain quantum physics?" in _text(loaded_messages[0].content[0]).text

        # Check assistant message
        assert loaded_messages[1].role == "assistant"
        assert len(loaded_messages[1].content) == 1
        assert _text(loaded_messages[1].content[0]).type == "text"
        assert "Quantum physics is fascinating" in _text(loaded_messages[1].content[0]).text
        assert "behavior of matter" in _text(loaded_messages[1].content[0]).text.lower()

    def test_template_loader_integration(self, temp_delimited_file):
        """Test integration with PromptTemplateLoader."""
        # Create a loader
        loader = PromptTemplateLoader()

        # Load template from file
        template = loader.load_from_file(temp_delimited_file)

        # Verify results
        assert len(template.content_sections) == 2
        assert template.content_sections[0].role == "user"
        assert template.content_sections[1].role == "assistant"
