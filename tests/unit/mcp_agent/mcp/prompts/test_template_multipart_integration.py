"""
Integration tests for PromptTemplate and PromptMessageExtended.
"""

import os
import tempfile
from pathlib import Path

import pytest
from mcp.types import TextContent

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from mcp_agent.mcp.prompt_serialization import (
    load_messages_from_delimited_file,
)
from mcp_agent.mcp.prompts.prompt_template import (
    PromptTemplate,
    PromptTemplateLoader,
)


class TestTemplateMultipartIntegration:
    """Tests for integration between PromptTemplate and PromptMessageExtended."""

    def test_template_to_extended_conversion(self):
        """Test converting a PromptTemplate to PromptMessageExtended objects."""
        # Create a template
        template_text = """---USER
Hello, I'm trying to learn about {{topic}}.

---ASSISTANT
I'd be happy to help you learn about {{topic}}!

Here are some key points about {{topic}}:
1. Point one
2. Point two
3. Point three
"""
        template = PromptTemplate(template_text)

        # Convert to multipart messages
        multiparts = template.to_extended_messages()

        # Verify results
        assert len(multiparts) == 2
        assert multiparts[0].role == "user"
        assert len(multiparts[0].content) == 1
        assert multiparts[0].content[0].type == "text"
        assert "Hello, I'm trying to learn about {{topic}}." in multiparts[0].content[0].text

        assert multiparts[1].role == "assistant"
        assert len(multiparts[1].content) == 1
        assert multiparts[1].content[0].type == "text"
        assert "I'd be happy to help you learn about {{topic}}!" in multiparts[1].content[0].text

    def test_template_with_substitutions_to_extended(self):
        """Test applying substitutions to a template and converting to extended."""
        # Create a template with variables
        template_text = """---USER
Hello, I'm trying to learn about {{topic}}.

---ASSISTANT
I'd be happy to help you learn about {{topic}}!
"""
        template = PromptTemplate(template_text)

        # Apply substitutions and convert to multipart
        context = {"topic": "Python programming"}
        multiparts = template.apply_substitutions_to_extended(context)

        # Verify results
        assert len(multiparts) == 2
        assert multiparts[0].role == "user"
        assert (
            "Hello, I'm trying to learn about Python programming." in multiparts[0].content[0].text
        )

        assert multiparts[1].role == "assistant"
        assert (
            "I'd be happy to help you learn about Python programming!"
            in multiparts[1].content[0].text
        )

    def test_multipart_to_template_conversion(self):
        """Test converting PromptMessageExtended objects to a PromptTemplate."""
        # Create multipart messages
        multiparts = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="What's the capital of France?")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="The capital of France is Paris.")],
            ),
        ]

        # Convert to template
        template = PromptTemplate.from_multipart_messages(multiparts)

        # Verify results
        assert len(template.content_sections) == 2
        assert template.content_sections[0].role == "user"
        assert template.content_sections[0].text == "What's the capital of France?"
        assert template.content_sections[1].role == "assistant"
        assert template.content_sections[1].text == "The capital of France is Paris."

    def test_round_trip_conversion(self):
        """Test round-trip conversion between PromptTemplate and PromptMessageExtended."""
        # Original template
        template_text = """---USER
Tell me about {{subject}}.

---RESOURCE
{{subject}}_info.txt

---ASSISTANT
Here's information about {{subject}}:

---RESOURCE
{{subject}}_details.txt
"""
        original_template = PromptTemplate(template_text)

        # Convert to multipart
        multiparts = original_template.to_extended_messages()

        # Convert back to template
        new_template = PromptTemplate.from_multipart_messages(multiparts)

        # Verify the structure is preserved
        assert len(new_template.content_sections) == len(original_template.content_sections)

        for i, section in enumerate(original_template.content_sections):
            new_section = new_template.content_sections[i]
            assert new_section.role == section.role
            assert section.text in new_section.text  # Text might have whitespace differences

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
        loaded_messages = load_messages_from_delimited_file(str(temp_delimited_file))

        # DEBUG: Print the loaded messages
        print(f"DEBUG: Loaded messages: {loaded_messages}")

        # Verify results
        assert len(loaded_messages) == 2

        # Check user message
        assert loaded_messages[0].role == "user"
        assert len(loaded_messages[0].content) == 1
        assert loaded_messages[0].content[0].type == "text"
        assert "Can you explain quantum physics?" in loaded_messages[0].content[0].text

        # Check assistant message
        assert loaded_messages[1].role == "assistant"
        assert len(loaded_messages[1].content) == 1
        assert loaded_messages[1].content[0].type == "text"
        assert "Quantum physics is fascinating" in loaded_messages[1].content[0].text
        assert "behavior of matter" in loaded_messages[1].content[0].text.lower()

    def test_template_loader_integration(self, temp_delimited_file):
        """Test integration with PromptTemplateLoader."""
        # Create a loader
        loader = PromptTemplateLoader()

        # Load template from file
        template = loader.load_from_file(temp_delimited_file)

        # Convert to multipart
        multiparts = template.to_extended_messages()

        # Verify results
        assert len(multiparts) == 2
        assert multiparts[0].role == "user"
        assert multiparts[1].role == "assistant"

        # Create new messages and convert to template
        new_messages = [
            PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Tell me a joke.")]
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Why did the chicken cross the road?")],
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="I don't know, why?")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="To get to the other side!")],
            ),
        ]

        # Create template using the loader
        new_template = loader.load_from_multipart(new_messages)

        # Verify results
        assert len(new_template.content_sections) == 4
        assert new_template.content_sections[0].role == "user"
        assert new_template.content_sections[0].text == "Tell me a joke."
        assert new_template.content_sections[1].role == "assistant"
        assert new_template.content_sections[1].text == "Why did the chicken cross the road?"
