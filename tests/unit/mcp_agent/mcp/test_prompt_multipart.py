"""
Tests for using PromptMessageExtended in augmented LLMs.
"""

import os
import tempfile
from pathlib import Path

from mcp_agent.mcp.prompts.prompt_load import create_messages_with_resources, load_prompt
from mcp_agent.mcp.prompts.prompt_template import PromptTemplateLoader


def test_create_messages_with_resources_alternating_roles():
    """Test create_messages_with_resources maintains correct role alternation."""
    # Create a temporary conversation file with alternating roles
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as tf:
        tf.write("""---USER
message 1
---ASSISTANT
message 2
---USER
message 3
---ASSISTANT
message 4
""")
        tf_path = Path(tf.name)

    try:
        # Use the PromptTemplateLoader to parse the file
        loader = PromptTemplateLoader()
        template = loader.load_from_file(tf_path)

        # Create messages with resources
        messages = create_messages_with_resources(template.content_sections, [tf_path])

        # Verify we get 4 messages with alternating roles
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
        assert messages[3].role == "assistant"

        # Verify contents
        assert "message 1" in messages[0].content.text  # type: ignore
        assert "message 2" in messages[1].content.text  # type: ignore
        assert "message 3" in messages[2].content.text  # type: ignore
        assert "message 4" in messages[3].content.text  # type: ignore
    finally:
        # Clean up
        os.unlink(tf_path)


def test_create_messages_with_resources_roles_with_resources():
    """Test create_messages_with_resources maintains roles even with resources."""
    # Create a temporary conversation file with resources
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as tf:
        tf.write("""---USER
user message
---RESOURCE
resource1.txt
---ASSISTANT
assistant message
---RESOURCE
resource2.txt
""")
        tf_path = Path(tf.name)

    # Create resource files
    resource1_path = tf_path.parent / "resource1.txt"
    resource2_path = tf_path.parent / "resource2.txt"

    try:
        # Create the resource files
        with open(resource1_path, "w") as f:
            f.write("user resource content")
        with open(resource2_path, "w") as f:
            f.write("assistant resource content")

        # Use the PromptTemplateLoader to parse the file
        loader = PromptTemplateLoader()
        template = loader.load_from_file(tf_path)

        # Create messages with resources
        messages = create_messages_with_resources(template.content_sections, [tf_path])

        # We should get 4 messages:
        # 1. User text
        # 2. User resource
        # 3. Assistant text
        # 4. Assistant resource
        assert len(messages) == 4

        # Check roles - this is where the bug manifests
        # Currently all messages from the user section (text + resources) will have role="user"
        # and all messages from the assistant section will have role="assistant"
        assert messages[0].role == "user"  # User text message
        assert (
            messages[1].role == "user"
        )  # User resource message (should this be user or assistant?)
        assert messages[2].role == "assistant"  # Assistant text message
        assert messages[3].role == "assistant"  # Assistant resource message

        # The current implementation groups messages by section, which breaks the alternating pattern
        # expected by the playback code.
    finally:
        # Clean up
        os.unlink(tf_path)
        if resource1_path.exists():
            os.unlink(resource1_path)
        if resource2_path.exists():
            os.unlink(resource2_path)


def test_load_prompt_from_file():
    """Test the load_prompt function preserves roles correctly."""
    # Create a temporary conversation file with alternating roles
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as tf:
        tf.write("""---USER
user1
---ASSISTANT
assistant1
---USER
user2
---ASSISTANT
assistant2
""")
        tf_path = Path(tf.name)

    try:
        # Load the prompt directly
        messages = load_prompt(tf_path)

        # Verify we get 4 messages with alternating roles - this will fail with the current implementation
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
        assert messages[3].role == "assistant"

        # Verify contents
        assert "user1" in messages[0].content.text  # type: ignore
        assert "assistant1" in messages[1].content.text  # type: ignore
        assert "user2" in messages[2].content.text  # type: ignore
        assert "assistant2" in messages[3].content.text  # type: ignore
    finally:
        # Clean up
        os.unlink(tf_path)
