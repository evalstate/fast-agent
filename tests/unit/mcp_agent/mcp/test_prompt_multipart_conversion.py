"""
Unit test demonstrating issues with PromptMessageExtended conversion.
"""

import os
import tempfile
from pathlib import Path

from mcp.types import PromptMessage, TextContent

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from mcp_agent.mcp.prompts.prompt_load import create_messages_with_resources, load_prompt
from mcp_agent.mcp.prompts.prompt_template import PromptTemplateLoader


def test_resource_message_role_merging():
    """
    Test that demonstrates how resources cause role merging issues.

    When a section has resources, create_messages_with_resources creates separate
    messages for each resource with the same role. This breaks the alternating
    pattern expected by the playback code when converting to multipart.
    """
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

        # Load the template
        loader = PromptTemplateLoader()
        template = loader.load_from_file(tf_path)

        # Step 1: Create messages with resources
        messages = create_messages_with_resources(template.content_sections, [tf_path])

        # We should get 4 messages with roles: user, user, assistant, assistant
        assert len(messages) == 4
        assert messages[0].role == "user"  # User text message
        assert messages[1].role == "user"  # User resource message
        assert messages[2].role == "assistant"  # Assistant text message
        assert messages[3].role == "assistant"  # Assistant resource message

        # Step 2: Convert to multipart (this is what happens in load_prompt_multipart)
        multipart_messages = PromptMessageExtended.to_extended(messages)

        # Here's the issue: we get only 2 messages instead of 4 because consecutive
        # messages with the same role are merged
        assert len(multipart_messages) == 2  # Should be 2 multipart messages
        assert multipart_messages[0].role == "user"
        assert multipart_messages[1].role == "assistant"

        # Each multipart message contains multiple content items
        assert len(multipart_messages[0].content) == 2  # Text + resource
        assert len(multipart_messages[1].content) == 2  # Text + resource

        # When used with playback, this causes issues because the playback code
        # expects 4 separate messages with alternating roles
    finally:
        # Clean up
        os.unlink(tf_path)
        if resource1_path.exists():
            os.unlink(resource1_path)
        if resource2_path.exists():
            os.unlink(resource2_path)


def test_alternating_roles_with_no_resources():
    """
    Test that demonstrates the correct behavior with no resources.
    When there are no resources, alternating roles are preserved.
    """
    # Create a temporary conversation file with alternating roles and no resources
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
        # Direct load_prompt approach (how playback.md is loaded in the failing test)
        messages = load_prompt(tf_path)

        # We should get 4 messages with alternating roles
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
        assert messages[3].role == "assistant"

        # Convert to multipart
        multipart = PromptMessageExtended.to_extended(messages)

        # We should still have 4 messages with alternating roles
        assert len(multipart) == 4
        assert multipart[0].role == "user"
        assert multipart[1].role == "assistant"
        assert multipart[2].role == "user"
        assert multipart[3].role == "assistant"

        # Each should have 1 content item
        assert len(multipart[0].content) == 1
        assert len(multipart[1].content) == 1
        assert len(multipart[2].content) == 1
        assert len(multipart[3].content) == 1
    finally:
        # Clean up
        os.unlink(tf_path)


def test_playback_pattern_with_simple_messages():
    """
    Test that demonstrates the expected pattern for playback.
    """
    # Create simple message sequence that matches playback.md
    messages = [
        PromptMessage(role="user", content=TextContent(type="text", text="user1")),
        PromptMessage(role="assistant", content=TextContent(type="text", text="assistant1")),
        PromptMessage(role="user", content=TextContent(type="text", text="user2")),
        PromptMessage(role="assistant", content=TextContent(type="text", text="assistant2")),
    ]

    # Convert to multipart - this should maintain 4 separate messages
    multipart = PromptMessageExtended.to_extended(messages)
    assert len(multipart) == 4

    # Check roles are preserved
    assert multipart[0].role == "user"
    assert multipart[1].role == "assistant"
    assert multipart[2].role == "user"
    assert multipart[3].role == "assistant"

    # Check content is preserved
    assert multipart[0].content[0].text == "user1"
    assert multipart[1].content[0].text == "assistant1"
    assert multipart[2].content[0].text == "user2"
    assert multipart[3].content[0].text == "assistant2"
