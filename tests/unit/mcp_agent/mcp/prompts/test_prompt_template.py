"""
Unit tests for the prompt template module.
"""

import asyncio
import base64
import os
import tempfile
from pathlib import Path

import pytest
from mcp.types import ImageContent, TextContent

from mcp_agent.mcp import mime_utils, resource_utils
from mcp_agent.mcp.prompts.prompt_load import create_messages_with_resources
from mcp_agent.mcp.prompts.prompt_template import (
    PromptContent,
    PromptMetadata,
    PromptTemplate,
    PromptTemplateLoader,
)

TINY_IMAGE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class TestPromptContent:
    """Tests for the PromptContent class"""

    def test_apply_substitutions_content(self):
        """Test substituting variables in content"""
        content = PromptContent(text="Hello {{name}}! Your age is {{age}}.", role="user")
        context = {"name": "Alice", "age": 30}

        result = content.apply_substitutions(context)

        assert result.text == "Hello Alice! Your age is 30."
        assert result.role == "user"
        assert result.resources == []

    def test_apply_substitutions_missing_var(self):
        """Test substituting with missing variables"""
        content = PromptContent(text="Hello {{name}}! Your age is {{age}}.", role="user")
        context = {"name": "Bob"}

        result = content.apply_substitutions(context)

        assert result.text == "Hello Bob! Your age is {{age}}."
        assert result.role == "user"

    def test_apply_substitutions_with_resources(self):
        """Test substituting variables in content with resources"""
        content = PromptContent(
            text="Hello {{name}}! Your age is {{age}}.",
            role="user",
            resources=["data_{{name}}.txt", "profile_{{age}}.json"],
        )
        context = {"name": "Alice", "age": 30}

        result = content.apply_substitutions(context)

        assert result.text == "Hello Alice! Your age is 30."
        assert result.role == "user"
        assert result.resources == ["data_Alice.txt", "profile_30.json"]


class TestPromptTemplate:
    """Tests for the PromptTemplate class"""

    def test_simple_mode(self):
        """Test parsing a simple template with no delimiters"""
        template_text = "Hello {{name}}!\nHow are you?"
        template = PromptTemplate(template_text)

        assert len(template.content_sections) == 1
        assert template.content_sections[0].role == "user"
        assert template.content_sections[0].text == template_text
        assert template.template_variables == {"name"}
        assert template.content_sections[0].resources == []

    def test_delimited_mode(self):
        """Test parsing a template with delimiters"""
        template_text = """---USER
Hello {{name}}!

---ASSISTANT
Hi {{name}}! How can I help you today?

---USER
Tell me about {{topic}}.
"""
        template = PromptTemplate(template_text)

        assert len(template.content_sections) == 3
        assert template.content_sections[0].role == "user"
        assert template.content_sections[0].text == "Hello {{name}}!"
        assert template.content_sections[1].role == "assistant"
        assert template.content_sections[1].text == "Hi {{name}}! How can I help you today?"
        assert template.content_sections[2].role == "user"
        assert template.content_sections[2].text == "Tell me about {{topic}}."
        assert template.template_variables == {"name", "topic"}

    def test_custom_delimiters(self):
        """Test parsing a template with custom delimiters"""
        template_text = """#USER
Hello {{name}}!

#ASSISTANT
Hi there!
"""
        delimiter_map = {"#USER": "user", "#ASSISTANT": "assistant"}
        template = PromptTemplate(template_text, delimiter_map)

        assert len(template.content_sections) == 2
        assert template.content_sections[0].role == "user"
        assert template.content_sections[1].role == "assistant"
        assert template.template_variables == {"name"}

    def test_resources_in_template(self):
        """Test parsing a template with resources"""
        template_text = """---USER
Hello! Check out this resource:

---RESOURCE
sample.txt

What do you think?

---ASSISTANT
I've analyzed the resource and created a response:

---RESOURCE
response.txt

Let me know if you need more details.
"""
        template = PromptTemplate(template_text)

        # Should have 2 sections (user and assistant), each with a resource
        assert len(template.content_sections) == 2

        # Check user section
        assert template.content_sections[0].role == "user"
        assert "Hello! Check out this resource:" in template.content_sections[0].text
        assert "What do you think?" in template.content_sections[0].text
        assert template.content_sections[0].resources == ["sample.txt"]

        # Check assistant section
        assert template.content_sections[1].role == "assistant"
        assert "I've analyzed the resource" in template.content_sections[1].text
        assert "Let me know if you need more details." in template.content_sections[1].text
        assert template.content_sections[1].resources == ["response.txt"]

    def test_multiple_resources_in_template(self):
        """Test parsing a template with multiple resources per section"""
        template_text = """---USER
Let me share some files with you:

---RESOURCE
file1.txt

---RESOURCE
file2.txt

What do you think of these?

---ASSISTANT
I've analyzed both files:

---RESOURCE
analysis1.txt

---RESOURCE
analysis2.txt

Here are my thoughts.
"""
        template = PromptTemplate(template_text)

        # Should have 2 sections (user and assistant), each with 2 resources
        assert len(template.content_sections) == 2

        # Check user section
        assert template.content_sections[0].role == "user"
        assert template.content_sections[0].resources == ["file1.txt", "file2.txt"]

        # Check assistant section
        assert template.content_sections[1].role == "assistant"
        assert template.content_sections[1].resources == [
            "analysis1.txt",
            "analysis2.txt",
        ]

    def test_apply_substitutions(self):
        """Test applying substitutions to an entire template"""
        template_text = """---USER
Hello {{name}}!

---RESOURCE
data_{{name}}.txt

---ASSISTANT
Hi {{name}}! How can I help you today?

---USER
Tell me about {{topic}}.
"""
        template = PromptTemplate(template_text)
        context = {"name": "Dave", "topic": "Python"}

        result = template.apply_substitutions(context)

        assert len(result) == 3
        assert result[0].text == "Hello Dave!"
        assert result[0].resources == ["data_Dave.txt"]
        assert result[1].text == "Hi Dave! How can I help you today?"
        assert result[2].text == "Tell me about Python."


class TestPromptTemplateLoader:
    """Tests for the PromptTemplateLoader class"""

    @pytest.fixture
    def temp_template_file(self):
        """Create a temporary template file for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""Hello {{name}}!
            
This is a test prompt with {{variable}} substitution.""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    @pytest.fixture
    def temp_delimited_file(self):
        """Create a temporary delimited template file for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
Hello {{name}}!

---RESOURCE
some_resource.txt

---ASSISTANT
Nice to meet you, {{name}}!

---RESOURCE
another_resource.txt""")
            tf_path = Path(tf.name)

        # Create the resource files in the same directory
        resource_path1 = tf_path.parent / "some_resource.txt"
        resource_path2 = tf_path.parent / "another_resource.txt"
        with open(resource_path1, "w", encoding="utf-8") as rf:
            rf.write("This is some resource content")
        with open(resource_path2, "w", encoding="utf-8") as rf:
            rf.write("This is another resource content")

        yield tf_path

        # Cleanup
        os.unlink(tf_path)
        if resource_path1.exists():
            os.unlink(resource_path1)
        if resource_path2.exists():
            os.unlink(resource_path2)

    def test_load_from_file(self, temp_template_file):
        """Test loading a template from a file"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(temp_template_file)

        assert isinstance(template, PromptTemplate)
        assert len(template.content_sections) == 1
        assert template.template_variables == {"name", "variable"}

    def test_get_metadata_simple(self, temp_template_file):
        """Test getting metadata from a simple template file"""
        loader = PromptTemplateLoader()
        metadata = loader.get_metadata(temp_template_file)

        assert isinstance(metadata, PromptMetadata)
        assert metadata.name == temp_template_file.stem
        # The description format can be either the first line or "Simple prompt: filename"
        # so we just check that we got a reasonable description
        assert metadata.description.startswith("Simple prompt:") or "Hello" in metadata.description
        assert metadata.template_variables == {"name", "variable"}
        assert metadata.resource_paths == []
        assert metadata.file_path == temp_template_file

    def test_get_metadata_delimited(self, temp_delimited_file):
        """Test getting metadata from a delimited template file"""
        loader = PromptTemplateLoader()
        metadata = loader.get_metadata(temp_delimited_file)

        assert isinstance(metadata, PromptMetadata)
        assert metadata.name == temp_delimited_file.stem
        # Check for the new format with role in brackets
        assert "[USER]" in metadata.description
        assert "Hello" in metadata.description
        # Make sure filename is not in the description
        assert temp_delimited_file.stem not in metadata.description
        assert metadata.template_variables == {"name"}
        # Should find both resources
        assert set(metadata.resource_paths) == {
            "some_resource.txt",
            "another_resource.txt",
        }
        assert metadata.file_path == temp_delimited_file

    def test_load_template_with_resources(self, temp_delimited_file):
        """Test loading a template with resources"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(temp_delimited_file)

        # Check that we have the right number of sections
        assert len(template.content_sections) == 2

        # Check that resources are properly associated with their sections
        user_section = template.content_sections[0]
        assistant_section = template.content_sections[1]

        assert user_section.role == "user"
        assert "Hello {{name}}!" in user_section.text
        assert user_section.resources == ["some_resource.txt"]

        assert assistant_section.role == "assistant"
        assert "Nice to meet you, {{name}}!" in assistant_section.text
        assert assistant_section.resources == ["another_resource.txt"]


# Integration test with realistic examples
class TestImageHandling:
    """Tests for image handling in prompt templates"""

    @pytest.fixture
    def temp_image_file(self):
        """Create a temporary PNG image file for testing"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as tf:
            # Decode the base64 PNG and write to file
            image_data = base64.b64decode(TINY_IMAGE_PNG)
            tf.write(image_data)
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    @pytest.fixture
    def temp_image_prompt_file(self, temp_image_file):
        """Create a prompt file that references an image"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write(f"""---USER
Can you analyze this image?

---RESOURCE
{temp_image_file.name}

---ASSISTANT
Here's my analysis of the image:

This appears to be a 1x1 pixel test image.
""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    def test_is_image_mime_type(self):
        """Test the image MIME type detection function"""
        # Image types should return True
        assert mime_utils.is_image_mime_type("image/png") is True
        assert mime_utils.is_image_mime_type("image/jpeg") is True
        assert mime_utils.is_image_mime_type("image/gif") is True
        assert mime_utils.is_image_mime_type("image/webp") is True

        # Non-image types should return False
        assert mime_utils.is_image_mime_type("text/plain") is False
        assert mime_utils.is_image_mime_type("application/json") is False
        assert mime_utils.is_image_mime_type("text/html") is False

        # SVG is treated as a special case (it's text-based)
        assert mime_utils.is_image_mime_type("image/svg+xml") is False

    def test_create_image_content(self):
        """Test creating ImageContent objects"""
        # Test with our sample PNG
        image_content = resource_utils.create_image_content(
            data=TINY_IMAGE_PNG, mime_type="image/png"
        )

        # Verify structure
        assert isinstance(image_content, ImageContent)
        assert image_content.type == "image"
        assert image_content.data == TINY_IMAGE_PNG
        assert image_content.mimeType == "image/png"

    def test_binary_resource_handling(self, temp_image_file):
        """Test binary resource handling with images"""
        # Test that we can properly detect and load binary resources
        mime_type = mime_utils.guess_mime_type(str(temp_image_file))

        # This should be detected as an image
        assert mime_utils.is_image_mime_type(mime_type) is True

        # Load the binary content
        content, mime_type, is_binary = resource_utils.load_resource_content(
            str(temp_image_file), prompt_files=[Path(temp_image_file).parent]
        )

        # Verify it's handled as binary
        assert is_binary is True
        assert mime_type == "image/png"

        # Ensure the content is a base64-encoded string
        # Try to decode it to verify it's valid base64
        try:
            decoded = base64.b64decode(content)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Failed to decode base64 content: {e}")

    def test_prompt_template_with_image(self, temp_image_prompt_file, temp_image_file):
        """Test parsing a template with an image resource"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(temp_image_prompt_file)

        # Check that we have the right number of sections and the image resource
        assert len(template.content_sections) == 2
        assert template.content_sections[0].role == "user"
        assert template.content_sections[0].resources == [temp_image_file.name]

    def test_create_messages_with_image(self, temp_image_prompt_file, temp_image_file):
        """Test creating messages with an image resource"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(temp_image_prompt_file)

        # Get the content sections
        content_sections = template.content_sections

        # Create messages with resources
        messages = create_messages_with_resources(
            content_sections, prompt_files=[temp_image_prompt_file]
        )

        # We should have 4 messages:
        # 1. User text message
        # 2. User image message
        # 3. Assistant text message
        assert len(messages) == 3

        # Check user text message
        assert messages[0].role == "user"
        assert isinstance(messages[0].content, TextContent)
        assert "Can you analyze this image?" in messages[0].content.text

        # Check user image message
        assert messages[1].role == "user"
        assert isinstance(messages[1].content, ImageContent)
        assert messages[1].content.type == "image"
        assert messages[1].content.mimeType == "image/png"
        # The data should be our base64 PNG (or equivalent)
        assert isinstance(messages[1].content.data, str)
        assert len(messages[1].content.data) > 0

        # Check assistant message
        assert messages[2].role == "assistant"
        assert isinstance(messages[2].content, TextContent)
        assert "Here's my analysis of the image:" in messages[2].content.text

    def test_resource_handling_functions(self, temp_image_file):
        """Test the internal resource handling functions used by the MCP server"""

        # Test a small custom resource handler function that mimics the server's implementation
        async def read_resource(resource_path):
            mime_type = mime_utils.guess_mime_type(str(resource_path))
            is_binary = mime_utils.is_image_mime_type(mime_type) or not mime_type.startswith(
                "text/"
            )

            if is_binary:
                # For binary files, read as binary and base64 encode
                with open(resource_path, "rb") as f:
                    binary_data = f.read()
                    # We need to explicitly base64 encode binary data
                    return base64.b64encode(binary_data).decode("utf-8")
            else:
                # For text files, read as text with UTF-8 encoding
                with open(resource_path, "r", encoding="utf-8") as f:
                    return f.read()

        # Run our simulated resource handler
        path = str(temp_image_file)
        file_result = asyncio.run(read_resource(path))

        # Verify it's a valid base64 string
        try:
            decoded = base64.b64decode(file_result)
            assert len(decoded) > 0
            # Verify the decoded content is a valid PNG file (should start with PNG signature)
            assert decoded.startswith(b"\x89PNG")
        except Exception as e:
            pytest.fail(f"Resource handler did not return valid base64: {e}")

        # Also verify that our direct load_resource_content function produces valid base64
        content, mime_type, is_binary = resource_utils.load_resource_content(
            path, prompt_files=[Path(temp_image_file).parent]
        )

        # The function should produce the same base64 content
        assert content == file_result


class TestIntegration:
    """Integration tests with realistic examples"""

    @pytest.fixture
    def simple_prompt_file(self):
        """Create a simple prompt file for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""Hello, World.

This is {{blah}} foo""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    @pytest.fixture
    def delimited_prompt_file(self):
        """Create a delimited prompt file for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
I want to learn about {{topic}}.

---ASSISTANT
I'd be happy to tell you about {{topic}}!

Here are some key points about {{topic}}:
1. It's very interesting
2. It has a rich history
3. Many people study it

Would you like to know more about any specific aspect of {{topic}}?""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    @pytest.fixture
    def resource_prompt_file(self):
        """Create a prompt file with resources for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
Can you analyze this {{language}} code?

---RESOURCE
sample_{{language}}.txt

---ASSISTANT
Here's my analysis of your {{language}} code:

---RESOURCE
analysis_{{language}}.txt

Would you like me to explain anything in more detail?""")
            tf_path = Path(tf.name)

        # Create sample resource files
        sample_path = tf_path.parent / "sample_python.txt"
        analysis_path = tf_path.parent / "analysis_python.txt"
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("def hello():\n    print('Hello, world!')")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write("# Analysis\nYour function looks good but could use a docstring.")

        yield tf_path

        # Cleanup
        os.unlink(tf_path)
        if sample_path.exists():
            os.unlink(sample_path)
        if analysis_path.exists():
            os.unlink(analysis_path)

    def test_simple_prompt_substitution(self, simple_prompt_file):
        """Test substituting variables in a simple prompt file"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(simple_prompt_file)

        # Verify template variables
        assert template.template_variables == {"blah"}

        # Apply substitutions
        context = {"blah": "substituted"}
        result = template.apply_substitutions(context)

        # Verify result
        assert len(result) == 1
        assert result[0].role == "user"
        assert "This is substituted foo" in result[0].text

    def test_delimited_prompt_substitution(self, delimited_prompt_file):
        """Test substituting variables in a delimited prompt file"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(delimited_prompt_file)

        # Verify template variables
        assert template.template_variables == {"topic"}

        # Apply substitutions
        context = {"topic": "Python programming"}
        result = template.apply_substitutions(context)

        # Verify result
        assert len(result) == 2
        assert result[0].role == "user"
        assert "I want to learn about Python programming." in result[0].text

        assert result[1].role == "assistant"
        assert "I'd be happy to tell you about Python programming!" in result[1].text
        assert "Here are some key points about Python programming:" in result[1].text

    def test_resource_prompt_substitution(self, resource_prompt_file):
        """Test substituting variables in a prompt file with resources"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(resource_prompt_file)

        # Verify template variables
        assert template.template_variables == {"language"}

        # Apply substitutions
        context = {"language": "python"}
        result = template.apply_substitutions(context)

        # Verify result
        assert len(result) == 2

        # Check user section
        assert result[0].role == "user"
        assert "Can you analyze this python code?" in result[0].text
        assert result[0].resources == ["sample_python.txt"]

        # Check assistant section
        assert result[1].role == "assistant"
        assert "Here's my analysis of your python code:" in result[1].text
        assert result[1].resources == ["analysis_python.txt"]
