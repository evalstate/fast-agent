from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ImageContent,
    SamplingMessage,
    TextContent,
)

from fast_agent.llm.sampling_converter import SamplingConverter


class TestSamplingConverter:
    """Tests for SamplingConverter"""

    def test_sampling_message_to_prompt_message_text(self):
        """Test converting a text SamplingMessage to PromptMessageExtended"""
        # Create a SamplingMessage with text content
        text_content = TextContent(type="text", text="Hello, world!")
        sampling_message = SamplingMessage(role="user", content=text_content)

        # Convert using our converter
        prompt_message = SamplingConverter.sampling_message_to_prompt_message(sampling_message)

        # Verify conversion
        assert prompt_message.role == "user"
        assert len(prompt_message.content) == 1
        assert prompt_message.content[0].type == "text"
        assert prompt_message.content[0].text == "Hello, world!"

    def test_sampling_message_to_prompt_message_image(self):
        """Test converting an image SamplingMessage to PromptMessageExtended"""
        # Create a SamplingMessage with image content
        image_content = ImageContent(
            type="image", data="base64_encoded_image_data", mimeType="image/png"
        )
        sampling_message = SamplingMessage(role="user", content=image_content)

        # Convert using our converter
        prompt_message = SamplingConverter.sampling_message_to_prompt_message(sampling_message)

        # Verify conversion
        assert prompt_message.role == "user"
        assert len(prompt_message.content) == 1
        assert prompt_message.content[0].type == "image"
        assert prompt_message.content[0].data == "base64_encoded_image_data"
        assert prompt_message.content[0].mimeType == "image/png"

    def test_convert_messages(self):
        """Test converting multiple SamplingMessages to PromptMessageExtended objects"""
        # Create a list of SamplingMessages with different roles
        messages = [
            SamplingMessage(role="user", content=TextContent(type="text", text="Hello")),
            SamplingMessage(role="assistant", content=TextContent(type="text", text="Hi there")),
            SamplingMessage(role="user", content=TextContent(type="text", text="How are you?")),
        ]

        # Convert all messages
        prompt_messages = SamplingConverter.convert_messages(messages)

        # Verify we got the right number of messages
        assert len(prompt_messages) == 3

        # Verify each message was converted correctly
        assert prompt_messages[0].role == "user"
        assert prompt_messages[0].content[0].text == "Hello"

        assert prompt_messages[1].role == "assistant"
        assert prompt_messages[1].content[0].text == "Hi there"

        assert prompt_messages[2].role == "user"
        assert prompt_messages[2].content[0].text == "How are you?"

    def test_convert_messages_with_mixed_content_types(self):
        """Test converting messages with different content types"""
        # Create a list with both text and image content
        messages = [
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text="What's in this image?"),
            ),
            SamplingMessage(
                role="user",
                content=ImageContent(
                    type="image", data="base64_encoded_image_data", mimeType="image/png"
                ),
            ),
        ]

        # Convert all messages
        prompt_messages = SamplingConverter.convert_messages(messages)

        # Verify conversion
        assert len(prompt_messages) == 2

        # First message (text)
        assert prompt_messages[0].role == "user"
        assert prompt_messages[0].content[0].type == "text"
        assert prompt_messages[0].content[0].text == "What's in this image?"

        # Second message (image)
        assert prompt_messages[1].role == "user"
        assert prompt_messages[1].content[0].type == "image"
        assert prompt_messages[1].content[0].data == "base64_encoded_image_data"
        assert prompt_messages[1].content[0].mimeType == "image/png"

    def test_extract_request_params_full(self):
        """Test extracting RequestParams from CreateMessageRequestParams with all fields"""
        # Create a CreateMessageRequestParams with all fields
        request_params = CreateMessageRequestParams(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
            maxTokens=1000,
            systemPrompt="You are a helpful assistant",
            temperature=0.7,
            stopSequences=["STOP", "\n\n"],
            includeContext="none",
        )

        # Extract parameters using our converter
        llm_params = SamplingConverter.extract_request_params(request_params)

        # Verify parameters
        assert llm_params.maxTokens == 1000
        assert llm_params.systemPrompt == "You are a helpful assistant"
        assert llm_params.temperature == 0.7
        assert llm_params.stopSequences == ["STOP", "\n\n"]
        assert llm_params.modelPreferences == request_params.modelPreferences

    def test_extract_request_params_minimal(self):
        """Test extracting RequestParams from CreateMessageRequestParams with minimal fields"""
        # Create a CreateMessageRequestParams with minimal fields
        request_params = CreateMessageRequestParams(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
            maxTokens=100,  # Only required field besides messages
        )

        # Extract parameters using our converter
        llm_params = SamplingConverter.extract_request_params(request_params)

        # Verify parameters
        assert llm_params.maxTokens == 100
        assert llm_params.systemPrompt is None
        assert llm_params.temperature is None
        assert llm_params.stopSequences is None
        assert llm_params.modelPreferences is None

    def test_error_result(self):
        """Test creating an error result"""
        # Error message and model
        error_message = "Error in sampling: Test error"
        model = "test-model"

        # Create error result using our converter
        result = SamplingConverter.error_result(error_message=error_message, model=model)

        # Verify result
        assert isinstance(result, CreateMessageResult)
        assert result.role == "assistant"
        assert result.content.type == "text"
        assert result.content.text == "Error in sampling: Test error"
        assert result.model == model
        assert result.stopReason == "error"

    def test_error_result_no_model(self):
        """Test creating an error result without a model"""
        # Create error result without specifying a model
        result = SamplingConverter.error_result(error_message="Error in sampling: Test error")

        # Verify the default model value is used
        assert result.model == "unknown"
        assert result.stopReason == "error"
