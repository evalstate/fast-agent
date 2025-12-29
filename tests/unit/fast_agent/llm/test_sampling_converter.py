from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CreateMessageRequestParams,
    CreateMessageResult,
    ImageContent,
    SamplingMessage,
    TextContent,
    ToolResultContent,
    ToolUseContent,
)

from fast_agent.llm.sampling_converter import SamplingConverter
from fast_agent.types import PromptMessageExtended


def _text(block: object) -> TextContent:
    assert isinstance(block, TextContent)
    return block


def _image(block: object) -> ImageContent:
    assert isinstance(block, ImageContent)
    return block


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
        assert _text(prompt_message.content[0]).type == "text"
        assert _text(prompt_message.content[0]).text == "Hello, world!"

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
        image_block = _image(prompt_message.content[0])
        assert image_block.type == "image"
        assert image_block.data == "base64_encoded_image_data"
        assert image_block.mimeType == "image/png"

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
        assert _text(prompt_messages[0].content[0]).text == "Hello"

        assert prompt_messages[1].role == "assistant"
        assert _text(prompt_messages[1].content[0]).text == "Hi there"

        assert prompt_messages[2].role == "user"
        assert _text(prompt_messages[2].content[0]).text == "How are you?"

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
        assert _text(prompt_messages[0].content[0]).type == "text"
        assert _text(prompt_messages[0].content[0]).text == "What's in this image?"

        # Second message (image)
        assert prompt_messages[1].role == "user"
        image_block = _image(prompt_messages[1].content[0])
        assert image_block.type == "image"
        assert image_block.data == "base64_encoded_image_data"
        assert image_block.mimeType == "image/png"

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
        assert _text(result.content).type == "text"
        assert _text(result.content).text == "Error in sampling: Test error"
        assert result.model == model
        assert result.stopReason == "error"

    def test_error_result_no_model(self):
        """Test creating an error result without a model"""
        # Create error result without specifying a model
        result = SamplingConverter.error_result(error_message="Error in sampling: Test error")

        # Verify the default model value is used
        assert result.model == "unknown"
        assert result.stopReason == "error"

    def test_sampling_message_with_tool_result(self):
        """Test converting a SamplingMessage with ToolResultContent"""
        # Create a SamplingMessage with tool result content
        tool_result = ToolResultContent(
            type="tool_result",
            toolUseId="call_123",
            content=[TextContent(type="text", text="Tool result: 42")],
        )
        sampling_message = SamplingMessage(role="user", content=tool_result)

        # Convert using our converter
        prompt_message = SamplingConverter.sampling_message_to_prompt_message(sampling_message)

        # Verify conversion
        assert prompt_message.role == "user"
        assert prompt_message.tool_results is not None
        assert "call_123" in prompt_message.tool_results
        tool_content = prompt_message.tool_results["call_123"].content
        assert tool_content is not None
        assert _text(tool_content[0]).text == "Tool result: 42"

    def test_sampling_message_with_tool_use(self):
        """Test converting a SamplingMessage with ToolUseContent (assistant response)"""
        # Create a SamplingMessage with tool use content
        tool_use = ToolUseContent(
            type="tool_use",
            id="call_456",
            name="calculator",
            input={"a": 5, "b": 3},
        )
        sampling_message = SamplingMessage(role="assistant", content=tool_use)

        # Convert using our converter
        prompt_message = SamplingConverter.sampling_message_to_prompt_message(sampling_message)

        # Verify conversion
        assert prompt_message.role == "assistant"
        assert prompt_message.tool_calls is not None
        assert "call_456" in prompt_message.tool_calls
        assert prompt_message.tool_calls["call_456"].params.name == "calculator"
        assert prompt_message.tool_calls["call_456"].params.arguments == {"a": 5, "b": 3}

    def test_sampling_message_with_multiple_tool_results(self):
        """Test converting a SamplingMessage with multiple tool results"""
        # Create a SamplingMessage with multiple tool results (list content)
        tool_results = [
            ToolResultContent(
                type="tool_result",
                toolUseId="call_1",
                content=[TextContent(type="text", text="Result 1")],
            ),
            ToolResultContent(
                type="tool_result",
                toolUseId="call_2",
                content=[TextContent(type="text", text="Result 2")],
            ),
        ]
        sampling_message = SamplingMessage(role="user", content=tool_results)

        # Convert using our converter
        prompt_message = SamplingConverter.sampling_message_to_prompt_message(sampling_message)

        # Verify conversion
        assert prompt_message.role == "user"
        assert prompt_message.tool_results is not None
        assert len(prompt_message.tool_results) == 2
        assert "call_1" in prompt_message.tool_results
        assert "call_2" in prompt_message.tool_results

    def test_llm_response_to_sampling_content_text_only(self):
        """Test converting an LLM response with only text to sampling content"""
        # Create an LLM response with text content
        response = PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Hello, world!")],
        )

        # Convert using our converter
        content_blocks = SamplingConverter.llm_response_to_sampling_content(response)

        # Verify conversion
        assert len(content_blocks) == 1
        assert isinstance(content_blocks[0], TextContent)
        assert content_blocks[0].text == "Hello, world!"

    def test_llm_response_to_sampling_content_with_tool_calls(self):
        """Test converting an LLM response with tool calls to sampling content"""
        # Create an LLM response with tool calls
        response = PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="I'll calculate that for you.")],
            tool_calls={
                "call_abc": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="add",
                        arguments={"a": 5, "b": 3},
                    ),
                ),
            },
        )

        # Convert using our converter
        content_blocks = SamplingConverter.llm_response_to_sampling_content(response)

        # Verify conversion
        assert len(content_blocks) == 2  # 1 text + 1 tool use

        # Check text content
        text_blocks = [b for b in content_blocks if isinstance(b, TextContent)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "I'll calculate that for you."

        # Check tool use content
        tool_blocks = [b for b in content_blocks if isinstance(b, ToolUseContent)]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].id == "call_abc"
        assert tool_blocks[0].name == "add"
        assert tool_blocks[0].input == {"a": 5, "b": 3}

    def test_llm_response_to_sampling_content_multiple_tool_calls(self):
        """Test converting an LLM response with multiple tool calls"""
        # Create an LLM response with multiple tool calls (parallel tool use)
        response = PromptMessageExtended(
            role="assistant",
            content=[],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="get_weather", arguments={"city": "Paris"}),
                ),
                "call_2": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="get_weather", arguments={"city": "London"}),
                ),
            },
        )

        # Convert using our converter
        content_blocks = SamplingConverter.llm_response_to_sampling_content(response)

        # Verify conversion
        assert len(content_blocks) == 2

        # Check all are tool use content
        tool_blocks = [b for b in content_blocks if isinstance(b, ToolUseContent)]
        assert len(tool_blocks) == 2

        # Verify the tool calls (order may vary)
        tool_ids = {b.id for b in tool_blocks}
        assert tool_ids == {"call_1", "call_2"}
