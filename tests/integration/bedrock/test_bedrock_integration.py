"""
Integration tests for AWS Bedrock provider integration with fast-agent.
These tests verify that the Bedrock provider works correctly with the fast-agent client.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.provider_key_manager import ProviderKeyManager
from mcp_agent.llm.providers.augmented_llm_bedrock import BedrockAugmentedLLM
from mcp_agent.llm.providers.multipart_converter_bedrock import BedrockConverter, ModelFamily
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import TextContent, ImageContent


@pytest.fixture
def mock_boto3():
    """Create a mock boto3 client for testing."""
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3') as mock:
        # Mock successful response for invoke_model
        mock_client = MagicMock()
        mock_response = {
            'body': MagicMock()
        }
        # Mock the read method to return a JSON string
        mock_response['body'].read.return_value = (
            '{"content": [{"type": "text", "text": "This is a mocked response"}]}'
        )
        mock_client.invoke_model.return_value = [mock_response], {}
        
        # Configure the boto3 mock to return our mock client
        mock.client.return_value = mock_client
        
        # Configure boto3.Session to return a mock session
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock.Session.return_value = mock_session
        
        yield mock


@pytest.fixture
def bedrock_config():
    """Create a configuration dictionary for Bedrock integration tests."""
    return {
        "bedrock": {
            "region": "us-east-1",
            "use_default_credentials": True,
            "default_params": {
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_agent_initialization(mock_boto3):
    """Test initializing a Bedrock agent with mocked client."""
    
    # Create the FastAgent with a mock config
    fast = FastAgent("Bedrock Integration Test Agent", config_dict={
        "default_model": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock": {
            "region": "us-east-1",
            "use_default_credentials": True
        }
    })
    
    @fast.agent(
        "bedrock_test",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a helpful assistant."
    )
    async def agent_function():
        async with fast.run() as agent:
            # Verify the agent was created successfully
            assert agent.bedrock_test is not None
            
            # Verify the model was properly initialized
            assert agent.bedrock_test.llm.provider == Provider.BEDROCK
            
            # Verify the configuration was correctly parsed
            assert isinstance(agent.bedrock_test.llm, BedrockAugmentedLLM)
            assert agent.bedrock_test.llm.region == "us-east-1"
            assert agent.bedrock_test.llm.use_default_credentials is True
    
    # Run the agent function
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_agent_send_message(mock_boto3):
    """Test sending a message with a Bedrock agent."""
    
    # Create the FastAgent with a mock config
    fast = FastAgent("Bedrock Message Test Agent", config_dict={
        "default_model": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock": {
            "region": "us-east-1",
            "use_default_credentials": True
        }
    })
    
    @fast.agent(
        "bedrock_test",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a helpful assistant."
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send a simple message and ensure we get a response
            response = await agent.bedrock_test.send("Hello, how are you?")
            assert "mocked response" in response
    
    # Run the agent function
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_model_family_detection():
    """Test Bedrock model family detection."""
    
    # Test Claude model detection
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert BedrockConverter.detect_model_family(model_id) == ModelFamily.CLAUDE
    
    # Test Nova model detection
    model_id = "amazon.nova-23-4k-1024:0"
    assert BedrockConverter.detect_model_family(model_id) == ModelFamily.NOVA
    
    # Test Meta model detection
    model_id = "meta.llama3-8b-instruct-v1:0"
    assert BedrockConverter.detect_model_family(model_id) == ModelFamily.META
    
    # Test unknown model
    model_id = "unknown.model-1-0:0"
    assert BedrockConverter.detect_model_family(model_id) == ModelFamily.UNKNOWN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_message_conversion():
    """Test conversion of messages to Bedrock format."""
    
    # Create a simple multipart message
    text_content = TextContent(type="text", text="Hello, Bedrock!")
    message = PromptMessageMultipart(role="user", content=[text_content])
    
    # Test conversion to Claude format
    claude_message = BedrockConverter.convert_to_bedrock(message, ModelFamily.CLAUDE)
    assert "role" in claude_message
    assert claude_message["role"] == "user"
    assert "content" in claude_message
    assert isinstance(claude_message["content"], list)
    assert claude_message["content"][0]["type"] == "text"
    assert claude_message["content"][0]["text"] == "Hello, Bedrock!"

    # Test conversion to Nova format (text-only)
    nova_message = BedrockConverter.convert_to_bedrock(message, ModelFamily.NOVA)
    assert "text" in nova_message
    assert nova_message["text"] == "Hello, Bedrock!"
    
    # Test conversion to Meta format
    meta_message = BedrockConverter.convert_to_bedrock(message, ModelFamily.META)
    assert "role" in meta_message
    assert meta_message["role"] == "user"
    assert "content" in meta_message
    assert meta_message["content"] == "Hello, Bedrock!"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_system_message_conversion():
    """Test conversion of system messages to Bedrock format."""
    
    # Create a system message
    text_content = TextContent(type="text", text="You are a helpful AI assistant.")
    message = PromptMessageMultipart(role="system", content=[text_content])
    
    # Test conversion to Claude format
    claude_message = BedrockConverter.convert_to_bedrock(message, ModelFamily.CLAUDE)
    assert "role" in claude_message
    assert claude_message["role"] == "user"
    assert "content" in claude_message
    assert isinstance(claude_message["content"], list)
    assert claude_message["content"][0]["type"] == "text"
    assert "<admin>" in claude_message["content"][0]["text"]
    assert "You are a helpful AI assistant." in claude_message["content"][0]["text"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_client_creation_methods(mock_boto3):
    """Test different methods for creating a Bedrock client."""
    
    # Test with default credentials
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockAugmentedLLM.__init__', return_value=None) as mock_init:
        llm = BedrockAugmentedLLM()
        llm.region = "us-east-1"
        llm.profile = None
        llm.endpoint_url = None
        llm.use_default_credentials = True
        llm.access_key_id = None
        llm.secret_access_key = None
        llm.logger = MagicMock()
        
        client = llm._bedrock_client()
        mock_boto3.client.assert_called_with(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
    
    # Test with named profile
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockAugmentedLLM.__init__', return_value=None) as mock_init:
        llm = BedrockAugmentedLLM()
        llm.region = "us-east-1"
        llm.profile = "test-profile"
        llm.endpoint_url = None
        llm.use_default_credentials = False
        llm.access_key_id = None
        llm.secret_access_key = None
        llm.logger = MagicMock()
        
        client = llm._bedrock_client()
        mock_boto3.Session.assert_called_with(profile_name="test-profile")
        mock_boto3.Session.return_value.client.assert_called_with(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
    
    # Test with explicit credentials
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockAugmentedLLM.__init__', return_value=None) as mock_init:
        with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockAugmentedLLM._api_key', return_value="test-key") as mock_api_key:
            with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockAugmentedLLM._secret_key', return_value="test-secret") as mock_secret_key:
                llm = BedrockAugmentedLLM()
                llm.region = "us-east-1"
                llm.profile = None
                llm.endpoint_url = None
                llm.use_default_credentials = False
                llm.access_key_id = None
                llm.secret_access_key = None
                llm.logger = MagicMock()
                llm.context = MagicMock()
                
                client = llm._bedrock_client()
                mock_boto3.client.assert_called_with(
                    aws_access_key_id="test-key",
                    aws_secret_access_key="test-secret",
                    service_name="bedrock-runtime",
                    region_name="us-east-1"
                )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_request_preparation():
    """Test preparation of Bedrock API requests."""
    
    # Create a test instance with minimal configuration
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockAugmentedLLM.__init__', return_value=None) as mock_init:
        llm = BedrockAugmentedLLM()
        llm.region = "us-east-1"
        llm.profile = None
        llm.use_default_credentials = True
        llm.default_params = {}
        llm.model_params = {}
        llm.logger = MagicMock()
        llm.prepare_provider_arguments = MagicMock(return_value={})
        llm.BEDROCK_EXCLUDE_FIELDS = set()
        
        # Create test messages and parameters
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, world!"}]
            }
        ]
        
        from mcp_agent.core.request_params import RequestParams
        request_params = RequestParams(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            temperature=0.7,
            maxTokens=1000,
            parallel_tool_calls=False
        )
        
        # Test Claude request preparation
        claude_request = llm._prepare_claude_request(
            messages, None, request_params, {}
        )
        assert "anthropic_version" in claude_request
        assert claude_request["max_tokens"] == 1000
        assert claude_request["temperature"] == 0.7
        assert "messages" in claude_request
        
        # Test Nova request preparation
        nova_request = llm._prepare_nova_request(
            messages, None, request_params, {}
        )
        assert "inputText" in nova_request
        assert "textGenerationConfig" in nova_request
        assert nova_request["textGenerationConfig"]["maxTokenCount"] == 1000
        assert nova_request["textGenerationConfig"]["temperature"] == 0.7
        
        # Test Meta request preparation
        meta_request = llm._prepare_meta_request(
            messages, None, request_params, {}
        )
        assert "prompt" in meta_request
        assert "max_gen_len" in meta_request
        assert meta_request["temperature"] == 0.7


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_authentication_errors():
    """Test handling of Bedrock authentication errors."""
    
    # Create a FastAgent with intentionally invalid credentials
    fast = FastAgent("Bedrock Auth Test Agent", config_dict={
        "default_model": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock": {
            "region": "us-east-1",
            "use_default_credentials": False,
            "access_key_id": "invalid-key",
            "secret_access_key": "invalid-secret"
        }
    })
    
    # Mock boto3 to raise NoCredentialsError
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3') as mock_boto3:
        from botocore.exceptions import NoCredentialsError
        mock_boto3.client.side_effect = NoCredentialsError()
        
        @fast.agent(
            "bedrock_test",
            model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
            instruction="You are a helpful assistant."
        )
        async def agent_function():
            with pytest.raises(ProviderKeyError):
                async with fast.run() as agent:
                    # This should raise a ProviderKeyError due to invalid credentials
                    pass
        
        # Run the agent function
        await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_tool_calling(mock_boto3):
    """Test tool calling with Bedrock."""
    
    # Mock invoke_model to return a tool_use response
    def mock_invoke_model_with_tool(**kwargs):
        # Parse the request body
        import json
        request_data = json.loads(kwargs["body"])
        
        # Check if this is a normal request or a tool result
        if any(msg.get("content", [{}])[0].get("type") == "tool_result" for msg in request_data.get("messages", [])):
            # This is a follow-up after tool execution, return final response
            mock_response = {
                'body': MagicMock()
            }
            mock_response['body'].read.return_value = (
                '{"content": [{"type": "text", "text": "I used the tool and got the result"}]}'
            )
            return [mock_response], {}
        else:
            # Return a tool_use response first time
            mock_response = {
                'body': MagicMock()
            }
            mock_response['body'].read.return_value = (
                '{"content": [{"type": "text", "text": "I need to use a tool"}], '
                '"tool_use": {"id": "tool123", "name": "get_weather", "input": {"location": "Seattle"}}}'
            )
            return [mock_response], {}
    
    # Configure the mock to use our custom function
    mock_boto3.client.return_value.invoke_model = MagicMock(side_effect=mock_invoke_model_with_tool)
    
    # Create the FastAgent
    fast = FastAgent("Bedrock Tool Test Agent", config_dict={
        "default_model": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock": {
            "region": "us-east-1",
            "use_default_credentials": True
        }
    })
    
    # Define a test tool
    @fast.tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny."
    
    @fast.agent(
        "bedrock_test",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a helpful assistant."
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send a message that will trigger tool use
            response = await agent.bedrock_test.send("What's the weather in Seattle?")
            
            # Verify the response contains the expected output
            assert "used the tool" in response
    
    # Run the agent function
    with patch('mcp_agent.llm.augmented_llm.AugmentedLLM.call_tool') as mock_call_tool:
        from mcp.types import CallToolResult
        # Mock successful tool call result
        mock_call_tool.return_value = CallToolResult(
            result='{"result": "The weather in Seattle is sunny."}'
        )
        
        await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_with_images_and_resources(mock_boto3):
    """Test Bedrock with image content."""
    
    # Create the FastAgent
    fast = FastAgent("Bedrock Image Test Agent", config_dict={
        "default_model": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock": {
            "region": "us-east-1",
            "use_default_credentials": True
        }
    })
    
    # Create a mock image for testing
    from pydantic import AnyUrl
    from mcp.types import ImageResourceContents, EmbeddedResource
    
    # Create a mock image resource
    image_resource = ImageResourceContents(
        uri=AnyUrl("file:///test/image.png"),
        mimeType="image/png",
        byteValues=[0, 1, 2, 3, 4]  # Dummy byte values
    )
    
    # Create an embedded resource
    image_embedded = EmbeddedResource(
        type="resource",
        resource=image_resource
    )
    
    @fast.agent(
        "bedrock_test",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a helpful assistant that can analyze images."
    )
    async def agent_function():
        async with fast.run() as agent:
            # Mock Prompt.user call with text and image
            with patch('mcp_agent.core.prompt.Prompt.user') as mock_user:
                # Configure mock_user to return a message with text and image content
                message = PromptMessageMultipart(
                    role="user",
                    content=[
                        TextContent(type="text", text="What's in this image?"),
                        image_embedded
                    ]
                )
                mock_user.return_value = message
                
                # Send the message
                response = await agent.bedrock_test.send(
                    Prompt.user("What's in this image?", image_embedded)
                )
                
                # Verify we got a response (the actual API is mocked)
                assert "mocked response" in response
    
    # Run the agent function
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_config_validation():
    """Test that Bedrock configuration is properly validated."""
    
    # Test missing region
    with pytest.raises(ProviderKeyError, match="Missing AWS region"):
        fast = FastAgent("Bedrock Config Test", config_dict={
            "default_model": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "bedrock": {
                # Missing region
                "use_default_credentials": True
            }
        })
        
        @fast.agent(
            "bedrock_test",
            model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        async def agent_function():
            async with fast.run():
                pass
        
        await agent_function()
    
    # Test missing credentials
    with pytest.raises(ProviderKeyError, match="Missing AWS credentials"):
        # Mock the _api_key and _secret_key methods to return None
        with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockAugmentedLLM._api_key', return_value=None):
            with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockAugmentedLLM._secret_key', return_value=None):
                fast = FastAgent("Bedrock Config Test", config_dict={
                    "default_model": "bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "bedrock": {
                        "region": "us-east-1",
                        "use_default_credentials": False
                        # Missing credentials
                    }
                })
                
                @fast.agent(
                    "bedrock_test",
                    model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0"
                )
                async def agent_function():
                    async with fast.run():
                        pass
                
                await agent_function()