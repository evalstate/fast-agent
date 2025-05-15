"""
Unit tests for the BedrockAugmentedLLM class.
"""

import json
import types
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from mcp_agent.llm.provider_types import Provider
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.llm.providers.augmented_llm_bedrock import BedrockAugmentedLLM
from mcp_agent.llm.providers.multipart_converter_bedrock import ModelFamily


class DummyLogger:
    enable_markup = True
    
    def debug(self, msg):
        pass
    
    def warning(self, msg):
        pass
    
    def error(self, msg):
        pass


class DummyBedrockConfig:
    def __init__(self):
        self.region: str = "us-east-1"
        self.profile: Optional[str] = None
        self.endpoint_url: Optional[str] = None
        self.use_default_credentials: bool = False
        self.access_key_id: Optional[str] = "test-access-key"
        self.secret_access_key: Optional[str] = "test-secret-key"
        self.default_params: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9
        }
        self.model_params: Dict[str, Dict[str, Any]] = {
            "anthropic.claude-3-5-sonnet-20241022-v2:0": {
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 50
            }
        }


class DummyConfig:
    def __init__(self, bedrock_cfg=None):
        self.bedrock = bedrock_cfg or DummyBedrockConfig()
        self.logger = DummyLogger()


class DummyExecutor:
    """Dummy executor for tests."""
    
    def __init__(self, result=None):
        self.result = result or {}
    
    async def execute(self, func, *args, **kwargs):
        """Return a fixed result or the default empty dict."""
        if isinstance(self.result, Exception):
            return self.result
        
        # For boto3 clients, modelId and body are passed as kwargs
        model_id = kwargs.get("modelId", "")
        response_body = {"content": [{"type": "text", "text": "Test response"}]}
        
        if "claude" in model_id.lower():
            # Claude-style response
            pass
        elif "titan" in model_id.lower() or "nova" in model_id.lower():
            # Nova-style response
            response_body = {"results": [{"outputText": "Test response"}]}
        elif "llama" in model_id.lower():
            # Meta-style response
            response_body = {"generation": "Test response"}
        
        return [{
            'body': types.SimpleNamespace(
                read=lambda: json.dumps(self.result or response_body).encode()
            )
        }]


class DummyAggregator:
    """Dummy aggregator for tests."""
    
    def __init__(self, tools=None):
        self.tools = tools or []
    
    async def list_tools(self):
        """Return a list of tools."""
        return types.SimpleNamespace(tools=self.tools)


class DummyContext:
    def __init__(self, bedrock_cfg=None, executor_result=None, tools=None):
        self.config = DummyConfig(bedrock_cfg=bedrock_cfg)
        self.executor = DummyExecutor(result=executor_result)
        self.aggregator = DummyAggregator(tools=tools)


def test_init_with_valid_config():
    """Test initialization with valid configuration."""
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        llm = BedrockAugmentedLLM(context=ctx)
        
        # Basic configuration checks
        assert llm.region == "us-east-1"
        assert llm.access_key_id == "test-access-key"
        assert llm.secret_access_key == "test-secret-key"
        assert llm.provider == Provider.BEDROCK


def test_init_missing_region():
    """Test initialization fails with missing region."""
    cfg = DummyBedrockConfig()
    cfg.region = None
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        with pytest.raises(ProviderKeyError, match="Missing AWS region"):
            BedrockAugmentedLLM(context=ctx)


def test_init_missing_credentials():
    """Test initialization fails with missing credentials."""
    cfg = DummyBedrockConfig()
    cfg.access_key_id = None
    cfg.secret_access_key = None
    # Disable default credentials
    cfg.use_default_credentials = False
    cfg.profile = None
    
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        with patch('mcp_agent.llm.providers.augmented_llm_bedrock.ProviderKeyManager.get_aws_secret_key', 
                   return_value=None):
            with pytest.raises(ProviderKeyError, match="Missing AWS credentials"):
                BedrockAugmentedLLM(context=ctx)


def test_init_with_default_credentials():
    """Test initialization with default credentials."""
    cfg = DummyBedrockConfig()
    cfg.access_key_id = None
    cfg.secret_access_key = None
    cfg.use_default_credentials = True
    
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        llm = BedrockAugmentedLLM(context=ctx)
        
        # Check configuration
        assert llm.region == "us-east-1"
        assert llm.use_default_credentials == True


def test_init_with_profile():
    """Test initialization with AWS profile."""
    cfg = DummyBedrockConfig()
    cfg.access_key_id = None
    cfg.secret_access_key = None
    cfg.profile = "test-profile"
    
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        llm = BedrockAugmentedLLM(context=ctx)
        
        # Check configuration
        assert llm.region == "us-east-1"
        assert llm.profile == "test-profile"


def test_init_missing_boto3():
    """Test initialization fails when boto3 is not installed."""
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', None):
        with pytest.raises(ProviderKeyError, match="boto3 not installed"):
            BedrockAugmentedLLM(context=ctx)


def test_bedrock_client_with_access_key():
    """Test _bedrock_client with access key authentication."""
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    mock_client = MagicMock()
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()) as mock_boto3:
        mock_boto3.client.return_value = mock_client
        
        llm = BedrockAugmentedLLM(context=ctx)
        client = llm._bedrock_client()
        
        # Check that boto3.client was called with the correct arguments
        mock_boto3.client.assert_called_with(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key"
        )
        
        assert client == mock_client


def test_bedrock_client_with_default_credentials():
    """Test _bedrock_client with default credentials."""
    cfg = DummyBedrockConfig()
    cfg.access_key_id = None
    cfg.secret_access_key = None
    cfg.use_default_credentials = True
    
    ctx = DummyContext(bedrock_cfg=cfg)
    
    mock_client = MagicMock()
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()) as mock_boto3:
        mock_boto3.client.return_value = mock_client
        
        llm = BedrockAugmentedLLM(context=ctx)
        client = llm._bedrock_client()
        
        # Check that boto3.client was called with the correct arguments
        mock_boto3.client.assert_called_with(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        
        assert client == mock_client


def test_bedrock_client_with_profile():
    """Test _bedrock_client with named profile."""
    cfg = DummyBedrockConfig()
    cfg.access_key_id = None
    cfg.secret_access_key = None
    cfg.profile = "test-profile"
    
    ctx = DummyContext(bedrock_cfg=cfg)
    
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_session.client.return_value = mock_client
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()) as mock_boto3:
        mock_boto3.Session.return_value = mock_session
        
        llm = BedrockAugmentedLLM(context=ctx)
        client = llm._bedrock_client()
        
        # Check that boto3.Session was called with the correct profile
        mock_boto3.Session.assert_called_with(profile_name="test-profile")
        
        # Check that session.client was called with the correct arguments
        mock_session.client.assert_called_with(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        
        assert client == mock_client


def test_bedrock_client_with_endpoint_url():
    """Test _bedrock_client with custom endpoint URL."""
    cfg = DummyBedrockConfig()
    cfg.endpoint_url = "https://bedrock-runtime.custom-endpoint.amazonaws.com"
    
    ctx = DummyContext(bedrock_cfg=cfg)
    
    mock_client = MagicMock()
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()) as mock_boto3:
        mock_boto3.client.return_value = mock_client
        
        llm = BedrockAugmentedLLM(context=ctx)
        client = llm._bedrock_client()
        
        # Check that boto3.client was called with the correct arguments
        mock_boto3.client.assert_called_with(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            endpoint_url="https://bedrock-runtime.custom-endpoint.amazonaws.com",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key"
        )
        
        assert client == mock_client


def test_bedrock_client_with_no_credentials_error():
    """Test _bedrock_client with NoCredentialsError."""
    cfg = DummyBedrockConfig()
    cfg.access_key_id = None
    cfg.secret_access_key = None
    
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()) as mock_boto3:
        with patch('mcp_agent.llm.providers.augmented_llm_bedrock.ProviderKeyManager.get_aws_secret_key', 
                   return_value=None):
            with patch('mcp_agent.llm.providers.augmented_llm_bedrock.NoCredentialsError', 
                       Exception) as mock_error:
                mock_boto3.client.side_effect = mock_error("No credentials found")
                
                llm = BedrockAugmentedLLM(context=ctx)
                
                with pytest.raises(ProviderKeyError, match="No AWS credentials found"):
                    llm._bedrock_client()


def test_initialize_default_params():
    """Test _initialize_default_params method."""
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        llm = BedrockAugmentedLLM(context=ctx)
        
        # Test with custom model
        params = llm._initialize_default_params({"model": "anthropic.claude-3-7-sonnet-20250219-v1:0"})
        
        assert params.model == "anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert params.systemPrompt == llm.instruction
        assert params.parallel_tool_calls == True
        assert params.max_iterations == 20
        assert params.use_history == True
        
        # Test with default model
        params = llm._initialize_default_params({})
        
        assert params.model == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"  # Default value


def test_prepare_claude_request():
    """Test _prepare_claude_request method."""
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        llm = BedrockAugmentedLLM(context=ctx)
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}
        ]
        
        # Create request params
        from mcp_agent.core.request_params import RequestParams
        request_params = RequestParams(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            temperature=0.8,
            top_p=0.95,
            maxTokens=1000
        )
        
        # Test without tools
        result = llm._prepare_claude_request(messages, None, request_params, {})
        
        assert result["anthropic_version"] == "bedrock-2023-05-31"
        assert result["max_tokens"] == 1000
        assert result["temperature"] == 0.8
        assert result["top_p"] == 0.95
        assert result["messages"] == messages
        
        # Test with tools
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
        
        result = llm._prepare_claude_request(messages, available_tools, request_params, {})
        
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["description"] == "Get the current weather"
        assert "tool_choice" in result
        assert result["tool_choice"] == "auto"


def test_prepare_nova_request():
    """Test _prepare_nova_request method."""
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        llm = BedrockAugmentedLLM(context=ctx)
        
        # Create a user message
        messages = [
            {"role": "user", "text": "Tell me about AI"}
        ]
        
        # Create request params
        from mcp_agent.core.request_params import RequestParams
        request_params = RequestParams(
            model="amazon.titan-text-express-v1",
            temperature=0.8,
            top_p=0.95,
            maxTokens=500
        )
        
        # Test without tools
        result = llm._prepare_nova_request(messages, None, request_params, {})
        
        assert "inputText" in result
        assert result["inputText"] == "Tell me about AI"
        assert "textGenerationConfig" in result
        assert result["textGenerationConfig"]["maxTokenCount"] == 500
        assert result["textGenerationConfig"]["temperature"] == 0.8
        assert result["textGenerationConfig"]["topP"] == 0.95


def test_prepare_meta_request():
    """Test _prepare_meta_request method."""
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        llm = BedrockAugmentedLLM(context=ctx)
        
        # Create messages with different roles
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about AI"},
            {"role": "assistant", "content": "AI stands for artificial intelligence."}
        ]
        
        # Create request params
        from mcp_agent.core.request_params import RequestParams
        request_params = RequestParams(
            model="meta.llama3-70b-instruct-v1:0",
            temperature=0.7,
            top_p=0.9,
            maxTokens=1024
        )
        
        # Test without tools
        result = llm._prepare_meta_request(messages, None, request_params, {})
        
        assert "prompt" in result
        assert "<s>" in result["prompt"]  # System message formatting
        assert "<user>" in result["prompt"]  # User message formatting
        assert "<assistant>" in result["prompt"]  # Assistant message formatting
        assert result["max_gen_len"] == 1024
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9


def test_prepare_api_request():
    """Test _prepare_api_request method with different model families."""
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        with patch('mcp_agent.llm.providers.augmented_llm_bedrock.BedrockConverter.detect_model_family') as mock_detect:
            llm = BedrockAugmentedLLM(context=ctx)
            
            messages = [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ]
            
            # Create request params
            from mcp_agent.core.request_params import RequestParams
            
            # Test with Claude model
            mock_detect.return_value = ModelFamily.CLAUDE
            request_params = RequestParams(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
            
            with patch.object(llm, '_prepare_claude_request') as mock_claude:
                mock_claude.return_value = {"model": "claude_request"}
                result = llm._prepare_api_request(messages, None, request_params)
                mock_claude.assert_called_once()
                assert result == {"model": "claude_request"}
            
            # Test with Nova model
            mock_detect.return_value = ModelFamily.NOVA
            request_params = RequestParams(model="amazon.titan-text-express-v1")
            
            with patch.object(llm, '_prepare_nova_request') as mock_nova:
                mock_nova.return_value = {"model": "nova_request"}
                result = llm._prepare_api_request(messages, None, request_params)
                mock_nova.assert_called_once()
                assert result == {"model": "nova_request"}
            
            # Test with Meta model
            mock_detect.return_value = ModelFamily.META
            request_params = RequestParams(model="meta.llama3-70b-instruct-v1:0")
            
            with patch.object(llm, '_prepare_meta_request') as mock_meta:
                mock_meta.return_value = {"model": "meta_request"}
                result = llm._prepare_api_request(messages, None, request_params)
                mock_meta.assert_called_once()
                assert result == {"model": "meta_request"}
            
            # Test with unknown model family
            mock_detect.return_value = ModelFamily.UNKNOWN
            request_params = RequestParams(model="unknown.model")
            
            with patch.object(llm, '_prepare_claude_request') as mock_claude:
                mock_claude.return_value = {"model": "claude_request"}
                result = llm._prepare_api_request(messages, None, request_params)
                mock_claude.assert_called_once()
                assert result == {"model": "claude_request"}


@pytest.mark.asyncio
async def test_apply_prompt_provider_specific():
    """Test _apply_prompt_provider_specific method."""
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart, Role
    from mcp_agent.mcp.prompt_message_multipart import TextContent
    from mcp_agent.core.request_params import RequestParams
    
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    # Create multipart messages
    messages = [
        PromptMessageMultipart(
            role=Role.USER,
            content=[TextContent(type="text", text="What is AI?")]
        ),
    ]
    
    # Success case with Claude model
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        with patch('mcp_agent.llm.providers.augmented_llm_bedrock.json.loads') as mock_json_loads:
            # Set up the mock response
            claude_response = {
                "content": [{"type": "text", "text": "AI stands for Artificial Intelligence."}]
            }
            mock_json_loads.return_value = claude_response
            
            # Initialize the LLM with mocked dependencies
            llm = BedrockAugmentedLLM(context=ctx)
            
            # Replace _bedrock_client with a mock
            llm._bedrock_client = MagicMock()
            
            # Test the method
            request_params = RequestParams(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
            result = await llm._apply_prompt_provider_specific(messages, request_params)
            
            # Check the result
            assert result.role == "assistant"
            assert len(result.content) == 1
            assert result.content[0].type == "text"
            assert "AI stands for Artificial Intelligence" in result.content[0].text


@pytest.mark.asyncio
async def test_apply_prompt_provider_specific_with_tool_call():
    """Test _apply_prompt_provider_specific method with tool calling."""
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart, Role
    from mcp_agent.mcp.prompt_message_multipart import TextContent
    from mcp_agent.core.request_params import RequestParams
    from mcp.types import CallToolResult
    
    # Create a mock tool definition
    tool = types.SimpleNamespace(
        name="get_weather",
        description="Get the current weather",
        inputSchema={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    )
    
    # Create mock tool response
    tool_response = CallToolResult(
        content=[TextContent(type="text", text="The weather is sunny")],
        isError=False
    )
    
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg, tools=[tool])
    
    # Create multipart messages
    messages = [
        PromptMessageMultipart(
            role=Role.USER,
            content=[TextContent(type="text", text="What's the weather like?")]
        ),
    ]
    
    # Setup response with tool use
    claude_first_response = {
        "content": [{"type": "text", "text": "I need to check the weather."}],
        "tool_use": {
            "id": "tool_123",
            "name": "get_weather",
            "input": {"location": "New York"}
        }
    }
    
    claude_final_response = {
        "content": [{"type": "text", "text": "The weather in New York is sunny."}]
    }
    
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        with patch('mcp_agent.llm.providers.augmented_llm_bedrock.json.loads') as mock_json_loads:
            # Set up the mock responses for two API calls
            mock_json_loads.side_effect = [claude_first_response, claude_final_response]
            
            # Initialize the LLM with mocked dependencies
            llm = BedrockAugmentedLLM(context=ctx)
            
            # Replace _bedrock_client with a mock
            llm._bedrock_client = MagicMock()
            
            # Mock the call_tool method
            llm.call_tool = MagicMock()
            llm.call_tool.return_value = tool_response
            
            # Mock the show_tool methods
            llm.show_tool_call = MagicMock()
            llm.show_tool_result = MagicMock()
            
            # Test the method with parallel tool calls enabled
            request_params = RequestParams(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                parallel_tool_calls=True
            )
            result = await llm._apply_prompt_provider_specific(messages, request_params)
            
            # Verify that call_tool was called
            llm.call_tool.assert_called_once()
            
            # Check the result
            assert result.role == "assistant"
            assert len(result.content) == 1
            assert result.content[0].type == "text"
            assert "The weather in New York is sunny" in result.content[0].text


@pytest.mark.asyncio
async def test_apply_prompt_provider_specific_error_handling():
    """Test error handling in _apply_prompt_provider_specific method."""
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart, Role
    from mcp_agent.mcp.prompt_message_multipart import TextContent
    from mcp_agent.core.request_params import RequestParams
    
    cfg = DummyBedrockConfig()
    ctx = DummyContext(bedrock_cfg=cfg)
    
    # Create multipart messages
    messages = [
        PromptMessageMultipart(
            role=Role.USER,
            content=[TextContent(type="text", text="What is AI?")]
        ),
    ]
    
    # Test with an exception during API call
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3', MagicMock()):
        # Initialize the LLM with mocked dependencies
        llm = BedrockAugmentedLLM(context=ctx)
        
        # Replace _bedrock_client with a mock that raises an exception
        llm._bedrock_client = MagicMock()
        test_error = Exception("Test API error")
        llm.executor.execute = MagicMock()
        llm.executor.execute.return_value = test_error
        
        # Test the method
        request_params = RequestParams(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
        result = await llm._apply_prompt_provider_specific(messages, request_params)
        
        # Check that the error is properly handled
        assert result.role == "assistant"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert "Error calling Bedrock API" in result.content[0].text
        assert "Test API error" in result.content[0].text