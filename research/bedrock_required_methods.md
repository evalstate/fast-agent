# Required Methods for Amazon Bedrock Provider Implementation

Based on the analysis of existing provider implementations in fast-agent, below are the key methods that need to be implemented for the Amazon Bedrock provider.

## Core Provider Class: `BedrockAugmentedLLM`

The `BedrockAugmentedLLM` class should extend the base `AugmentedLLM` class and implement the following key methods:

### 1. Constructor and Initialization

```python
def __init__(self, provider: Provider = Provider.BEDROCK, *args, **kwargs) -> None:
    """
    Initialize the Bedrock provider with appropriate settings.
    """
    # Initialize type converter for sampling if needed
    if "type_converter" not in kwargs:
        kwargs["type_converter"] = BedrockSamplingConverter
    
    # Call parent constructor
    super().__init__(provider=provider, *args, **kwargs)
    
    # Configure AWS credentials and settings from config
    context = getattr(self, "context", None)
    config = getattr(context, "config", None) if context else None
    bedrock_cfg = getattr(config, "bedrock", None) if config else None
    
    # Handle configuration validation
    if bedrock_cfg is None:
        raise ProviderKeyError(
            "Missing Bedrock configuration",
            "Bedrock provider requires configuration section 'bedrock' in your config file."
        )
    
    # Extract configuration parameters
    self.aws_region = getattr(bedrock_cfg, "region", None)
    self.aws_profile = getattr(bedrock_cfg, "profile", None)
    self.endpoint_url = getattr(bedrock_cfg, "endpoint_url", None)
    self.use_default_credentials = getattr(bedrock_cfg, "use_default_credentials", False)
    
    # Validate required configuration
    if not self.aws_region:
        raise ProviderKeyError(
            "Missing AWS region",
            "AWS Bedrock requires 'region' in your 'bedrock' config section."
        )
```

### 2. Default Parameters Configuration

```python
def _initialize_default_params(self, kwargs: dict) -> RequestParams:
    """Initialize Bedrock-specific default parameters"""
    chosen_model = kwargs.get("model", DEFAULT_BEDROCK_MODEL)
    
    return RequestParams(
        model=chosen_model,
        systemPrompt=self.instruction,
        parallel_tool_calls=True,  # For compatible models
        max_iterations=20,
        use_history=True,
    )
```

### 3. Client Creation

```python
def _bedrock_client(self):
    """
    Returns a boto3 bedrock-runtime client.
    """
    try:
        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": self.aws_region,
        }
        
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        
        if self.use_default_credentials:
            # Use AWS credential provider chain
            if self.aws_profile:
                session = boto3.Session(profile_name=self.aws_profile)
                client = session.client(**client_kwargs)
            else:
                client = boto3.client(**client_kwargs)
        else:
            # Use explicit credentials
            client = boto3.client(
                aws_access_key_id=self._api_key(),
                aws_secret_access_key=self._secret_key(),
                **client_kwargs
            )
            
        return client
    except Exception as e:
        raise ProviderKeyError("Bedrock client creation failed", str(e))
```

### 4. Completion Method

```python
async def _bedrock_completion(
    self,
    message,
    request_params: RequestParams | None = None,
) -> List[TextContent | ImageContent | EmbeddedResource]:
    """
    Process a query using Amazon Bedrock.
    """
    request_params = self.get_request_params(request_params=request_params)
    responses: List[TextContent | ImageContent | EmbeddedResource] = []
    
    # Process message history and prepare converse request
    messages = []
    system_prompt = self.instruction or request_params.systemPrompt
    
    if system_prompt:
        # System prompts are handled differently by Converse API
        system = system_prompt
    else:
        system = None
    
    # Add history messages
    messages.extend(self.history.get(include_completion_history=request_params.use_history))
    messages.append(message)
    
    # Get available tools
    response = await self.aggregator.list_tools()
    tools = [
        {
            "name": tool.name,
            "description": tool.description if tool.description else "",
            "schema": self.adjust_schema(tool.inputSchema),
        }
        for tool in response.tools
    ] if response.tools else None
    
    # Convert message history into Bedrock Converse format
    # Create inference configuration
    inference_config = {
        "temperature": request_params.temperature,
        "topP": request_params.topP,
        "maxTokens": request_params.maxTokens,
    }
    
    # Remove None values
    inference_config = {k: v for k, v in inference_config.items() if v is not None}
    
    # Create tool config if tools are available
    tool_config = {"tools": tools} if tools else None
    
    # Log progress
    self._log_chat_progress(self.chat_turn(), model=request_params.model)
    
    try:
        # Make the Converse API call
        client = self._bedrock_client()
        response = await self.executor.execute(
            client.converse,
            modelId=request_params.model,
            messages=messages,
            system=system,
            inferenceConfig=inference_config,
            toolConfig=tool_config
        )
        
        # Process the response
        if response.get("output"):
            response_text = response["output"].get("text", "")
            responses.append(TextContent(type="text", text=response_text))
            
            # Show response in the console
            await self.show_assistant_message(response_text, "")
            
            # Add to message history
            messages.append({"role": "assistant", "content": response_text})
        
        # Handle tool calls if present in the response
        tool_calls = response.get("output", {}).get("toolUse", [])
        if tool_calls:
            tool_results = []
            
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                args = tool_call.get("input", {})
                tool_id = tool_call.get("id", "")
                
                # Show the tool call in the console
                self.show_tool_call(tools, tool_name, json.dumps(args))
                
                # Execute the tool
                tool_call_request = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name=tool_name,
                        arguments=args
                    ),
                )
                
                result = await self.call_tool(tool_call_request, tool_id)
                self.show_tool_result(str(result))
                
                # Add to results
                tool_results.append((tool_id, result))
                responses.extend(result.content)
            
            # Add tool results to message history
            for tool_id, result in tool_results:
                tool_message = {"role": "tool", "toolCallId": tool_id, "content": str(result)}
                messages.append(tool_message)
        
        # Log completion
        self._log_chat_finished(model=request_params.model)
        
        # Update conversation history if enabled
        if request_params.use_history:
            # Calculate new conversation messages
            prompt_messages = self.history.get(include_completion_history=False)
            new_messages = messages[len(prompt_messages):]
            
            if system_prompt:
                new_messages = new_messages[1:]
            
            self.history.set(new_messages)
            
        return responses
        
    except Exception as e:
        # Handle errors
        error_msg = f"Bedrock API error: {str(e)}"
        self.logger.error(error_msg)
        
        # Show error to user
        await self.show_assistant_message(
            Text(f"Error: {error_msg}", style="bold red"),
            ""
        )
        
        raise ProviderKeyError("Bedrock API error", error_msg)
```

### 5. Apply Prompt Provider Specific

```python
async def _apply_prompt_provider_specific(
    self,
    multipart_messages: List["PromptMessageMultipart"],
    request_params: RequestParams | None = None,
    is_template: bool = False,
) -> PromptMessageMultipart:
    """
    Apply the prompt to the provider-specific implementation
    """
    last_message = multipart_messages[-1]
    
    # Add messages to history
    messages_to_add = multipart_messages[:-1] if last_message.role == "user" else multipart_messages
    
    # Convert to provider-specific format and add to history
    converted = []
    for msg in messages_to_add:
        # Convert based on the model provider
        if msg.role == "user":
            converted.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            converted.append({"role": "assistant", "content": msg.content})
        elif msg.role == "system":
            converted.append({"role": "system", "content": msg.content})
    
    self.history.extend(converted, is_prompt=is_template)
    
    if last_message.role == "assistant":
        return last_message
    
    # For user messages: Process with Bedrock
    bedrock_message = {"role": "user", "content": last_message.content}
    responses = await self._bedrock_completion(
        bedrock_message,
        request_params,
    )
    
    return Prompt.assistant(*responses)
```

### 6. Tool Handling Methods

```python
async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
    """Hook called before a tool is called"""
    return request

async def post_tool_call(
    self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
):
    """Hook called after a tool call with the result"""
    return result

def adjust_schema(self, input_schema: Dict) -> Dict:
    """
    Adjust JSON schema for Bedrock model compatibility
    """
    if "properties" in input_schema:
        return input_schema
        
    result = input_schema.copy()
    result["properties"] = {}
    return result
```

### 7. API Key Management

```python
def _api_key(self) -> str:
    """Get the AWS access key ID"""
    if self.context and self.context.config and self.context.config.bedrock:
        return getattr(self.context.config.bedrock, "access_key_id", None) or ProviderKeyManager.get_api_key("bedrock", self.context.config)
    return ProviderKeyManager.get_api_key("bedrock", {})

def _secret_key(self) -> str:
    """Get the AWS secret access key"""
    if self.context and self.context.config and self.context.config.bedrock:
        return getattr(self.context.config.bedrock, "secret_access_key", None) or os.environ.get("AWS_SECRET_ACCESS_KEY")
    return os.environ.get("AWS_SECRET_ACCESS_KEY")
```

## Required Provider Registration and Type Updates

### 1. Add BEDROCK to Provider Enum in provider_types.py

```python
class Provider(Enum):
    """Supported LLM providers"""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    FAST_AGENT = "fast-agent"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    GENERIC = "generic"
    OPENROUTER = "openrouter"
    TENSORZERO = "tensorzero"  
    AZURE = "azure"
    BEDROCK = "bedrock"  # Add this line
```

### 2. Add Bedrock Settings Model in config.py

```python
class BedrockSettings(BaseModel):
    """
    Settings for using Amazon Bedrock in the fast-agent application.
    """
    # AWS credentials (optional - can use credential chain)
    access_key_id: str | None = None
    secret_access_key: str | None = None
    
    # Required AWS region
    region: str | None = None
    
    # Optional settings
    profile: str | None = None  # AWS profile name
    endpoint_url: str | None = None  # Custom endpoint URL
    use_default_credentials: bool = False  # Use AWS credential provider chain
    
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
```

### 3. Add Bedrock to Settings class in config.py

```python
class Settings(BaseSettings):
    # Existing settings...
    
    bedrock: BedrockSettings | None = None
    """Settings for using Amazon Bedrock in the fast-agent application"""
```

### 4. Add Bedrock to Model Factory in model_factory.py

```python
# Update imports
from mcp_agent.llm.providers.augmented_llm_bedrock import BedrockAugmentedLLM

# Update LLMClass type alias
LLMClass = Union[
    # Existing types...
    Type[BedrockAugmentedLLM],
]

# Update DEFAULT_PROVIDERS dictionary
DEFAULT_PROVIDERS = {
    # Existing providers...
    "bedrock.anthropic.claude-3-sonnet-20240229-v1:0": Provider.BEDROCK,
    "bedrock.anthropic.claude-3-haiku-20240307-v1:0": Provider.BEDROCK,
    # Add other Bedrock models...
}

# Update MODEL_ALIASES dictionary
MODEL_ALIASES = {
    # Existing aliases...
    "bedrock.claude3": "bedrock.anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock.claude3-sonnet": "bedrock.anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock.claude3-haiku": "bedrock.anthropic.claude-3-haiku-20240307-v1:0",
}

# Update PROVIDER_CLASSES dictionary
PROVIDER_CLASSES: Dict[Provider, LLMClass] = {
    # Existing classes...
    Provider.BEDROCK: BedrockAugmentedLLM,
}
```

### 5. Update Provider Key Manager

```python
# Add to PROVIDER_ENVIRONMENT_MAP
PROVIDER_ENVIRONMENT_MAP: Dict[str, str] = {
    # Existing mappings...
    "bedrock": "AWS_ACCESS_KEY_ID",
}
```

## Required Multipart and Sampling Converters

### 1. Create multipart_converter_bedrock.py

```python
from typing import Any, Dict, List, Optional, Union

from mcp_agent.llm.providers.multipart_converter_openai import OpenAIConverter
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

class BedrockConverter(OpenAIConverter):
    """
    Converter for Bedrock message formats.
    Extends the OpenAI converter since the formats are similar.
    """
    
    @classmethod
    def convert_function_results_to_bedrock(cls, tool_results):
        """
        Convert function results to Bedrock tool message format.
        
        Args:
            tool_results: List of (tool_id, result) tuples
            
        Returns:
            List of Bedrock tool messages
        """
        # Similar to OpenAI but with Bedrock-specific format
        messages = []
        for tool_id, result in tool_results:
            content = str(result)
            messages.append({
                "role": "tool",
                "toolCallId": tool_id,
                "content": content
            })
        return messages
```

### 2. Create sampling_converter_bedrock.py

```python
from typing import Any, Dict, List, Optional, Union

from mcp_agent.llm.sampling_format_converter import BaseSamplingFormatConverter

class BedrockSamplingConverter(BaseSamplingFormatConverter):
    """
    Converter for Bedrock sampling formats.
    """
    
    @classmethod
    def convert_from_provider(cls, provider_data: Any) -> Dict[str, Any]:
        """
        Convert Bedrock sampling format to a common format.
        
        Args:
            provider_data: The raw sampling data from Bedrock
            
        Returns:
            Dictionary with standardized sampling data
        """
        return {
            "samples": provider_data.get("samples", []),
            "finish_reason": provider_data.get("stopReason", "unknown")
        }
    
    @classmethod
    def convert_to_provider(cls, sampling_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert common sampling format to Bedrock format.
        
        Args:
            sampling_request: The common sampling request
            
        Returns:
            Dictionary with Bedrock-specific sampling request
        """
        return {
            "temperature": sampling_request.get("temperature", 0.7),
            "topP": sampling_request.get("top_p", 0.9),
            "maxTokens": sampling_request.get("max_tokens", 1000),
            "n": sampling_request.get("n", 1)
        }
```

## Conclusion

These are the core methods and components that need to be implemented for the Amazon Bedrock provider in fast-agent. The implementation should follow the patterns established by other providers, especially the Azure provider which demonstrates how to handle cloud-specific authentication methods.

The initial focus should be on supporting Claude models through Bedrock using the Converse API, with additional model support added incrementally once the base implementation is working correctly.