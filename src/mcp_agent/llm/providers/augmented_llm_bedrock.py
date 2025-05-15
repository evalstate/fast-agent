"""
Amazon Bedrock implementation of AugmentedLLM.
Implements the AWS Bedrock client integration using the Converse API.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, cast
import json

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from pydantic_core import from_json
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_key_manager import ProviderKeyManager
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.multipart_converter_bedrock import (
    BedrockConverter,
    BedrockMessage,
    ModelFamily,
)
from mcp_agent.llm.providers.sampling_converter_bedrock import BedrockSamplingConverter
from mcp_agent.logging.logger import get_logger

from mcp_agent.mcp.interfaces import AugmentedLLMProtocol, ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Import boto3 as an optional dependency
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

_logger = get_logger(__name__)

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


class BedrockAugmentedLLM(AugmentedLLM[Dict[str, Any], Dict[str, Any]]):
    """
    AWS Bedrock implementation of AugmentedLLM using the Converse API.
    Handles multiple authentication methods including default credentials,
    profiles from AWS credentials file, and direct API keys.
    """

    # Bedrock-specific parameter exclusions
    BEDROCK_EXCLUDE_FIELDS = {
        AugmentedLLM.PARAM_MESSAGES,
        AugmentedLLM.PARAM_MODEL,
        AugmentedLLM.PARAM_MAX_TOKENS,
        AugmentedLLM.PARAM_SYSTEM_PROMPT,
        AugmentedLLM.PARAM_PARALLEL_TOOL_CALLS,
        AugmentedLLM.PARAM_USE_HISTORY,
        AugmentedLLM.PARAM_MAX_ITERATIONS,
        AugmentedLLM.PARAM_TEMPLATE_VARS,
        AugmentedLLM.PARAM_METADATA,
    }

    def __init__(self, provider: Provider = Provider.BEDROCK, *args, **kwargs) -> None:
        # Set type_converter before calling super().__init__
        if "type_converter" not in kwargs:
            kwargs["type_converter"] = BedrockSamplingConverter

        super().__init__(*args, provider=provider, **kwargs)

        # Initialize logger with name if available
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        # Extract config
        context = getattr(self, "context", None)
        config = getattr(context, "config", None) if context else None
        bedrock_cfg = getattr(config, "bedrock", None) if config else None

        # Validate config
        if bedrock_cfg is None:
            raise ProviderKeyError(
                "Missing Bedrock configuration",
                "Bedrock provider requires configuration section 'bedrock' in your config file."
            )

        # Check for boto3
        if boto3 is None:
            raise ProviderKeyError(
                "boto3 not installed",
                "You must install 'boto3' to use Amazon Bedrock. Install with: pip install fast-agent-mcp[bedrock]"
            )

        # Extract configuration with proper priorities
        self.region = getattr(bedrock_cfg, "region", None)
        self.profile = getattr(bedrock_cfg, "profile", None)
        self.endpoint_url = getattr(bedrock_cfg, "endpoint_url", None)
        self.use_default_credentials = getattr(bedrock_cfg, "use_default_credentials", False)
        self.access_key_id = getattr(bedrock_cfg, "access_key_id", None)
        self.secret_access_key = getattr(bedrock_cfg, "secret_access_key", None)
        
        # Extract model parameters
        self.default_params = getattr(bedrock_cfg, "default_params", None) or {}
        self.model_params = getattr(bedrock_cfg, "model_params", None) or {}

        # Validate required parameters
        if not self.region:
            raise ProviderKeyError(
                "Missing AWS region",
                "Field 'region' is required in bedrock config."
            )
            
        # Setup a priority order for authentication methods:
        # 1. Named profile if specified
        # 2. Default credentials if enabled
        # 3. Explicit credentials (access_key_id and secret_access_key)
        
        # If profile is specified, it takes highest priority
        if self.profile:
            self.logger.debug(f"Using AWS credentials from profile: {self.profile}")
            # No validation needed here - boto3 will handle profile errors
        
        # If default credentials are enabled, they're used next (when no profile is specified)
        elif self.use_default_credentials:
            self.logger.debug("Using AWS default credentials chain")
            # No validation needed here - boto3 will attempt to find credentials
        
        # Otherwise, explicit credentials are required
        elif not (self.access_key_id and self.secret_access_key):
            # Check if they might be available from environment variables
            access_key = self._api_key() if not self.access_key_id else self.access_key_id
            secret_key = self._secret_key() if not self.secret_access_key else self.secret_access_key
            
            if not (access_key and secret_key):
                raise ProviderKeyError(
                    "Missing AWS credentials",
                    "When not using default credentials or a named profile, you must provide "
                    "both 'access_key_id' and 'secret_access_key' in your configuration, "
                    "or set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
                )

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Bedrock-specific default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_BEDROCK_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=20,
            use_history=True,
        )

    def _bedrock_client(self):
        """
        Returns a boto3 bedrock-runtime client configured with the appropriate credentials.
        
        This method handles multiple authentication methods:
        1. Default credentials from the AWS SDK (environment, shared credentials file, IAM roles)
        2. Named profile from ~/.aws/credentials
        3. Explicit access key and secret key
        
        Returns:
            A boto3 client instance for bedrock-runtime
            
        Raises:
            ProviderKeyError: If credentials are missing or invalid
        """
        if boto3 is None:
            raise ProviderKeyError(
                "boto3 not installed",
                "You must install 'boto3' to use Amazon Bedrock. Install with: pip install fast-agent-mcp[bedrock]"
            )
        
        try:
            # Common client arguments
            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": self.region,
            }
            
            # Add custom endpoint if specified
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            
            # Log configuration for debugging
            self.logger.debug(f"Creating Bedrock client with region {self.region}")
            
            # Authentication method 1: Named profile
            if self.profile:
                self.logger.debug(f"Using AWS profile: {self.profile}")
                session = boto3.Session(profile_name=self.profile)
                return session.client(**client_kwargs)
            
            # Authentication method 2: Default credentials chain (environment, shared config, EC2 role, etc.)
            if self.use_default_credentials:
                self.logger.debug("Using AWS default credentials chain")
                return boto3.client(**client_kwargs)
            
            # Authentication method 3: Explicit credentials
            access_key = self.access_key_id or self._api_key()
            secret_key = self.secret_access_key or self._secret_key()
            
            if not access_key or not secret_key:
                raise ProviderKeyError(
                    "Missing AWS credentials",
                    "AWS access key and secret key are required when not using default credentials.\n"
                    "Configure them in your bedrock configuration or environment variables."
                )
            
            self.logger.debug("Using explicit AWS credentials")
            return boto3.client(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                **client_kwargs
            )
            
        except NoCredentialsError as e:
            raise ProviderKeyError(
                "No AWS credentials found",
                "No credentials found in the AWS credentials file or environment variables.\n"
                "Please configure AWS credentials through environment variables, "
                "the AWS credentials file, IAM roles, or explicit configuration."
            ) from e
            
        except ClientError as e:
            # Extract helpful information from the error message
            error_msg = str(e)
            if "ExpiredToken" in error_msg:
                raise ProviderKeyError(
                    "AWS credentials expired",
                    "Your AWS credentials have expired. Please refresh your credentials."
                ) from e
            elif "InvalidSignatureException" in error_msg:
                raise ProviderKeyError(
                    "Invalid AWS signature",
                    "Your AWS credentials are invalid or do not have permission to access Bedrock."
                ) from e
            elif "AccessDenied" in error_msg:
                raise ProviderKeyError(
                    "AWS access denied",
                    "Access denied. Your AWS credentials do not have permission to access Bedrock.\n"
                    "Make sure your user/role has bedrock:InvokeModel and other required permissions."
                ) from e
            else:
                raise ProviderKeyError(
                    "AWS Bedrock client error",
                    f"Error creating Bedrock client: {error_msg}"
                ) from e
                
        except Exception as e:
            # Catch-all for other unexpected errors
            raise ProviderKeyError(
                "Bedrock client creation failed", 
                f"Unexpected error: {str(e)}"
            ) from e

    def _secret_key(self) -> str:
        """Get the AWS secret access key from settings or environment variables."""
        return ProviderKeyManager.get_aws_secret_key(self.context.config)

    def _prepare_api_request(
        self,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]] | None,
        request_params: RequestParams,
    ) -> Dict[str, Any]:
        """
        Prepare the request for the Bedrock API based on model family.
        
        Args:
            messages: List of Bedrock-format messages
            available_tools: List of available tools (optional)
            request_params: Request parameters
            
        Returns:
            Complete request object for the Bedrock API
        """
        model = request_params.model
        
        # Ensure model ID has proper prefix for Claude models
        if ("claude" in model.lower() and not model.startswith("us.") and 
            not model.startswith("eu.") and "anthropic" in model.lower()):
            # Add the US prefix (default region)
            self.logger.debug(f"Adding 'us.' prefix to Claude model ID: {model}")
            model = f"us.{model}"
            request_params.model = model
        
        # Detect the model family
        model_family = BedrockConverter.detect_model_family(model)
        
        # Determine model-specific parameters by merging default and model-specific parameters
        model_specific_params = self.default_params.copy()
        if model in self.model_params:
            model_specific_params.update(self.model_params[model])
        
        # Choose request preparation method based on model family
        if model_family == ModelFamily.CLAUDE:
            return self._prepare_claude_request(
                messages, available_tools, request_params, model_specific_params
            )
        elif model_family == ModelFamily.NOVA:
            return self._prepare_nova_request(
                messages, available_tools, request_params, model_specific_params
            ) 
        elif model_family == ModelFamily.META:
            return self._prepare_meta_request(
                messages, available_tools, request_params, model_specific_params
            )
        else:
            # Default to Claude format if unknown
            self.logger.warning(f"Unknown model family for {model}, using Claude format")
            return self._prepare_claude_request(
                messages, available_tools, request_params, model_specific_params
            )
    
    def _prepare_claude_request(
        self,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]] | None,
        request_params: RequestParams,
        model_specific_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare the request for Claude models on Bedrock.
        """
        # Create the base request structure for Claude on Bedrock
        bedrock_request = {
            "anthropic_version": "bedrock-2023-05-31",  # Required by Bedrock API
            "max_tokens": request_params.maxTokens or 2048,
            "messages": messages,
        }
        
        # Add basic parameters
        if hasattr(request_params, "temperature") and request_params.temperature is not None:
            bedrock_request["temperature"] = request_params.temperature
            
        if hasattr(request_params, "top_p") and request_params.top_p is not None:
            bedrock_request["top_p"] = request_params.top_p
            
        if hasattr(request_params, "top_k") and request_params.top_k is not None:
            bedrock_request["top_k"] = request_params.top_k
            
        # Add stop sequences if provided
        if hasattr(request_params, "stopSequences") and request_params.stopSequences:
            bedrock_request["stop_sequences"] = request_params.stopSequences
            
        # Add additional parameters from request_params
        additional_params = self.prepare_provider_arguments(
            base_args={},
            request_params=request_params,
            exclude_fields=self.BEDROCK_EXCLUDE_FIELDS,
        )
        
        # Add additional params to the request
        for key, value in additional_params.items():
            if value is not None and key not in bedrock_request:
                bedrock_request[key] = value
        
        # Add model-specific parameters
        for key, value in model_specific_params.items():
            if value is not None and key not in bedrock_request:
                bedrock_request[key] = value
        
        # Add tool calling if available
        if available_tools and request_params.parallel_tool_calls:
            tools = []
            for tool in available_tools:
                if isinstance(tool, Dict) and "function" in tool:
                    tool_def = tool["function"]
                    tools.append({
                        "name": tool_def.get("name", ""),
                        "description": tool_def.get("description", ""),
                        "input_schema": tool_def.get("parameters", {})
                    })
            
            if tools:
                bedrock_request["tools"] = tools
                
                # Enable tool use
                bedrock_request["tool_choice"] = "auto"
        
        # Log the request for debugging (with sensitive data redacted)
        debug_request = bedrock_request.copy()
        if "messages" in debug_request:
            debug_request["messages"] = f"[{len(debug_request['messages'])} messages]"
        self.logger.debug(f"Claude request prepared: {debug_request}")
        
        return bedrock_request
    
    def _prepare_nova_request(
        self,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]] | None,
        request_params: RequestParams,
        model_specific_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare the request for Amazon Nova models on Bedrock.
        
        This method supports two formats:
        1. The newer Converse API format for Nova Pro, supporting structured messages
        2. The original InvokeModel format with inputText for older Nova models
        
        Nova Pro (us.amazon.nova-pro-v1:0) can use the Converse API with structured messages
        similar to Claude models.
        """
        # Check if this is Nova Pro model which supports the Converse API
        model = request_params.model
        is_nova_pro = "nova-pro" in model.lower()
        
        if is_nova_pro:
            # For Nova Pro, use a format similar to Claude with structured messages
            self.logger.debug(f"Using Converse API format for Nova Pro model: {model}")
            
            # Create base request similar to Claude format
            bedrock_request = {
                "messages": messages,
                "max_tokens": request_params.maxTokens or 8192,  # Nova Pro supports up to 8192 tokens
            }
            
            # Add basic parameters
            if hasattr(request_params, "temperature") and request_params.temperature is not None:
                bedrock_request["temperature"] = request_params.temperature
                
            if hasattr(request_params, "top_p") and request_params.top_p is not None:
                bedrock_request["top_p"] = request_params.top_p
                
            if hasattr(request_params, "top_k") and request_params.top_k is not None:
                bedrock_request["top_k"] = request_params.top_k
                
            # Add stop sequences if provided
            if hasattr(request_params, "stopSequences") and request_params.stopSequences:
                bedrock_request["stop_sequences"] = request_params.stopSequences
            
            # Add additional parameters from request_params
            additional_params = self.prepare_provider_arguments(
                base_args={},
                request_params=request_params,
                exclude_fields=self.BEDROCK_EXCLUDE_FIELDS,
            )
            
            # Add additional params to the request
            for key, value in additional_params.items():
                if value is not None and key not in bedrock_request:
                    bedrock_request[key] = value
            
            # Add model-specific parameters
            for key, value in model_specific_params.items():
                if value is not None and key not in bedrock_request:
                    bedrock_request[key] = value
            
            # Add tool calling if available
            if available_tools and request_params.parallel_tool_calls:
                tools = []
                for tool in available_tools:
                    if isinstance(tool, Dict) and "function" in tool:
                        tool_def = tool["function"]
                        tools.append({
                            "name": tool_def.get("name", ""),
                            "description": tool_def.get("description", ""),
                            "input_schema": tool_def.get("parameters", {})
                        })
                
                if tools:
                    bedrock_request["tools"] = tools
                    
                    # Enable tool use
                    bedrock_request["tool_choice"] = "auto"
            
            # Log the request for debugging (with sensitive data redacted)
            debug_request = bedrock_request.copy()
            if "messages" in debug_request:
                debug_request["messages"] = f"[{len(debug_request['messages'])} messages]"
            self.logger.debug(f"Nova Pro request prepared: {debug_request}")
            
            return bedrock_request
        
        else:
            # For older Nova models, use the traditional InvokeModel format
            self.logger.debug(f"Using InvokeModel API format for Nova model: {model}")
            
            # Get the last user message
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                self.logger.warning("No user messages found for Nova request")
                input_text = ""
            else:
                # Use the last user message
                last_user_msg = user_messages[-1]
                # Get text from message - Nova expects plain text
                if "text" in last_user_msg:
                    input_text = last_user_msg["text"]
                else:
                    # Try to extract text from content
                    text_parts = []
                    if "content" in last_user_msg and isinstance(last_user_msg["content"], list):
                        for item in last_user_msg["content"]:
                            if isinstance(item, dict) and "text" in item:
                                text_parts.append(item["text"])
                    input_text = "\n".join(text_parts)
            
            # Combine with system message if available
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            if system_messages:
                system_msg = system_messages[0]
                system_text = ""
                
                if "text" in system_msg:
                    system_text = system_msg["text"]
                elif "content" in system_msg and isinstance(system_msg["content"], list):
                    text_parts = []
                    for item in system_msg["content"]:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                    system_text = "\n".join(text_parts)
                
                if system_text:
                    input_text = f"<<SYS>>\n{system_text}\n<</SYS>>\n\n{input_text}"
            
            # Combine with any assistant message history if needed
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            if assistant_messages and "text" in assistant_messages[-1]:
                # Some Nova models support chat history in a special format
                conversation_history = []
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        continue  # Handled separately
                        
                    role = msg.get("role", "")
                    content = ""
                    
                    if "text" in msg:
                        content = msg["text"]
                    elif "content" in msg and isinstance(msg["content"], list):
                        text_parts = []
                        for item in msg["content"]:
                            if isinstance(item, dict) and "text" in item:
                                text_parts.append(item["text"])
                        content = "\n".join(text_parts)
                        
                    if role and content:
                        conversation_history.append(f"{role}: {content}")
                
                # Set the combined history as input
                input_text = "\n".join(conversation_history)
            
            # Create the base request for Nova with traditional format
            bedrock_request = {
                "inputText": input_text,
                "textGenerationConfig": {
                    "maxTokenCount": request_params.maxTokens or 2048,
                    "temperature": request_params.temperature or 0.7,
                    "topP": request_params.top_p or 0.9,
                    "topK": request_params.top_k or 50,
                }
            }
            
            # Add additional parameters from request_params
            additional_params = self.prepare_provider_arguments(
                base_args={},
                request_params=request_params,
                exclude_fields=self.BEDROCK_EXCLUDE_FIELDS,
            )
            
            # Add any additional params to textGenerationConfig
            for key, value in additional_params.items():
                if value is not None and key not in bedrock_request["textGenerationConfig"]:
                    bedrock_request["textGenerationConfig"][key] = value
            
            # Add model-specific parameters
            for key, value in model_specific_params.items():
                if value is not None:
                    if key in ["stopSequences", "stop"]:
                        bedrock_request["textGenerationConfig"]["stopSequences"] = value
                    elif key not in bedrock_request["textGenerationConfig"]:
                        bedrock_request["textGenerationConfig"][key] = value
            
            # Log the request for debugging
            self.logger.debug(f"Nova request prepared: {bedrock_request}")
            
            return bedrock_request
    
    def _prepare_meta_request(
        self,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]] | None,
        request_params: RequestParams,
        model_specific_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare the request for Meta Llama models on Bedrock.
        """
        # Meta Llama models on Bedrock typically use a format like:
        # {"prompt": "...", "max_gen_length": 512, ...}
        
        # For Meta Llama, construct a formatted prompt from messages
        prompt_parts = []
        
        # First, look for system messages
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        if system_messages and "content" in system_messages[0]:
            prompt_parts.append(f"<system>\n{system_messages[0]['content']}\n</system>")
        
        # Then add conversation history
        for msg in messages:
            if msg.get("role") == "system":
                continue  # Already handled
            
            role = msg.get("role", "")
            if role and "content" in msg:
                content = msg["content"]
                # Handle different content formats
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                    content = "\n".join(text_parts)
                
                prompt_parts.append(f"<{role}>\n{content}\n</{role}>")
        
        # Create the final prompt
        prompt = "\n".join(prompt_parts)
        
        # Create the base request for Meta Llama
        bedrock_request = {
            "prompt": prompt,
            "max_gen_len": request_params.maxTokens or 2048,
            "temperature": request_params.temperature or 0.7,
            "top_p": request_params.top_p or 0.9,
        }
        
        # Add additional parameters from request_params
        additional_params = self.prepare_provider_arguments(
            base_args={},
            request_params=request_params,
            exclude_fields=self.BEDROCK_EXCLUDE_FIELDS,
        )
        
        # Add additional params to the request
        for key, value in additional_params.items():
            if value is not None and key not in bedrock_request:
                bedrock_request[key] = value
        
        # Add model-specific parameters
        for key, value in model_specific_params.items():
            if value is not None and key not in bedrock_request:
                bedrock_request[key] = value
        
        # Log the request for debugging
        debug_request = bedrock_request.copy()
        debug_request["prompt"] = f"[{len(debug_request['prompt'])} chars]"
        self.logger.debug(f"Meta Llama request prepared: {debug_request}")
        
        return bedrock_request

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """
        Provider-specific implementation for Bedrock.
        
        Args:
            multipart_messages: List of multipart messages to send to the model
            request_params: Request parameters
            is_template: Whether this is a template call
            
        Returns:
            The model's response as a PromptMessageMultipart
        """
        # Get request parameters with defaults
        request_params = self.get_request_params(request_params=request_params)
        
        # Ensure model ID has proper prefix for Claude models
        model = request_params.model or self.default_request_params.model
        
        # Check if this is a Claude model without a regional prefix
        if ("claude" in model.lower() and not model.startswith("us.") and 
            not model.startswith("eu.") and "anthropic" in model.lower()):
            # Add the US prefix (default region)
            self.logger.debug(f"Adding 'us.' prefix to Claude model ID: {model}")
            model = f"us.{model}"
            request_params.model = model
        
        # Determine the model family
        model_family = BedrockConverter.detect_model_family(model)
        
        # Convert multipart messages to Bedrock format based on model family
        bedrock_messages = []
        system_found = False
        
        for message in multipart_messages:
            # Check for system messages
            if message.role == "system":
                system_found = True
            
            # Convert message to Bedrock format for the specific model family
            bedrock_message = BedrockConverter.convert_to_bedrock(message, model_family)
            bedrock_messages.append(bedrock_message)
        
        # Add system message if none found and we have an instruction
        if not system_found and self.instruction:
            # Convert system prompt based on model family
            if model_family == ModelFamily.CLAUDE:
                system_message = {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"<admin>\n{self.instruction}\n</admin>"
                    }]
                }
            elif model_family == ModelFamily.NOVA:
                system_message = {
                    "text": f"<<SYS>>\n{self.instruction}\n<</SYS>>"
                }
            elif model_family == ModelFamily.META:
                system_message = {
                    "role": "system",
                    "content": self.instruction
                }
            else:
                # Default to Claude format
                system_message = {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"<admin>\n{self.instruction}\n</admin>"
                    }]
                }
            
            # Insert at the beginning
            bedrock_messages.insert(0, system_message)
        
        # Get available tools
        available_tools = None
        if request_params.parallel_tool_calls:
            response = await self.aggregator.list_tools()
            if response.tools:
                available_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description if tool.description else "",
                            "parameters": self.adjust_schema(tool.inputSchema),
                        },
                    }
                    for tool in response.tools
                ]
        
        # Process responses over multiple iterations if necessary
        responses = []
        
        for iteration in range(request_params.max_iterations or 1):
            # Prepare the API request
            bedrock_request = self._prepare_api_request(
                bedrock_messages, available_tools, request_params
            )
            
            # Log chat progress
            self._log_chat_progress(self.chat_turn(), model=request_params.model)
            
            try:
                # Get Bedrock client
                client = self._bedrock_client()
                
                # Extract the model ID from the request parameters
                model_id = request_params.model
                
                # Ensure model ID has proper prefix for Claude models
                if ("claude" in model_id.lower() and not model_id.startswith("us.") and 
                    not model_id.startswith("eu.") and "anthropic" in model_id.lower()):
                    # Add the US prefix (default region)
                    self.logger.debug(f"Adding 'us.' prefix to Claude model ID: {model_id}")
                    model_id = f"us.{model_id}"
                    # Update the request parameters for future use
                    request_params.model = model_id
                
                # Bedrock requires the request body to be a JSON string
                request_body = json.dumps(bedrock_request)
                
                # Log that we're about to make the API call
                self.logger.debug(f"Invoking Bedrock model: {model_id}")
                
                # Make the API call
                response = await self.executor.execute(
                    client.invoke_model,
                    modelId=model_id,
                    body=request_body,
                    contentType="application/json",
                    accept="application/json",
                )
                
                # Parse the response
                if isinstance(response, Exception):
                    self.logger.error(f"Bedrock API error: {str(response)}")
                    # Return error message
                    return PromptMessageMultipart(
                        role="assistant",
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error calling Bedrock API: {str(response)}"
                            )
                        ]
                    )
                
                # Response is a boto3 response object with 'body'
                raw_response_body = response[0]['body'].read()
                response_json = json.loads(raw_response_body)
                
                # Log response for debugging
                self.logger.debug(f"Bedrock response received: {response_json.keys()}")
                
                # Parse the response based on model family
                if model_family == ModelFamily.CLAUDE:
                    # Handle Claude response format
                    if "content" in response_json:
                        text_content = []
                        
                        # Extract text from response
                        for content_item in response_json["content"]:
                            if content_item.get("type") == "text":
                                text_content.append(content_item.get("text", ""))
                        
                        # Join text content
                        text = "\n".join(text_content)
                        responses.append(TextContent(type="text", text=text))
                    
                    # Check for tool calls in the response
                    if "tool_use" in response_json:
                        # Handle tool use response
                        tool_calls = response_json["tool_use"]
                        tool_use_id = tool_calls.get("id")
                        tool_name = tool_calls.get("name")
                        tool_input = tool_calls.get("input", {})
                        
                        # Show the tool call
                        self.show_tool_call(
                            available_tools, tool_name, json.dumps(tool_input)
                        )
                        
                        # Create the tool request
                        tool_call_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(
                                name=tool_name,
                                arguments=tool_input,
                            ),
                        )
                        
                        # Call the tool
                        result = await self.call_tool(tool_call_request, tool_use_id)
                        self.show_tool_result(result)
                        
                        # Convert the tool result to a Bedrock tool result block
                        tool_result = BedrockConverter.convert_tool_result_to_bedrock(
                            result, tool_use_id, model_family
                        )
                        
                        # Add the tool result to the messages
                        bedrock_messages.append({
                            "role": "user",
                            "content": [tool_result]
                        })
                        
                        # Continue to the next iteration
                        continue
                
                elif model_family == ModelFamily.NOVA:
                    # Check if this is Nova Pro model (supports Converse API with structured responses)
                    is_nova_pro = "nova-pro" in model_id.lower()
                    
                    if is_nova_pro and "content" in response_json:
                        # Handle Nova Pro response similar to Claude format
                        text_content = []
                        
                        # Extract text from response - should be similar to Claude format
                        for content_item in response_json.get("content", []):
                            if content_item.get("type") == "text":
                                text_content.append(content_item.get("text", ""))
                        
                        # Join text content
                        text = "\n".join(text_content)
                        responses.append(TextContent(type="text", text=text))
                        
                        # Log response parsing approach
                        self.logger.debug(f"Parsed Nova Pro response using Converse API format")
                        
                        # Check for tool calls in the response (if Nova Pro supports tools)
                        if "tool_use" in response_json:
                            # Handle tool use response - similar to Claude
                            tool_calls = response_json["tool_use"]
                            tool_use_id = tool_calls.get("id")
                            tool_name = tool_calls.get("name")
                            tool_input = tool_calls.get("input", {})
                            
                            # Show the tool call
                            self.show_tool_call(
                                available_tools, tool_name, json.dumps(tool_input)
                            )
                            
                            # Create the tool request
                            tool_call_request = CallToolRequest(
                                method="tools/call",
                                params=CallToolRequestParams(
                                    name=tool_name,
                                    arguments=tool_input,
                                ),
                            )
                            
                            # Call the tool
                            result = await self.call_tool(tool_call_request, tool_use_id)
                            self.show_tool_result(result)
                            
                            # Convert the tool result to a Bedrock tool result block
                            tool_result = BedrockConverter.convert_tool_result_to_bedrock(
                                result, tool_use_id, model_family
                            )
                            
                            # Add the tool result to the messages
                            bedrock_messages.append({
                                "role": "user",
                                "content": [tool_result]
                            })
                            
                            # Continue to the next iteration
                            continue
                    
                    elif "results" in response_json and len(response_json["results"]) > 0:
                        # Handle traditional Nova response format (InvokeModel API)
                        result = response_json["results"][0]
                        if "outputText" in result:
                            text = result["outputText"]
                            responses.append(TextContent(type="text", text=text))
                            
                            # Log response parsing approach
                            self.logger.debug(f"Parsed Nova response using InvokeModel API format")
                    
                    else:
                        # Fallback: attempt to find text content in other response fields
                        self.logger.warning(f"Unexpected Nova response format: {response_json.keys()}")
                        
                        # Try to extract text from various possible fields
                        for key in ["output", "text", "message", "response", "generated_text"]:
                            if key in response_json:
                                text = response_json[key]
                                if isinstance(text, str):
                                    responses.append(TextContent(type="text", text=text))
                                    break
                
                elif model_family == ModelFamily.META:
                    # Handle Meta Llama response format
                    if "generation" in response_json:
                        # Meta Llama returns a "generation" field with the response
                        text = response_json["generation"]
                        responses.append(TextContent(type="text", text=text))
                
                else:
                    # Generic handling for unknown model families
                    # Look for common response patterns
                    for key in ["content", "text", "output", "generated_text", "response"]:
                        if key in response_json:
                            content = response_json[key]
                            if isinstance(content, list):
                                # Handle list of content items
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        text_parts.append(item["text"])
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                text = "\n".join(text_parts)
                            else:
                                # Handle string content
                                text = str(content)
                            
                            responses.append(TextContent(type="text", text=text))
                            break
                
                # Create the assistant message from the response
                assistant_message = PromptMessageMultipart(
                    role="assistant", 
                    content=responses if responses else [
                        TextContent(type="text", text="[No content in response]")
                    ]
                )
                
                # If we reach here, no more iterations needed
                break
                
            except Exception as e:
                self.logger.error(f"Error in Bedrock API call: {str(e)}")
                return PromptMessageMultipart(
                    role="assistant",
                    content=[
                        TextContent(
                            type="text",
                            text=f"Error calling Bedrock API: {str(e)}"
                        )
                    ]
                )
        
        # Log chat finished
        self._log_chat_finished(model=request_params.model)
        
        # Create message from final response
        return PromptMessageMultipart(
            role="assistant", 
            content=responses if responses else [
                TextContent(type="text", text="[No content in response]")
            ]
        )

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Provider-specific implementation for structured outputs with Bedrock.
        
        This is a placeholder implementation that will be expanded in Task 3.
        """
        # This will be implemented with appropriate message converters and structure handling
        # For now we'll reuse the base implementation
        
        return await super()._apply_prompt_provider_specific_structured(
            multipart_messages, model, request_params
        )