# Amazon Bedrock Integration with Fast-Agent

This document compares the Amazon Bedrock API with existing provider implementations in fast-agent and outlines the integration approach.

## Provider Implementation Pattern

Fast-agent follows a consistent pattern for implementing LLM providers:

1. **Provider Enum**: Add provider type to `Provider` enum in `provider_types.py`
2. **Settings Model**: Define provider-specific settings in `config.py`
3. **Provider Class**: Implement provider-specific class extending `AugmentedLLM`
4. **Model Factory**: Register the provider and models in `model_factory.py`
5. **Authentication**: Handle provider-specific authentication in `provider_key_manager.py`

## Comparison with Azure Implementation

The Azure implementation in `augmented_llm_azure.py` is a good reference for the Bedrock implementation, as it follows similar patterns:

### 1. Inheritance Strategy
- Azure extends the OpenAI implementation: `AzureOpenAIAugmentedLLM(OpenAIAugmentedLLM)`
- Bedrock should likely be a direct implementation of `AugmentedLLM`

### 2. Configuration Structure
- Azure uses `AzureSettings` with fields for API keys, resource names, and endpoints
- Bedrock would need a similar `BedrockSettings` model with fields for AWS credentials, region, etc.

### 3. Authentication Patterns
- Azure supports both direct API key authentication and managed identity (DefaultAzureCredential)
- Bedrock should support:
  - Direct AWS credentials (access key, secret key)
  - AWS credential provider chain (similar to DefaultAzureCredential)
  - Region specification

### 4. Client Creation
- Azure creates an `AzureOpenAI` client from the OpenAI SDK
- Bedrock would create a boto3 `bedrock-runtime` client

## Key Differences

### 1. SDK Differences
- Azure uses the OpenAI SDK with Azure-specific parameters
- Bedrock requires the boto3 SDK with specific service name 'bedrock-runtime'

### 2. Authentication Mechanism
- Azure uses a simple API key or managed identity token
- Bedrock uses AWS credentials with region-specific endpoints

### 3. Model Format
- Azure uses OpenAI-compatible formats
- Bedrock has provider-specific formats for each foundation model

### 4. API Structure
- Azure uses a direct chat completions endpoint
- Bedrock offers both direct model invocation (InvokeModel) and a unified Converse API

## Integration Approach

### 1. BedrockSettings Model
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

### 2. Add to Settings Class
```python
class Settings(BaseSettings):
    # Existing providers...
    
    bedrock: BedrockSettings | None = None
    """Settings for using Amazon Bedrock in the fast-agent application"""
```

### 3. Provider Implementation Strategy
- Create a new `BedrockAugmentedLLM` class that extends `AugmentedLLM`
- Implement the abstract methods and AWS-specific client creation
- Use the Converse API as the primary interface
- Handle provider-specific formatting for different model providers

### 4. AWS SDK Dependency Management
- Make the boto3 dependency optional, similar to the Azure approach with azure-identity
- Add exception handling for missing AWS SDK

## Provider-Specific Considerations

### 1. Client Creation
```python
def _bedrock_client(self):
    # Similar pattern to the Azure _openai_client method
    try:
        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": self.region
        }
        
        if self.use_default_credentials:
            # Use AWS credential provider chain
            if self.profile:
                session = boto3.Session(profile_name=self.profile)
                client = session.client(**client_kwargs)
            else:
                client = boto3.client(**client_kwargs)
        else:
            # Use explicit credentials
            client = boto3.client(
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                **client_kwargs
            )
            
        return client
    except Exception as e:
        raise ProviderKeyError("Bedrock client creation failed", str(e))
```

### 2. Converse API Approach
```python
async def _bedrock_completion(self, message, request_params=None):
    # Similar pattern to the OpenAI _openai_completion method
    client = self._bedrock_client()
    
    # Convert fast-agent formats to Bedrock Converse format
    # Handle messages, system prompt, and tool calls
    
    response = await self.executor.execute(
        client.converse,
        modelId=self.model_id,
        messages=[...],  # Convert from fast-agent format
        inferenceConfig={...},  # Map request_params
        toolConfig={...} if tools else None
    )
    
    # Process response and return content
```

## Conclusion

The integration of Amazon Bedrock with fast-agent should follow the established provider implementation pattern, with specific accommodations for AWS authentication and the Bedrock API structure. The Azure implementation provides an excellent template, although Bedrock will require handling more model-specific formatting variations.

The Converse API offers a more consistent interface across different foundation models, which aligns well with fast-agent's provider abstraction layer. Using boto3 with optional dependencies will ensure minimal overhead for users who don't need Bedrock support.