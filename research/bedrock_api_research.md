# Amazon Bedrock API Research

## Overview
Amazon Bedrock is a fully managed AWS service that provides API access to foundation models from leading AI companies through a unified interface. It enables serverless model experimentation, customization, and deployment for building generative AI applications.

## Key Components

### 1. Bedrock Service
- **bedrock** client: For model management, customization, and evaluation
- **bedrock-runtime** client: For model inference and conversation

### 2. Main API Methods

#### Runtime API Methods
- **converse**: For conversational interactions with models
  - Provides a consistent interface across different models
  - Supports multimodal inputs and tool use
  
- **converse_stream**: Streaming version of the converse API
  
- **invoke_model**: Direct model invocation without conversation tracking
  - Model-specific request/response formats
  - Used for single-turn interactions

- **invoke_model_with_response_stream**: Streaming version of invoke_model

### 3. Converse API Structure

#### Request Parameters
- **modelId** (required): Specifies the model to use
- **messages** (required): List of message inputs with roles and content
- **system** (optional): Context/instructions for the model
- **inferenceConfig** (optional): Controls generation parameters (temperature, max tokens, etc.)
- **toolConfig** (optional): Configures tools for the model to use
- **guardrailConfig** (optional): Applies content safety filters
- **additionalModelRequestFields** (optional): Model-specific parameters

#### Response Structure
- **output**: The model's generated message
- **stopReason**: Why generation stopped
- **usage**: Token consumption metrics
- **metrics**: Performance data

### 4. Authentication
- Requires AWS credentials
- Uses standard AWS authentication mechanisms:
  - Environment variables
  - Shared credentials file
  - IAM roles
  - AWS SSO

### 5. Available Models

Major providers and models include:

#### Anthropic Claude
- claude-3-opus-20240229-v1:0
- claude-3-sonnet-20240229-v1:0
- claude-3-haiku-20240307-v1:0
- claude-3.5-sonnet-20240620-v1:0

#### Meta Llama
- meta.llama-3-70b-v1:0
- meta.llama-3-8b-v1:0
- meta.llama-3.1-405b-v1:0
- meta.llama-3.1-70b-v1:0
- meta.llama-3.1-8b-v1:0

#### Cohere
- cohere.command-r-v1:0
- cohere.command-r-plus-v1:0

#### Mistral
- mistral.mistral-7b-v0:2
- mistral.mistral-large-v1:0

#### Amazon Titan
- amazon.titan-text-express-v1
- amazon.titan-text-lite-v1
- amazon.titan-text-premier-v1
- amazon.titan-embed-text-v1

## Implementation Considerations

### 1. Client Initialization
```python
import boto3

# Standard client initialization
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# With additional configuration
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com',
    aws_access_key_id='YOUR_ACCESS_KEY',  # Optional
    aws_secret_access_key='YOUR_SECRET_KEY'  # Optional
)
```

### 2. Converse API Example (Conceptual)
```python
response = bedrock_runtime.converse(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    messages=[
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'What is Amazon Bedrock?'
                }
            ]
        }
    ],
    inferenceConfig={
        'temperature': 0.7,
        'topP': 0.9,
        'maxTokens': 1000
    }
)

# Extract response
response_message = response['output']
```

### 3. Model-Specific Considerations

Different models have unique parameter requirements:

#### Anthropic Claude
- Supports system instructions
- Handles multimodal inputs
- Provides tool use capability
- Uses 'anthropic_version' parameter

#### Cohere
- Different message format
- Uses 'chat_history' for context

#### Amazon Titan
- Different parameter naming conventions
- Unique response formats

### 4. Error Handling

Common error types:
- **ValidationException**: Invalid request parameters
- **AccessDeniedException**: Permissions issues
- **ResourceNotFoundException**: Model not found
- **ThrottlingException**: Rate limit exceeded
- **ServiceUnavailableException**: Service unavailable

### 5. Integration Requirements

For fast-agent integration:
- Boto3 dependency (boto3>=1.34.0)
- Model provider mapping
- Message format conversion layers
- Authentication handling
- Error handling and retries
- Configuration flexibility

## Conclusion

The Amazon Bedrock API provides a unified interface to multiple foundation models with the Converse API offering a consistent interaction pattern across different model providers. Integration will require careful handling of model-specific formats, authentication mechanisms, and robust error handling.

The Converse API is particularly well-suited for multi-turn conversations and closely aligns with the existing provider pattern in fast-agent, making it a good candidate for implementation.