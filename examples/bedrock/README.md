# Amazon Bedrock Integration for FastAgent

This directory contains examples, configuration templates, and documentation for using Amazon Bedrock with FastAgent. Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) through a unified API.

## Overview

The AWS Bedrock integration for FastAgent allows you to:

1. Connect to AWS Bedrock using various authentication methods
2. Use all available Bedrock foundation models (Claude, Nova, Meta Llama, etc.)
3. Take advantage of model-specific features like function calling and multimodal capabilities
4. Manage model configuration with a flexible YAML-based approach

## Prerequisites

1. **AWS Account** with access to Amazon Bedrock
   - Sign up for AWS at [aws.amazon.com](https://aws.amazon.com/)
   - Navigate to the Bedrock console and request access to desired models

2. **Models Enabled** in your AWS account
   - In the AWS Bedrock console, go to "Model access"
   - Request access to the models you want to use (Claude, Nova, Llama, etc.)
   - Wait for approval (some models are approved automatically)

3. **Necessary IAM Permissions**
   - Your IAM user or role needs these permissions:
     - `bedrock:ListFoundationModels`
     - `bedrock:GetFoundationModel`
     - `bedrock-runtime:InvokeModel`
     - `bedrock-runtime:Converse`

4. **FastAgent Installed** with Bedrock support:
   ```bash
   pip install fast-agent-mcp[bedrock]
   ```

## Authentication Options

You can authenticate with AWS Bedrock in several ways:

### 1. Default Credential Chain

The simplest approach is to let AWS SDK use its default credential chain, which checks these sources in order:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS credential file (`~/.aws/credentials`)
- IAM roles (for EC2, Lambda, or ECS environments)

```yaml
providers:
  bedrock:
    use_default_credentials: true
    region_name: "us-east-1"
```

### 2. Named Profile

If you have multiple AWS profiles in your credentials file, specify a profile:

```yaml
providers:
  bedrock:
    profile_name: "my-bedrock-profile"
    region_name: "us-east-1"
```

### 3. Direct API Keys

Provide API keys directly in the configuration (use with caution, better for testing):

```yaml
providers:
  bedrock:
    access_key_id: "YOUR_ACCESS_KEY"
    secret_access_key: "YOUR_SECRET_KEY"
    region_name: "us-east-1"
```

### 4. Assume IAM Role (Advanced)

For cross-account access or enhanced security, assume an IAM role:

```yaml
providers:
  bedrock:
    use_default_credentials: true
    assume_role_arn: "arn:aws:iam::123456789012:role/BedockAccessRole"
    region_name: "us-east-1"
```

## Examples

### 1. Simple Bedrock Agent

The `simple_agent.py` script demonstrates basic usage of Bedrock models with FastAgent:

```python
import asyncio
from mcp_agent import FastAgent

async def main():
    # Initialize FastAgent with configuration
    fast = FastAgent(config_file="examples/bedrock/fastagent.config.yaml")
    
    # Create an agent with Bedrock provider
    async with fast.run() as agent:
        # Chat with the agent
        response = await agent.chat("Tell me about Amazon Bedrock in 3 sentences.")
        print(response.content)
        
        # Follow-up question
        response = await agent.chat("Which model am I currently using?")
        print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

Run the example with:
```bash
python -m examples.bedrock.simple_agent
```

### 2. Multimodal Bedrock Agent

The `multimodal_agent.py` script shows how to use Bedrock Claude models with images:

```python
import asyncio
import pathlib
from mcp_agent import FastAgent
from mcp_agent.mcp.mime_utils import create_image_content

async def main():
    # Initialize FastAgent with configuration
    fast = FastAgent(config_file="examples/bedrock/fastagent.config.yaml")
    
    # Load an image file
    image_path = pathlib.Path("examples/bedrock/images/sample.jpg")
    image_content = create_image_content(image_path)
    
    # Create a multimodal agent
    async with fast.run() as agent:
        # Send a message with both text and image
        response = await agent.chat(
            "What's in this image? Describe it in detail.", 
            attachments=[image_content]
        )
        print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

Run the example with:
```bash
python -m examples.bedrock.multimodal_agent
```

### 3. Function Calling with Bedrock

The `function_calling.py` script demonstrates using tool/function calling with Claude 3 Sonnet models:

```python
import asyncio
import json
from mcp_agent import FastAgent
from mcp_agent.mcp.prompt_message_multipart import MPTool

# Define tools/functions available to the model
WEATHER_TOOL = MPTool(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["city"]
    }
)

async def main():
    # Initialize FastAgent with configuration
    fast = FastAgent(config_file="examples/bedrock/fastagent.config.yaml")
    
    # Create an agent with tool support
    async with fast.run() as agent:
        # Enable tool calling and register tools
        agent.tools = [WEATHER_TOOL]
        
        # Ask a question that should trigger tool use
        response = await agent.chat("What's the weather like in Seattle right now?")
        
        # Handle tool calls
        if response.tool_calls:
            print(f"Tool called: {response.tool_calls[0].name}")
            print(f"Arguments: {json.dumps(response.tool_calls[0].arguments, indent=2)}")
            
            # In a real implementation, you would call the actual service
            # Here we just mock a response
            tool_response = {
                "temperature": 65,
                "condition": "Partly Cloudy",
                "humidity": 72
            }
            
            # Send the tool response back to the model
            final_response = await agent.chat(
                "",  # Empty message for tool response
                tool_results=[{
                    "tool_call_id": response.tool_calls[0].id,
                    "name": response.tool_calls[0].name,
                    "results": tool_response
                }]
            )
            
            print("\nFinal response:")
            print(final_response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

Run the example with:
```bash
python -m examples.bedrock.function_calling
```

## Configuration Files

### 1. Standard Configuration (`fastagent.config.yaml`)

A basic configuration for getting started with Bedrock:

```yaml
providers:
  bedrock:
    use_default_credentials: true
    region_name: "us-east-1"

agents:
  bedrock_agent:
    provider: "bedrock"
    model: "bedrock.sonnet35"  # Using an alias for Claude 3.5 Sonnet
    temperature: 0.7
    max_tokens: 4096
```

Key configuration options:
- `region_name`: AWS region where Bedrock is available
- `model`: Model ID or alias (see [Available Bedrock Models](#available-bedrock-models))
- `temperature`: Controls randomness (0.0-1.0)
- `max_tokens`: Maximum tokens to generate in response
- `top_p`: Nucleus sampling parameter (0.0-1.0)

### 2. Production Configuration (`fastagent.production.yaml`)

A more advanced configuration optimized for production use cases:

```yaml
providers:
  bedrock:
    use_default_credentials: true
    assume_role_arn: "arn:aws:iam::123456789012:role/BedockAccessRole"
    region_name: "us-east-1"
    fallback_region_name: "us-west-2"
    # Retry configuration
    max_retries: 3
    timeout: 60
    # Optional custom endpoint for PrivateLink or VPC endpoints
    endpoint_url: "https://bedrock-runtime.us-east-1.amazonaws.com"

logging:
  level: "INFO"
  handlers:
    - type: "console"
    - type: "file"
      filename: "bedrock_agent.log"
```

Additional production settings:
- `assume_role_arn`: IAM role to assume for enhanced security
- `fallback_region_name`: Backup region if primary is unavailable
- `max_retries`: Number of retry attempts for API calls
- `timeout`: Request timeout in seconds
- `endpoint_url`: Custom endpoint for PrivateLink or VPC endpoints

## Available Bedrock Models

FastAgent supports all available AWS Bedrock models, including:

1. **Anthropic Claude Models (US Region)**
   - `us.anthropic.claude-3-haiku-20240307-v1:0`
   - `us.anthropic.claude-3-5-haiku-20241022-v1:0`
   - `us.anthropic.claude-3-5-sonnet-20240620-v1:0`
   - `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (Latest Sonnet 3.5)
   - `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (Premium)
   - `us.anthropic.claude-3-opus-20240229-v1:0`
   - `us.anthropic.claude-3-sonnet-20240229-v1:0`

   **Anthropic Claude Models (EU Region)**
   - `eu.anthropic.claude-3-haiku-20240307-v1:0`
   - `eu.anthropic.claude-3-5-sonnet-20240620-v1:0`
   - `eu.anthropic.claude-3-7-sonnet-20250219-v1:0`
   - `eu.anthropic.claude-3-sonnet-20240229-v1:0`

2. **Amazon Nova Models (US Region)**
   - `us.amazon.nova-lite-v1:0`
   - `us.amazon.nova-micro-v1:0`
   - `us.amazon.nova-premier-v1:0`
   - `us.amazon.nova-pro-v1:0`

   **Amazon Nova Models (EU Region)**
   - `eu.amazon.nova-lite-v1:0`
   - `eu.amazon.nova-micro-v1:0`
   - `eu.amazon.nova-pro-v1:0`

3. **Meta Llama 3 Models (US Region)**
   - `us.meta.llama3-1-8b-instruct-v1:0`
   - `us.meta.llama3-1-70b-instruct-v1:0`
   - `us.meta.llama3-1-405b-instruct-v1:0`
   - `us.meta.llama3-2-1b-instruct-v1:0`
   - `us.meta.llama3-2-3b-instruct-v1:0`
   - `us.meta.llama3-2-11b-instruct-v1:0`
   - `us.meta.llama3-2-90b-instruct-v1:0`
   - `us.meta.llama3-3-70b-instruct-v1:0`

   **Meta Llama 3 Models (EU Region)**
   - `eu.meta.llama3-2-1b-instruct-v1:0`
   - `eu.meta.llama3-2-3b-instruct-v1:0`

4. **Meta Llama 4 Models (US Region)**
   - `us.meta.llama4-maverick-17b-instruct-v1:0`
   - `us.meta.llama4-scout-17b-instruct-v1:0`

## Important Note on Model IDs

All Claude models on Bedrock require a region prefix in their model IDs:
- `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (US regions)
- `eu.anthropic.claude-3-5-sonnet-20241022-v2:0` (EU regions, where available)

If you encounter errors like:
```
Validation error: Invocation of model ID anthropic.claude-3-7-sonnet-20250219-v1:0 with on-demand throughput isn't supported
```

Make sure you're using the correct region prefix in your model ID.

## Convenient Aliases

FastAgent provides aliases for common Bedrock models:

### Anthropic Claude Models (US Region)
- `bedrock.haiku` → `us.anthropic.claude-3-haiku-20240307-v1:0`
- `bedrock.haiku35` → `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- `bedrock.sonnet3` → `us.anthropic.claude-3-sonnet-20240229-v1:0`
- `bedrock.sonnet35` → `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
- `bedrock.sonnet` → `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (Latest Sonnet)
- `bedrock.sonnet-latest` → `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
- `bedrock.sonnet-previous` → `us.anthropic.claude-3-5-sonnet-20240620-v1:0`
- `bedrock.sonnet37` → `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (Premium)
- `bedrock.opus` → `us.anthropic.claude-3-opus-20240229-v1:0`

### Anthropic Claude Models (EU Region)
- `bedrock.eu.haiku` → `eu.anthropic.claude-3-haiku-20240307-v1:0`
- `bedrock.eu.sonnet3` → `eu.anthropic.claude-3-sonnet-20240229-v1:0`
- `bedrock.eu.sonnet35` → `eu.anthropic.claude-3-5-sonnet-20240620-v1:0`
- `bedrock.eu.sonnet37` → `eu.anthropic.claude-3-7-sonnet-20250219-v1:0`

### Amazon Nova Models (US Region)
- `bedrock.nova-lite` → `us.amazon.nova-lite-v1:0`
- `bedrock.nova-micro` → `us.amazon.nova-micro-v1:0`
- `bedrock.nova-premier` → `us.amazon.nova-premier-v1:0`
- `bedrock.nova-pro` → `us.amazon.nova-pro-v1:0`

### Amazon Nova Models (EU Region)
- `bedrock.eu.nova-lite` → `eu.amazon.nova-lite-v1:0`
- `bedrock.eu.nova-micro` → `eu.amazon.nova-micro-v1:0`
- `bedrock.eu.nova-pro` → `eu.amazon.nova-pro-v1:0`

### Meta Llama Models
- `bedrock.llama3-8b` → `us.meta.llama3-1-8b-instruct-v1:0`
- `bedrock.llama3-70b` → `us.meta.llama3-1-70b-instruct-v1:0`
- `bedrock.llama3-405b` → `us.meta.llama3-1-405b-instruct-v1:0`
- `bedrock.llama4-maverick` → `us.meta.llama4-maverick-17b-instruct-v1:0`
- `bedrock.llama4-scout` → `us.meta.llama4-scout-17b-instruct-v1:0`

## AWS Region Considerations

Not all Bedrock models are available in all AWS regions. Common regions with good model coverage:

- `us-east-1` (N. Virginia)
- `us-west-2` (Oregon)

Check [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for the latest model availability by region.

## Model-Specific Features

### Claude 3 Models

Claude 3 models on Bedrock support several advanced features:

1. **Multimodal Input**
   - All Claude 3 models can analyze images
   - Example: Use the `attachments` parameter with `create_image_content`
   
2. **Function/Tool Calling**
   - Claude 3 Sonnet and Opus support tool calling
   - Define tools using the `MPTool` class
   - Handle tool responses in your application logic

### Amazon Nova Models

Nova models provide Amazon's optimized foundation models:

1. **Performance Tiers**
   - Nova Pro: High-end, full-featured model
   - Nova Lite: Balanced performance and cost
   - Nova Micro: Light-weight, cost-effective option

2. **Parameter Adjustment**
   - `temperature`: 0.0-1.0 (default: 0.7)
   - `top_p`: 0.0-1.0 (default: 0.9)
   - `top_k`: 1-500 (default: 250)

### Meta Llama Models

Meta's Llama 3 and 4 models offer specialized capabilities:

1. **Scale Options**
   - Range from 1B to 405B parameters
   - Smaller models offer faster response times

2. **Specialized Versions**
   - Llama 4 Maverick: Optimized for creative tasks
   - Llama 4 Scout: Optimized for analytical tasks

## Best Practices

1. **Cost Management**
   - Start with smaller, less expensive models (Claude 3 Haiku, Nova Micro)
   - Use `max_tokens` to limit response length
   - Monitor your AWS Bedrock usage in the AWS console

2. **Error Handling**
   - Implement retry logic for transient errors
   - Use try/except blocks to catch and handle AWS service exceptions
   - Add logging to track API call success/failure

3. **Security**
   - Store credentials securely (AWS Secrets Manager, environment variables)
   - Use IAM roles with minimal permissions
   - Consider using VPC endpoints for enhanced network security

4. **Performance Optimization**
   - Choose the appropriate AWS region for your workload
   - Use streaming responses for improved user experience
   - Pre-warm connections for high-volume applications

## Troubleshooting

### Authentication Errors

```
botocore.exceptions.ClientError: An error occurred (AccessDeniedException) when calling the InvokeModel operation: Access denied
```

**Solutions**:
- Ensure your AWS credentials have the required IAM permissions
- Verify you've set the correct region where your models are activated
- Check that your credentials aren't expired
- If using a role, ensure the role has the necessary trust relationship

### Model Access Errors

```
botocore.exceptions.ClientError: An error occurred (AccessDeniedException) when calling the InvokeModel operation: User doesn't have access to model us.anthropic.claude-3-5-sonnet-20241022-v2:0
```

**Solutions**:
- In the AWS Bedrock console, go to "Model access" and request access
- Wait for model access approval (can take minutes to hours)
- Verify you're using a region where the model is available

### Model ID Format Errors

```
Validation error: Invocation of model ID anthropic.claude-3-5-sonnet-20241022-v2:0 with on-demand throughput isn't supported
```

**Solutions**:
- Add the required region prefix (`us.` or `eu.`) to your model ID
- Use the correct model version including the version tag
- Try using a model alias instead of the full ID

### Configuration Issues

```
botocore.exceptions.ParamValidationError: Parameter validation failed: Parameter 'resourceName' and 'baseUrl' cannot be specified together
```

**Solutions**:
- When using `resource_name`, don't include `base_url` and vice versa
- When using IAM roles, set `use_default_credentials: true`
- Remove any conflicting parameters from your configuration

### Request Formatting Errors

```
botocore.exceptions.ClientError: An error occurred (ValidationException) when calling the InvokeModel operation: 1 validation error detected: Value at 'body' failed to satisfy constraint: Member must have length less than or equal to 16000000
```

**Solutions**:
- Reduce the size of your input (especially for images)
- Split large requests into multiple smaller requests
- Check model-specific input size limitations

For more help, see [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/) or reach out to AWS Support.