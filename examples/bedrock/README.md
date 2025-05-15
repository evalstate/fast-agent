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

### 1. Basic Bedrock Agent

The `basic_agent.py` script demonstrates basic usage of Bedrock models with FastAgent:

```python
import asyncio
from mcp_agent import FastAgent

async def main():
    # Initialize FastAgent with configuration
    fast = FastAgent("Bedrock Basic Demo", config_path="examples/bedrock/fastagent.config.yaml")
    
    # Define a Bedrock agent using the decorator pattern
    @fast.agent(
        "bedrock_agent",
        "You are a helpful assistant that provides concise responses.",
        model="bedrock.us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    async def run_bedrock_agent():
        # This function is just a placeholder for the decorator
        pass
    
    # Run the agent
    async with fast.run() as agent:
        # Send a message to the agent
        response = await agent.bedrock_agent.send("Tell me about Amazon Bedrock in 3 sentences.")
        print(response)
        
        # Follow-up question
        response = await agent.bedrock_agent.send("Which model am I currently using?")
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

Run the example with:
```bash
python -m examples.bedrock.basic_agent
```

### 2. Multimodal Bedrock Agent

The `multimodal_agent.py` script shows how to use Bedrock Claude models with images:

```python
import asyncio
import base64
import mimetypes
from pathlib import Path
from mcp_agent import FastAgent
from mcp_agent.core.prompt import Prompt

async def main():
    # Initialize FastAgent with configuration
    fast = FastAgent("Bedrock Multimodal Demo", config_path="examples/bedrock/fastagent.config.yaml")
    
    # Load an image file
    image_path = Path("examples/bedrock/images/new_york.jpg")
    
    # Create an agent with Claude model that supports vision
    @fast.agent(
        "multimodal_agent",
        "You are a helpful AI assistant that can analyze images and text.",
        model="bedrock.us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    async def bedrock_multimodal():
        # This function is just a placeholder for the decorator
        pass
    
    # Run the agent
    async with fast.run() as agent:
        # Create a prompt with both text and image using the Prompt.user helper
        prompt = Prompt.user(
            "What do you see in this image? What city is shown?",
            image_path  # Pass the Path object directly - Prompt.user will handle it
        )
        
        # Send the prompt to the agent
        response = await agent.multimodal_agent.send(prompt)
        print(response)
        
        # Follow-up question (including the image again for context)
        await asyncio.sleep(10)  # Delay to avoid throttling
        follow_up_prompt = Prompt.user(
            "What famous landmarks can you identify?",
            image_path
        )
        follow_up = await agent.multimodal_agent.send(follow_up_prompt)
        print(follow_up)

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
from typing import Dict, Any
from mcp_agent import FastAgent
from mcp.types import Tool

# Define handler functions for tools
def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get the current weather for a location.
    
    Args:
        location: The name of the city or location
        unit: Temperature unit (celsius or fahrenheit)
        
    Returns:
        Weather data dictionary
    """
    # This would normally call a weather API
    # Here we're returning mock data for demonstration
    if unit == "fahrenheit":
        return {
            "location": location,
            "temperature": 72,
            "unit": "°F",
            "condition": "Partly Cloudy",
            "humidity": 65,
            "wind_speed": 8,
            "wind_direction": "NW"
        }
    else:
        return {
            "location": location,
            "temperature": 22,
            "unit": "°C",
            "condition": "Partly Cloudy", 
            "humidity": 65,
            "wind_speed": 13,
            "wind_direction": "NW"
        }

async def main():
    # Initialize FastAgent with configuration
    fast = FastAgent("Function Calling Demo", config_path="examples/bedrock/fastagent.config.yaml")
    
    # Create tools that will be available to the model
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        inputSchema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The name of the city or location"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    )
    
    # Map tool names to handler functions
    tool_handlers = {
        "get_weather": get_weather
    }
    
    # Create an agent with Claude model that supports tool calling
    @fast.agent(
        "tool_calling_agent",
        "You are a helpful assistant that can use tools to provide accurate information.",
        model="bedrock.us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        tools=[weather_tool]
    )
    async def tool_calling_agent():
        # This function is just a placeholder for the decorator
        pass
    
    # Run the agent
    async with fast.run() as agent:
        print("Sending message that should trigger tool use...")
        response = await agent.tool_calling_agent.send("What's the weather in Seattle?")
        
        # Check if the model has requested to use a tool
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"\nTool called: {tool_call.name}")
            print(f"Arguments: {json.dumps(tool_call.arguments, indent=2)}")
            
            # Get the appropriate handler function for this tool
            if tool_call.name in tool_handlers:
                handler = tool_handlers[tool_call.name]
                
                # Call the handler with the arguments
                try:
                    result = handler(**tool_call.arguments)
                    print(f"\nTool result: {json.dumps(result, indent=2)}")
                    
                    # Send the tool result back to the model
                    final_response = await agent.tool_calling_agent.send(
                        "",  # Empty message for tool response
                        tool_results=[{
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "result": result
                        }]
                    )
                    
                    print("\nFinal response:")
                    print(final_response)
                    
                except Exception as e:
                    print(f"Error executing tool: {e}")
            else:
                print(f"No handler found for tool: {tool_call.name}")
        else:
            print("Model did not request to use a tool.")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the example with:
```bash
python -m examples.bedrock.function_calling
```

### 4. Model Format Tester

The `model_format_tester.py` script helps you determine which message formats work with different model families in your specific AWS region:

```python
import asyncio
import json
import boto3
import argparse

async def main():
    parser = argparse.ArgumentParser(description='Test AWS Bedrock model formats')
    parser.add_argument('--region', type=str, default='us-east-1')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    # Test different model formats
    await test_claude_format(args.region, args.verbose)
    await test_nova_format(args.region, args.verbose)
    await test_llama_prompt_format(args.region, args.verbose)
    await test_llama_messages_format(args.region, args.verbose)
```

Run the format tester with:
```bash
python -m examples.bedrock.model_format_tester --region us-east-1 --verbose
```

This will test:
- Claude models with the Converse API + anthropic_version
- Nova models with the minimal messages format
- Meta Llama models with both prompt format and messages format

The script will show which formats work in your region, helping you configure FastAgent correctly.

### 5. AWS Bedrock Model Test Suite

The test scripts for different model families:

- `test_amazon_nova.py`: Tests Amazon Nova models (Pro, Lite)
- `test_anthropic_claude.py`: Tests Anthropic Claude models (3.5, 3.7)
- `test_meta_llama.py`: Tests Meta Llama models (3 70B)

These scripts test both direct API access and FastAgent integration for each model family, helping to validate your setup.

Run the test scripts with:
```bash
python -m examples.bedrock.test_amazon_nova
python -m examples.bedrock.test_anthropic_claude
python -m examples.bedrock.test_meta_llama
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

3. **Message Format**
   - Claude models use the `messages` format with `anthropic_version` parameter:
   ```json
   {
     "anthropic_version": "bedrock-2023-05-31",
     "max_tokens": 1024,
     "messages": [
       {
         "role": "user",
         "content": [
           {"type": "text", "text": "Your message here"}
         ]
       }
     ]
   }
   ```

### Amazon Nova Models

Nova models provide Amazon's optimized foundation models:

1. **Performance Tiers**
   - Nova Pro: High-end, full-featured model
   - Nova Lite: Balanced performance and cost
   - Nova Micro: Light-weight, cost-effective option

2. **Strict Message Format Requirements**
   - All Nova models (Pro, Lite, Micro, Premier) use the Converse API
   - They require a very specific format with NO additional parameters:
   ```json
   {
     "messages": [
       {
         "role": "user",
         "content": [{"text": "Your message here"}]
       }
     ]
   }
   ```
   - **⚠️ CRITICAL**: Each message MUST have both 'role' and 'content' keys
   - **⚠️ CRITICAL**: The 'content' field MUST be a list containing objects with 'text' keys
   - **⚠️ WARNING**: Adding ANY parameters like temperature, top_p, max_tokens will cause ValidationException errors

3. **Response Format**
   - Response includes an `output` field with the message content:
   ```json
   {
     "output": {"message": {"content": [{"text": "Response text"}], "role": "assistant"}},
     "usage": {"inputTokens": 10, "outputTokens": 50, "totalTokens": 60}
   }
   ```
   
4. **Configuration in FastAgent**
   - For Nova models, use an empty parameter block in fastagent.config.yaml:
   ```yaml
   model_params:
     "us.amazon.nova-pro-v1:0": {}
     "us.amazon.nova-lite-v1:0": {}
     "us.amazon.nova-micro-v1:0": {}
     "us.amazon.nova-premier-v1:0": {}
   ```

### Meta Llama Models

Meta's Llama 3 and 4 models offer specialized capabilities:

1. **Scale Options**
   - Range from 1B to 405B parameters
   - Smaller models offer faster response times

2. **Specialized Versions**
   - Llama 4 Maverick: Optimized for creative tasks
   - Llama 4 Scout: Optimized for analytical tasks

3. **Format Variations by Region**
   - Some regions use traditional **prompt format**:
   ```json
   {
     "prompt": "<s>[INST] Your system message [/INST]\n\nYour user message</s>",
     "max_gen_len": 1024,
     "temperature": 0.7
   }
   ```
   
   - Other regions use Converse API with **messages format**:
   ```json
   {
     "messages": [
       {
         "role": "system",
         "content": "Your system message"
       },
       {
         "role": "user",
         "content": "Your user message"
       }
     ]
   }
   ```
   
   - **⚠️ WARNING**: Some parameters like `top_p` can cause `'RequestParams' object has no attribute 'top_p'` errors. The safest approach is to use minimal parameters.
   - Test which format works in your region using the `model_format_tester.py` example

4. **Configuration in FastAgent**
   - For Meta Llama models, use an empty parameter block in fastagent.config.yaml to avoid errors:
   ```yaml
   model_params:
     "us.meta.llama3-1-70b-instruct-v1:0": {}
   ```

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

### Model-Specific Format Errors

For Nova Pro and other Nova models:

```
An error occurred (ValidationException) when calling the InvokeModel operation: Malformed input request: #/messages/0: required key [role] not found, please reformat your input and try again.
```
```
An error occurred (ValidationException) when calling the InvokeModel operation: Malformed input request: #/messages/0: required key [content] not found, please reformat your input and try again.
```
```
An error occurred (ValidationException) when calling the InvokeModel operation: Malformed input request: #/messages/0: extraneous key [text] is not permitted, please reformat your input and try again.
```

**Solutions**:
- Make sure your messages format follows the exact structure required:
  ```json
  {"messages": [{"role": "user", "content": [{"text": "Your message"}]}]}
  ```
- **CRITICAL**: Each message MUST have both 'role' and 'content' keys
- **CRITICAL**: The 'content' field MUST be a list containing objects with 'text' keys
- Remove ALL additional parameters like temperature, top_p, etc.
- If using a system prompt, include it properly with "role": "system"
- In FastAgent config, use empty parameter blocks for all Nova models:
  ```yaml
  model_params:
    "us.amazon.nova-pro-v1:0": {}
    "us.amazon.nova-lite-v1:0": {}
  ```

For Meta Llama models:

```
An error occurred (ValidationException) when calling the InvokeModel operation: Malformed input request: #: required key [prompt] not found#: extraneous key [messages] is not permitted, please reformat your input and try again.
```
```
'RequestParams' object has no attribute 'top_p'
```

**Solutions**:
- Check which API format your region uses (prompt vs messages)
- If error mentions "required key [prompt]", use the traditional format:
  ```json
  {"prompt": "<s>[INST]...[/INST]</s>", "max_gen_len": 1024, "temperature": 0.7}
  ```
- **IMPORTANT**: Use minimal parameters to avoid attribute errors, especially avoid 'top_p' and 'top_k'
- In FastAgent config, use empty parameter blocks for all Meta models:
  ```yaml
  model_params:
    "us.meta.llama3-1-70b-instruct-v1:0": {}
  ```
- Reference the examples in this repository for properly formatted requests

### Error Recovery Strategies

For cases where model API formats change or region-specific differences occur:

1. **Start with minimal requests**: Remove all optional parameters first
2. **Test with direct AWS SDK**: Use boto3 directly to test the format before integration
3. **Check region-specific requirements**: Models may have different formats in different regions
4. **Implement fallback mechanisms**: Your code should gracefully handle format errors and try alternatives
5. **Monitor AWS announcements**: The Bedrock API evolves, so stay updated on changes

For more help, see [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/) or reach out to AWS Support.