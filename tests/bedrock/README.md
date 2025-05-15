# AWS Bedrock Testing Suite

This directory contains scripts for testing the AWS Bedrock integration with fast-agent. These tests help validate that your AWS credentials are properly configured and that the Bedrock integration works correctly.

## Testing Scripts

### 1. AWS Credentials Verification

**`test_aws_credentials.py`**

This script verifies that your AWS credentials can access Bedrock services. It tests your credentials by making API calls to Bedrock and validates permissions.

```bash
# Basic test
./test_aws_credentials.py

# List available models
./test_aws_credentials.py --list-models

# Specify region and profile
./test_aws_credentials.py --region us-west-2 --profile bedrock-profile

# Test with a specific model
./test_aws_credentials.py --model us.anthropic.claude-3-7-sonnet-20250219-v1:0
```

### 2. Real Agent Test

**`test_real_agent.py`**

Tests a real Bedrock agent with actual AWS credentials, verifying that the core functionality works.

```bash
# Basic test
./test_real_agent.py

# Interactive mode
./test_real_agent.py -i

# Specify model, region, and profile
./test_real_agent.py --model us.anthropic.claude-3-7-sonnet-20250219-v1:0 --region us-west-2 --profile bedrock-profile

# Use a config file
./test_real_agent.py --config path/to/fastagent.config.yaml
```

### 3. Multimodal Capabilities Test

**`test_multimodal.py`**

Tests Bedrock's multimodal capabilities by sending an image to a Claude model.

```bash
# Send an image with a query
./test_multimodal.py --image path/to/image.jpg

# Customize query, model, and region
./test_multimodal.py --image path/to/image.jpg --query "What's in this image?" --model us.anthropic.claude-3-7-sonnet-20250219-v1:0 --region us-west-2
```

### 4. Tool Calling Test

**`test_tool_calling.py`**

Tests Bedrock's function calling capabilities by using tools with Claude models.

```bash
# Basic test with default query
./test_tool_calling.py

# Customize query and model
./test_tool_calling.py --query "What's the weather like in Seattle?" --model us.anthropic.claude-3-7-sonnet-20250219-v1:0
```

## AWS Authentication Methods

The test scripts support multiple authentication methods:

1. **Default Credentials Chain** (Default)
   - Environment variables (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
   - AWS credentials file (`~/.aws/credentials`)
   - IAM roles (EC2, ECS, Lambda)

2. **Named Profile**
   - Specify a profile from your AWS credentials file
   - Use the `--profile` option with any script

3. **Config File**
   - Use a fast-agent configuration file with Bedrock settings
   - Use the `--config` option with the agent test scripts

## Important Notes About Model IDs

All Claude models on Bedrock require a region prefix in their model IDs. The correct format is:

- `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (US regions)
- `eu.anthropic.claude-3-5-sonnet-20241022-v2:0` (EU regions, where available)

If you encounter errors like:
```
Validation error: Invocation of model ID anthropic.claude-3-7-sonnet-20250219-v1:0 with on-demand throughput isn't supported
```

Make sure you're using the correct region prefix in your model ID.

## Supported Models

The scripts default to using modern Claude models, but you can specify any Bedrock model:

- `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (latest Claude 3.7 Sonnet model)
- `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (latest Claude 3.5 Sonnet model)
- `us.anthropic.claude-3-5-sonnet-20240620-v1:0` (previous Claude 3.5 Sonnet model)
- `us.anthropic.claude-3-haiku-20240307-v1:0` (Claude 3 Haiku model)
- Other models available in your AWS region

To see available models in your region, run:
```bash
./test_aws_credentials.py --list-models
```

## Troubleshooting

If you encounter issues:

1. **Credential Errors**
   - Verify your AWS credentials are set up correctly
   - Make sure you have the necessary Bedrock permissions
   - Try using a specific profile if your default credentials don't have access

2. **Model Errors**
   - Ensure the model is available in your region
   - Check that your account has access to the specified model
   - Make sure you're using the correct region prefix for the model (e.g., `us.` or `eu.`)

3. **Permission Errors**
   - Verify your IAM policy includes required Bedrock permissions:
     - `bedrock:ListFoundationModels`
     - `bedrock:GetFoundationModel`
     - `bedrock-runtime:InvokeModel`
     - `bedrock-runtime:Converse`

4. **Region Issues**
   - Bedrock may not be available in all AWS regions
   - Try using `us-east-1` or `us-west-2` if your region doesn't support Bedrock