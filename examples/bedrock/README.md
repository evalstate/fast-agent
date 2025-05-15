# Amazon Bedrock Examples for FastAgent

This directory contains examples and configuration templates for using Amazon Bedrock with FastAgent.

## Prerequisites

1. An AWS account with access to Amazon Bedrock
2. Models enabled in your AWS account (Claude, Nova, Llama, etc.)
3. FastAgent installed with Bedrock support:
   ```bash
   pip install fast-agent-mcp[bedrock]
   ```

## Authentication Options

You can authenticate with AWS Bedrock in several ways:

1. **Direct API Keys** - Provide `access_key_id` and `secret_access_key` in config
2. **Default Credential Chain** - Uses environment variables, AWS credential files, and IAM roles
3. **Named Profile** - Specify a profile from your `~/.aws/credentials` file

See the configuration files for examples of each approach.

## Examples

### 1. Simple Bedrock Agent

The `simple_agent.py` script demonstrates basic usage of Bedrock models with FastAgent.

```bash
python -m examples.bedrock.simple_agent
```

### 2. Multimodal Bedrock Agent

The `multimodal_agent.py` script shows how to use Bedrock Claude models with images.

```bash
# Add an image to the images directory first
python -m examples.bedrock.multimodal_agent
```

## Configuration Files

### 1. Standard Configuration (`fastagent.config.yaml`)

A basic configuration for getting started with Bedrock. Includes settings for:
- Region selection
- Authentication options
- Model parameters

### 2. Production Configuration (`fastagent.production.yaml`)

A more advanced configuration optimized for production use cases:
- IAM role-based authentication
- Regional failover
- Logging configuration
- Monitoring settings

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

## Troubleshooting

1. **Authentication Errors**:
   - Ensure your AWS credentials have permission to call Bedrock APIs
   - Verify your AWS region has the models you're trying to use

2. **Model Access Errors**:
   - Make sure you've enabled the models in your AWS account
   - Some models require explicit model access requests in the AWS console

3. **Model ID Format Errors**:
   - Check that Claude models have the correct region prefix (e.g., `us.` or `eu.`)
   - Example: `us.anthropic.claude-3-5-sonnet-20241022-v2:0`

4. **Configuration Issues**:
   - When using `resource_name`, don't include `base_url` and vice versa
   - When using IAM roles, set `use_default_credentials: true`

For more help, see [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/)