# Amazon Bedrock Examples for FastAgent

This directory contains examples and configuration templates for using Amazon Bedrock with FastAgent.

## Prerequisites

1. An AWS account with access to Amazon Bedrock
2. Models enabled in your AWS account (Claude, Titan, Llama, etc.)
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

FastAgent supports all available Bedrock models, including:

1. **Anthropic Claude Models**
   - `anthropic.claude-3-haiku-20240307`
   - `anthropic.claude-3-5-sonnet-20240620-v1:0`
   - `anthropic.claude-3-5-sonnet-20241022-v2:0` (Latest)
   - `anthropic.claude-3-7-sonnet-20250219-v1:0` (Premium)
   - `anthropic.claude-3-opus-20240229`

2. **Amazon Titan Models**
   - `amazon.titan-text-express-v1`
   - `amazon.titan-text-lite-v1`
   - `amazon.titan-text-premier-v1`

3. **Meta Llama Models**
   - `meta.llama3-8b-instruct-v1:0`
   - `meta.llama3-70b-instruct-v1:0`
   - `meta.llama3-405b-instruct-v1:0`

## Convenient Aliases

FastAgent provides aliases for common Bedrock models:

- `bedrock.haiku` → `anthropic.claude-3-haiku-20240307`
- `bedrock.sonnet` → `anthropic.claude-3-5-sonnet-20241022-v2:0` (Latest)
- `bedrock.sonnet-latest` → `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `bedrock.sonnet-previous` → `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `bedrock.sonnet37` → `anthropic.claude-3-7-sonnet-20250219-v1:0` (Premium)
- `bedrock.opus` → `anthropic.claude-3-opus-20240229`
- `bedrock.titan-express` → `amazon.titan-text-express-v1`
- `bedrock.llama3-70b` → `meta.llama3-70b-instruct-v1:0`

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

3. **Configuration Issues**:
   - When using `resource_name`, don't include `base_url` and vice versa
   - When using IAM roles, set `use_default_credentials: true`

For more help, see [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/).