---
title: AWS Bedrock
social:
  title: AWS Bedrock
  tagline: Configure Amazon Bedrock models, AWS authentication, and model IDs.
  description: Configure Amazon Bedrock models, AWS authentication, and model IDs.
  alt: fast-agent social card — AWS Bedrock
---

# AWS Bedrock

AWS Bedrock provides access to multiple foundation models from Amazon, Anthropic, AI21, Cohere, Meta, Mistral, and other providers through a unified API. fast-agent supports the full range of Bedrock models with intelligent capability detection and optimization.

**Key Features:**

- **Multi-provider model access**: Nova, Claude, Titan, Cohere, Llama, Mistral, and more
- **Intelligent capability detection**: Automatically handles models that don't support system messages or tool use
- **Optimized streaming**: Uses streaming when supported, falls back to non-streaming when required
- **Model-specific optimizations**: Tailored configurations for different model families

**YAML Configuration:**

```yaml
bedrock:
  region: "us-east-1" # Required - AWS region where Bedrock is available
  profile: "default"  # Optional - AWS profile to use (defaults to "default")
                      # Only needed on local machines, not required on AWS
```

**Environment Variables:**

- `AWS_REGION` or `AWS_DEFAULT_REGION`: AWS region (e.g., `us-east-1`)
- `AWS_PROFILE`: Named AWS profile to use
- `AWS_ACCESS_KEY_ID`: Your AWS access key (handled by boto3)
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key (handled by boto3)
- `AWS_SESSION_TOKEN`: AWS session token for temporary credentials (handled by boto3)

**Model Name Format:**

Use `bedrock.model-id` where `model-id` is the Bedrock model identifier:

- `bedrock.amazon.nova-premier-v1:0` - Amazon Nova Premier
- `bedrock.amazon.nova-pro-v1:0` - Amazon Nova Pro
- `bedrock.amazon.nova-lite-v1:0` - Amazon Nova Lite
- `bedrock.anthropic.claude-3-7-sonnet-20241022-v1:0` - Claude 3.7 Sonnet
- `bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0` - Claude 3.5 Sonnet v2
- `bedrock.meta.llama3-1-405b-instruct-v1:0` - Meta Llama 3.1 405B
- `bedrock.mistral.mistral-large-2402-v1:0` - Mistral Large

**Supported Models:**

The provider automatically detects and handles model-specific capabilities:

- **System messages**: Automatically injects system prompts into user messages for models that don't support them (Titan, Cohere Command Text, etc.)
- **Tool use**: Skips tool preparation for models that don't support tools (Titan, Claude v2, Llama 2/3, etc.)
- **Streaming**: Uses non-streaming API when models don't support streaming with tools

Note that Bedrock contains some models that may perform poorly in some areas, including INSTRUCT models as well as models that are made to be fine-tuned for specific use cases.  If you are unsure about model capabilities, be sure to read the documentation.

**Model Capabilities:**

Refer to the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html) for the latest model capabilities including system prompts, tool use, vision, and streaming support.

**Authentication:**

AWS Bedrock uses standard AWS authentication. Configure credentials using:

1. **AWS CLI**: Run `aws configure` to set up credentials.  AWS SSO is a great choice for local development.
2. **Environment variables**: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
3. **IAM roles**: Use IAM roles when running on EC2 or other AWS services
4. **AWS profiles**: Use named profiles with `AWS_PROFILE` environment variable

Required IAM permissions:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`
