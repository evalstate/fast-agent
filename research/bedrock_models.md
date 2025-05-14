# Amazon Bedrock Models

## Foundation Models Available on Bedrock

Amazon Bedrock offers a variety of foundation models from leading AI providers. Below is a comprehensive list of available models, organized by provider.

## Anthropic Claude Models

| Model ID | Description | Features |
|----------|-------------|----------|
| anthropic.claude-3-opus-20240229-v1:0 | Most capable Claude model | Multimodal, tool use, highest quality |
| anthropic.claude-3-sonnet-20240229-v1:0 | Balance of intelligence and speed | Multimodal, tool use, good for most use cases |
| anthropic.claude-3-haiku-20240307-v1:0 | Fastest Claude model | Multimodal, tool use, optimized for speed |
| anthropic.claude-3.5-sonnet-20240620-v1:0 | Improved Sonnet model | Enhanced reasoning, better coding |
| anthropic.claude-3.5-haiku-20241022-v1:0 | Latest Haiku model | Improved reasoning with speed |

## Meta Llama Models

| Model ID | Description | Features |
|----------|-------------|----------|
| meta.llama-3-70b-v1:0 | Llama 3 70B parameters | Strong reasoning, text generation |
| meta.llama-3-8b-v1:0 | Smaller Llama 3 model | Good performance/cost balance |
| meta.llama-3.1-405b-v1:0 | Largest Llama model | Advanced reasoning, highest quality |
| meta.llama-3.1-70b-v1:0 | Updated 70B model | Improved over 3.0, good all-rounder |
| meta.llama-3.1-8b-v1:0 | Updated 8B model | Improved over 3.0, efficient |

## Cohere Models

| Model ID | Description | Features |
|----------|-------------|----------|
| cohere.command-r-v1:0 | Core reasoning model | Strong reasoning, efficiency |
| cohere.command-r-plus-v1:0 | Enhanced command model | Better reasoning, higher quality |
| cohere.embed-english-v3:0 | Embedding model | Text embeddings, semantic search |
| cohere.embed-multilingual-v3:0 | Multilingual embeddings | Support for 100+ languages |

## Mistral Models

| Model ID | Description | Features |
|----------|-------------|----------|
| mistral.mistral-7b-v0:2 | Efficient 7B model | Good performance/cost ratio |
| mistral.mistral-large-v1:0 | Flagship Mistral model | Strong reasoning and capabilities |
| mistral.mixtral-8x7b-v0:1 | Mixture of experts | Efficient multi-domain expertise |

## Amazon Titan Models

| Model ID | Description | Features |
|----------|-------------|----------|
| amazon.titan-text-express-v1 | Fast text generation | Efficient general-purpose model |
| amazon.titan-text-lite-v1 | Lightweight text model | Fast, cost-effective |
| amazon.titan-text-premier-v1 | Premium text model | Highest quality Amazon model |
| amazon.titan-embed-text-v1 | Text embedding model | Semantic search, clustering |
| amazon.titan-image-generator-v1 | Image generation | Text-to-image capabilities |

## AI21 Labs Models

| Model ID | Description | Features |
|----------|-------------|----------|
| ai21.jamba-1.5-mini-v1:0 | Compact Jamba model | Efficient text generation |
| ai21.jamba-1.5-large-v1:0 | Full-size Jamba model | Strong reasoning and generation |

## Stability AI Models

| Model ID | Description | Features |
|----------|-------------|----------|
| stability.stable-diffusion-xl-v1:0 | Image generation | Text-to-image generation |
| stability.stable-image-v1:0 | Image editing | Image manipulation capabilities |

## Availability Considerations

- Model availability varies by AWS region
- Some models support streaming, while others don't
- Pricing varies significantly between models
- Tool use/function calling support varies by model
- Multimodal capabilities (image, document understanding) vary by model

## Recommended Models for Initial Integration

1. **anthropic.claude-3-sonnet-20240229-v1:0** - Best balance of capabilities and performance
2. **meta.llama-3-70b-v1:0** - Strong open model alternative
3. **amazon.titan-text-express-v1** - Native AWS model
4. **mistral.mistral-large-v1:0** - Good performance/cost ratio

These models represent a good cross-section of available providers and capabilities, making them ideal candidates for initial integration into fast-agent.