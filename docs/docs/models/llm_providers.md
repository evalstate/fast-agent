
For each model provider, you can configure parameters either through environment variables or in your `fastagent.config.yaml` file.

Be sure to run `fast-agent check` to troubleshoot API Key issues:

![Key Check](check.png)

## Common Configuration Format

In your `fastagent.config.yaml`:

```yaml
<provider>:
  api_key: "your_api_key" # Override with API_KEY env var
  base_url: "https://api.example.com" # Base URL for API calls
  default_headers: # Optional - custom headers for all API requests
    X-Custom-Header: "value"
```

The `default_headers` option is available for OpenAI-compatible providers (including Azure).

## Anthropic

Anthropic models support Text, Vision and PDF content.

**YAML Configuration:**

```yaml
anthropic:
  api_key: "your_anthropic_key" # Required
  base_url: "https://api.anthropic.com/v1" # Default, only include if required
  cache_mode: "auto" # Options: off, prompt, auto (default: auto)
  cache_ttl: "5m" # Options: 5m, 1h (default: 5m)
  web_search:
    enabled: false
    # max_uses: 3
    # allowed_domains: ["example.com", "*.docs.example.com"]
    # blocked_domains: ["social.example"]  # mutually exclusive with allowed_domains
    # user_location:
    #   type: approximate
    #   city: "London"
    #   country: "UK"
  web_fetch:
    enabled: false
    citations_enabled: false
    # max_uses: 3
    # max_content_tokens: 4096
    # allowed_domains: ["example.com"]
    # blocked_domains: ["tracking.example"]  # mutually exclusive with allowed_domains
```

**Environment Variables:**

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ANTHROPIC_BASE_URL`: Override the API endpoint

**Caching Options:**

The `cache_mode` setting controls how prompt caching is applied:

- `off`: No caching, even if global `prompt_caching` is enabled
- `prompt`: Caches tools, system prompt, and template content
- `auto`: Same as `prompt` (default)

The `cache_ttl` setting controls how long cached content persists:

- `5m`: Standard 5-minute cache (default)
- `1h`: Extended 1-hour cache (additional cost)

**Reasoning + Structured Outputs:**

`claude-opus-4-6` uses adaptive thinking by default. Use effort levels (`low`, `medium`, `high`,
`max`) or `auto` with `anthropic.reasoning`:

```yaml
anthropic:
  reasoning: "high"
```

Adaptive models default to `auto` (provider‑chosen) and do not accept explicit budgets.

Anthropic models using budget-based thinking default to **reasoning on** with a **1024 token budget**.
Use `anthropic.reasoning` to set a budget, map from effort aliases, or disable reasoning entirely:

```yaml
anthropic:
  reasoning: 16000 # Reasoning budget tokens (minimum: 1024)
```

- Disable reasoning with `reasoning: "0"`, `reasoning: "off"`, or `reasoning: false`.
- Budget models also accept `low`/`medium`/`high`/`max` to map to preset budgets.
- The reasoning budget must be less than `max_tokens`. If you set a budget that meets/exceeds
  `max_tokens`, fast-agent raises `max_tokens` so the budget fits.

You can also set reasoning per run using the model string:

- `sonnet?reasoning=4096`
- `anthropic.claude-4-5-sonnet-latest?reasoning=4096`
- `claude-opus-4-6?reasoning=auto`

**Structured output selection (Anthropic JSON schema vs tool_use):**

- Models that support the `structured-outputs-2025-11-13` feature default to JSON schema output
  (`structured_output_mode: json`). This mode **is compatible with reasoning**.
- Older models default to the legacy `tool_use` structured output flow. `tool_use` **is not compatible
  with reasoning** — fast-agent disables reasoning when tool-forced structured output is selected.

You can override the structured output mode explicitly:

```yaml
anthropic:
  structured_output_mode: auto # auto (default), json, or tool_use
```

Deprecated: `thinking_enabled` and `thinking_budget_tokens` are ignored. Use `reasoning`.

**Built-in Anthropic web tools (`web_search` + `web_fetch`):**

fast-agent can enable Anthropic server-side web tools directly (these are not MCP tool calls):

- `anthropic.web_search.enabled: true`
- `anthropic.web_fetch.enabled: true`

Optional controls:

- `max_uses`
- `allowed_domains` / `blocked_domains` (mutually exclusive)
- `web_search.user_location` (approximate city/region/country/timezone)
- `web_fetch.max_content_tokens`
- `web_fetch.citations_enabled`

You can override per run in the model string:

- `claude-opus-4-6?web_search=on&web_fetch=on`
- `sonnet?web_search=off`

Supported values are `on`/`off` (also accepts `true`/`false`, `1`/`0`).

Version policy is model-aware:

- Claude 4.6 models use `web_search_20260209` and `web_fetch_20260209`
  (with required beta header `code-execution-web-tools-2026-02-09`).
- Other supported Anthropic models use legacy versions
  (`web_search_20250305`, `web_fetch_20250910`).


**Model Name Aliases:**

--8<-- "_generated/model_aliases_anthropic.md"

## OpenAI

**fast-agent** supports OpenAI `gpt-5` series, `gpt-4.1` series, `o1-preview`, `o1` and `o3-mini` models. Arbitrary model names are supported with `openai.<model_name>`. Supported modalities are model-dependent, check the [OpenAI Models Page](https://platform.openai.com/docs/models) for the latest information.

OpenAI multimodal models support text, images, and PDF input (`application/pdf`). For PDFs, provide a local file/blob rather than a URL.

For reasoning models, you can specify `low`, `medium`, or `high` effort as follows:

```bash
fast-agent --model o3-mini.medium
fast-agent --model gpt-5.high
```

`gpt-5` also supports a `minimal` reasoning effort.

Structured outputs use the OpenAI API Structured Outputs feature.

**YAML Configuration:**

```yaml
openai:
  api_key: "your_openai_key" # Default
  base_url: "https://api.openai.com/v1" # Default, only include if required
```

**Environment Variables:**

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

--8<-- "_generated/model_aliases_openai.md"

## Responses (OpenAI Responses API)

Use the `responses` provider for OpenAI Responses API models (for example `gpt-5`, `o3`, `o4-mini`).

```yaml
responses:
  api_key: "your_openai_key"
  base_url: "https://api.openai.com/v1" # Optional override
  reasoning: "medium" # Optional default
  text_verbosity: "medium" # Optional default for supporting models
  transport: "sse" # sse | websocket | auto
  web_search:
    enabled: false
    tool_type: web_search # web_search | web_search_preview
    # search_context_size: medium # low | medium | high
    # allowed_domains: ["openai.com", "docs.openai.com"]
    # external_web_access: false # only applies to tool_type=web_search
    # user_location:
    #   type: approximate
    #   city: "Minneapolis"
    #   region: "Minnesota"
    #   country: "US"
    #   timezone: "America/Chicago"
```

Per-run override via model string is also supported:

- `responses.gpt-5-mini?web_search=on`
- `responses.gpt-5-mini?web_search=off`
- `responses.gpt-5.3-codex?transport=ws`

Websocket transport is available for all models used through the `responses` provider. When
websocket transport is active, follow-up turns may be sent incrementally for efficiency.


## Codex (OAuth Responses)

**`fast-agent`** supports using your OpenAI Codex subscription. Run `fast-agent auth codexplan`
once, then use a Codex OAuth model alias such as `codexplan` (GPT-5.3 Codex) or
`codexplan52` (GPT-5.2 Codex).

**Quick Start:**

```bash
# Start OAuth login (stores tokens in your OS keyring)
fast-agent auth codexplan

# Use the Codex planning model
fast-agent --model codexplan

# Use the GPT-5.2 Codex planning model via OAuth
fast-agent --model codexplan52
```

**Provider Configuration:**

```yaml
codexresponses:
  # Optional: override defaults
  base_url: "https://chatgpt.com/backend-api/codex"
  text_verbosity: "medium"  # low | medium | high
  web_search:
    enabled: false
  default_headers:
    X-Custom-Header: "value"
```

**Environment Variables:**

- `CODEX_API_KEY`: Optional. Provide a Codex OAuth access token directly.

**Notes:**

- Tokens are stored in your OS keyring via `fast-agent auth codexplan`.
- `codexplan` maps to `codexresponses.gpt-5.3-codex` and `codexplan52` maps to
  `codexresponses.gpt-5.2-codex`; both use the same stored OAuth token.
- To remove tokens, use: `fast-agent auth codex-clear`.
- `fast-agent check` and `fast-agent auth` show Codex OAuth status.
- Encrypted reasoning is not transferable between API keys/credentials. Remove reasoning traces if transporting between sessions (use the bundled session skill).

**Model Name Aliases:**

--8<-- "_generated/model_aliases_codexresponses.md"

## Open Responses

Open Responses is an open standard for interoperable LLM interfaces. Read more at [https://www.openresponses.org/](https://www.openresponses.org/).

Use the provide string `openresponses` to select a model:

```bash
fast-agent --model openresponses.openai/gpt-oss-120b:groq
```

The default reasoning effort is `medium`. Configure other levels in your YAML:

```yaml
openresponses:
  reasoning_effort: "high"  # Options: minimal, low, medium, high
```

**YAML Configuration:**

```yaml
openresponses:
  api_key: "your_api_key"
  base_url: "https://api.example.com"  # Your Open Responses endpoint
  reasoning_effort: "medium"  # Default reasoning effort level
  default_headers:  # Optional custom headers
    X-Custom-Header: "value"
```

**Environment Variables:**

- `OPENRESPONSES_API_KEY`: Your API key
- `OPENRESPONSES_BASE_URL`: Override the API endpoint

**Model Name Format:**

Use `openresponses.<model_name>` to specify models, where `<model_name>` is the model identifier supported by your Open Responses endpoint.

## Hugging Face

Use models via [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index).

```yaml
hf:
  api_key: "${HF_TOKEN}"
  base_url: "https://router.huggingface.co/v1" # Default
  default_provider: # Optional: groq, fireworks-ai, cerebras, etc.
```

**Environment Variables:**

- `HF_TOKEN` - HuggingFace authentication token (required)
- `HF_DEFAULT_PROVIDER` - Default inference provider (optional)

### Model Syntax

Use `hf.<model_name>[:provider]` to specify models. If no provider is specified, the model is auto-routed.

**Examples:**

```bash
# Auto-routed
fast-agent --model hf.openai/gpt-oss-120b
fast-agent --model hf.moonshotai/kimi-k2-instruct-0905

# Explicit provider
fast-agent --model hf.moonshotai/kimi-k2-instruct-0905:groq
fast-agent --model hf.deepseek-ai/deepseek-v3.1:fireworks-ai
```

### Kimi K2.5 Instant Mode

Kimi K2.5 supports an **instant** toggle that disables reasoning when enabled. Use the
`instant` query parameter with the Kimi 2.5 model string:

```bash
fast-agent --model "hf.moonshotai/Kimi-K2.5?instant=on"  # thinking disabled
fast-agent --model "hf.moonshotai/Kimi-K2.5?instant=off" # thinking enabled
```

### Finding Available Providers

If you have a Hugging Face model ID (for example, `moonshotai/Kimi-K2-Thinking`) and want to see which Inference Providers are available, use `lookup_inference_providers` from `fast_agent.llm` (or `lookup_inference_providers_sync` for non-async code).

### Model Aliases

Aliased models are verified and tested to work with Structured Outputs and Tool Use. Functionality may vary between providers, or be clamped in some situations.

--8<-- "_generated/model_aliases_hf.md"

**Using Aliases:**

```bash
fast-agent --model kimi
fast-agent --model deepseek31
fast-agent --model kimi:together # provider can be specified with alias
```

### MCP Server Connections

`HF_TOKEN` is **automatically** applied when connecting to HuggingFace MCP servers.

**Supported domains:**

- `hf.co` / `huggingface.co` - Uses `Authorization: Bearer {HF_TOKEN}`
- `*.hf.space` - Uses `X-HF-Authorization: Bearer {HF_TOKEN}`

**Examples:**

```yaml
# fastagent.config.yaml
mcp:
  servers:
    huggingface:
      url: "https://huggingface.co/mcp"
      # HF_TOKEN automatically applied!
```

```bash
# Command line - HF_TOKEN automatically applied
fast-agent --model kimi --url https://hf.co/mcp
fast-agent --url https://my-space.hf.space/mcp
```

## Azure OpenAI

### ⚠️ Check Model and Feature Availability by Region

Before deploying an LLM model in Azure, **always check the official Azure documentation to verify that the required model and capabilities (vision, audio, etc.) are available in your region**. Availability varies by region and by feature. Use the links below to confirm support for your use case:

**Key Capabilities and Official Documentation:**

- **General model list & region availability:**
  [Azure OpenAI Service models – Region availability (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?utm_source=chatgpt.com)
- **Vision (GPT-4 Turbo with Vision, GPT-4o, o1, etc.):**
  [How-to: GPT with Vision (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?utm_source=chatgpt.com)
- **Audio / Whisper:**
  [The Whisper model from OpenAI (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/whisper-overview?utm_source=chatgpt.com)
  [Audio concepts in Azure OpenAI (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/audio?utm_source=chatgpt.com)
- **PDF / Documents:**
  [Azure AI Foundry feature availability across clouds regions (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-foundry/reference/region-support?utm_source=chatgpt.com)

**Summary:**

- **Vision (multimodal):** Models like GPT-4 Turbo with Vision, GPT-4o, o1, etc. are only available in certain regions. In the Azure Portal, the "Model deployments" → "Add deployment" tab lists only those available in your region. See the linked guide for input limits and JSON output.
- **Audio / Whisper:** There are two options: (1) Azure OpenAI (same `/audio/*` routes as OpenAI, limited regions), and (2) Azure AI Speech (more regions, different billing). See the links for region tables.
- **PDF / Documents:** Azure OpenAI does not natively process PDFs. Use [Azure AI Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/form-recognizer/) or [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/) for document processing. The AI Foundry table shows where each feature is available.

**Conclusion:** Before deploying, verify that your Azure resource's region supports the required model and features. If not, create the resource in a supported region or wait for general availability.

Azure OpenAI provides all the capabilities of OpenAI models within Azure's secure and compliant cloud environment. fast-agent supports three authentication methods:

1. Using `resource_name` and `api_key` (standard method)
2. Using `base_url` and `api_key` (for custom endpoints or sovereign clouds)
3. Using `base_url` and DefaultAzureCredential (for managed identity, Azure CLI, etc.)

**YAML Configuration:**

```yaml
# Option 1: Standard configuration with resource_name
azure:
  api_key: "your_azure_openai_key" # Required unless using DefaultAzureCredential
  resource_name: "your-resource-name" # Resource name (do NOT include if using base_url)
  azure_deployment: "deployment-name" # Required - the model deployment name
  api_version: "2023-05-15" # Optional, default shown
  default_headers:
    Ocp-Apim-Subscription-Key: "${AZURE_OPENAI_API_KEY}"
  # Do NOT include base_url if you use resource_name

# Option 2: Custom endpoint with base_url
azure:
  api_key: "your_azure_openai_key"
  base_url: "https://your-resource-name.openai.azure.com" # Full endpoint URL
  azure_deployment: "deployment-name"
  api_version: "2023-05-15" # Optional
  # Do NOT include resource_name if you use base_url

# Option 3: Using DefaultAzureCredential (requires azure-identity package)
azure:
  use_default_azure_credential: true
  base_url: "https://your-resource-name.openai.azure.com"
  azure_deployment: "deployment-name"
  api_version: "2023-05-15" # Optional
  # Do NOT include api_key or resource_name when using DefaultAzureCredential
```

**Important Configuration Notes:**
- Use either `resource_name` or `base_url`, not both.
- When using `DefaultAzureCredential`, do NOT include `api_key` or `resource_name`.
- When using `base_url`, do NOT include `resource_name`.
- When using `resource_name`, do NOT include `base_url`.
- `default_headers` can be used with any option (for example, APIM subscription keys).

**Environment Variables:**

- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Override the API endpoint

**Model Name Format:**

Use `azure.deployment-name` as the model string, where `deployment-name` is the name of your Azure OpenAI deployment.


## Groq

Groq is supported for Structured Outputs and Tool Calling, and has been tested with `moonshotai/kimi-k2-instruct`, `qwen/qwen3-32b` and `deepseek-r1-distill-llama-70b`.

**YAML Configuration:**

```yaml
groq:
  api_key: "your_groq_api_key"
  base_url: "https://api.groq.com/openai/v1"
```

**Environment Variables:**

- `GROQ_API_KEY`: Your Groq API key
- `GROQ_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

--8<-- "_generated/model_aliases_groq.md"


## DeepSeek

DeepSeek v3 is supported for Text and Tool calling.

**YAML Configuration:**

```yaml
deepseek:
  api_key: "your_deepseek_key"
  base_url: "https://api.deepseek.com/v1"
```

**Environment Variables:**

- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `DEEPSEEK_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

--8<-- "_generated/model_aliases_deepseek.md"


## Google

Google is natively supported in `fast-agent` using the Google genai libraries.

**YAML Configuration:**

```yaml
google:
  api_key: "your_google_key"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
```

**Environment Variables:**

- `GOOGLE_API_KEY`: Your Google API key

**Model Name Aliases:**

--8<-- "_generated/model_aliases_google.md"

### OpenAI Mode

You can also access Google via the OpenAI Provider. Use `googleoai` in the YAML file, or `GOOGLEOAI_API_KEY` for API KEY access.

## XAI Grok

XAI Grok 3, Grok 4 and Grok 4 Fast are available through the XAI Provider.

**YAML Configuration:**

```yaml
xai:
  api_key: "your_xai_key"
  base_url: "https://api.x.ai/v1"
```

**Environment Variables:**

- `XAI_API_KEY`: Your Grok API key
- `XAI_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

--8<-- "_generated/model_aliases_xai.md"


## Generic OpenAI / Ollama


Models prefixed with `generic` will use a generic OpenAI endpoint, with the defaults configured to work with Ollama [OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md).

This means that to run Llama 3.2 latest you can specify `generic.llama3.2:latest` for the model string, and no further configuration should be required.


!!! warning

    The generic provider is tested for tool calling and structured generation with `qwen2.5:latest` and `llama3.2:latest`. Other models and configurations may not work as expected - use at your own risk.


**YAML Configuration:**

```yaml
generic:
  api_key: "ollama" # Default for Ollama, change as needed
  base_url: "http://localhost:11434/v1" # Default for Ollama
```

**Environment Variables:**

- `GENERIC_API_KEY`: Your API key (defaults to `ollama` for Ollama)
- `GENERIC_BASE_URL`: Override the API endpoint

**Usage with other OpenAI API compatible providers:**
By configuring the `base_url` and appropriate `api_key`, you can connect to any OpenAI API-compatible provider.

## OpenRouter

Uses the [OpenRouter](https://openrouter.ai/) aggregation service. Models are accessed via an OpenAI-compatible API. Supported modalities depend on the specific model chosen on OpenRouter.

Models *must* be specified using the `openrouter.` prefix followed by the full model path from OpenRouter (e.g., `openrouter.google/gemini-flash-1.5`).

!!! warning

    There is an issue with between OpenRouter and Google Gemini models causing large Tool Call block content to be removed.


**YAML Configuration:**

```yaml
openrouter:
  api_key: "your_openrouter_key" # Required
  base_url: "https://openrouter.ai/api/v1" # Default, only include to override
```

**Environment Variables:**

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

OpenRouter does not use aliases in the same way as Anthropic or OpenAI. You must always use the `openrouter.provider/model-name` format.

## TensorZero Integration

[TensorZero](https://tensorzero.com/) is an open-source framework for building production-grade LLM applications. It unifies an LLM gateway, observability, optimization, evaluations, and experimentation into a single, cohesive system.

**Why Choose This Integration?**

While `fast-agent` can connect directly to many LLM providers, integrating with TensorZero offers powerful advantages for building robust, scalable, and maintainable agentic systems:

  * **Decouple Your Agent from Models:** Define task-specific "functions" (e.g., `summarizer`, `code_generator`) in TensorZero. Your `fast-agent` code calls these simple functions, while TensorZero handles the complexity of which model or provider to use. You can swap `GPT-4o` for `Claude 3.5 Sonnet` on the backend without changing a single line of your agent's code.
  * **Effortless Fallbacks & Retries:** Configure sophisticated failover strategies. If your primary model fails or is too slow, TensorZero can automatically retry with a different model or provider, making your agent far more resilient.
  * **Advanced Prompt Management:** Keep your complex system prompts and configurations in TensorZero's templates, not hardcoded in your Python strings. This cleans up your agent logic and allows for easier experimentation.
  * **Unified Observability:** All inference calls from your agents are logged, cached, and analyzed in one place, giving you a powerful, centralized view of your system's performance and costs.

**Getting Started: The `quickstart` Command**

The fastest way to get started is with the built-in, self-contained example. From your terminal, run:

```bash
fast-agent quickstart tensorzero
```

This command will create a new `tensorzero/` directory containing a fully dockerized project that includes:

1.  A pre-configured **TensorZero Gateway**.
2.  A custom **MCP Server** for your agent to use.
3.  Support for multimodal inputs using a **MiniIO** service.
4.  An interactive **`fast-agent`** that is ready to run by invoking `make agent`.

Just follow the "Next Steps" printed in your terminal to launch the agent.

**How it Works**

The `fast-agent` implementation uses TensorZero's OpenAI-compatible inference API. To call a "function" defined in your TensorZero configuration (e.g., in `tensorzero.toml`), simply specify it as the model name, prefixed with `tensorzero.`:

```bash
# Example from the quickstart Makefile
uv run agent.py --model=tensorzero.test_chat
```

By leveraging the common OpenAI interface, the integration remains simple and benefits from the extensive work done to support OpenAI-based models and features within both `fast-agent` and TensorZero.

TensorZero is an [Apache 2.0 licensed project](https://github.com/sproutfi/tensorzero?tab=License-1-ov-file) and you can find more details in the [official documentation](https://www.tensorzero.com/docs).

**YAML Configuration**

By default, the TenzorZero Gateway runs on `http://localhost:3000`. You can override this by specifying the `base_url` in your configuration.

```yaml
tensorzero:
  base_url: "http://localhost:3000" # Optional, only include to override
```

**Environment Variables:**

None (model provider credentials should be provided to the TensorZero Gateway instead)

## Aliyun

Tongyi Qianwen is a large-scale language model independently developed by Alibaba Cloud, featuring strong natural language understanding and generation capabilities. It can answer various questions, create written content, express opinions, and write code, playing a role in multiple fields.

**YAML Configuration:**

```yaml
aliyun:
  api_key: "your_aliyun_key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**Environment Variables:**

- `ALIYUN_API_KEY`: Your Aliyun API key
- `ALIYUN_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

--8<-- "_generated/model_aliases_aliyun.md"

## AWS Bedrock

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
