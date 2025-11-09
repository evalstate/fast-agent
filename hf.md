## Hugging Face

Use models via [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index).

```yaml
huggingface:
  api_key: "${HF_TOKEN}"
  base_url: "https://router.huggingface.co/v1"  # Default
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

### Model Aliases

Aliased models are verified and tested to work with Structured Outputs and Tool Use. Functionality may vary between providers.

| Alias         | Maps to                               |
|---------------|---------------------------------------|
| `kimi`        | `hf.moonshotai/Kimi-K2-Instruct-0905` |
| `gpt-oss`     | `hf.openai/gpt-oss-120b`              |
| `gpt-oss-20b` | `hf.openai/gpt-oss-20b`               |
| `glm`         | `hf.zai-org/GLM-4.6`                  |
| `qwen3`       | `hf.Qwen/Qwen3-Next-80B-A3B-Instruct` |
| `deepseek31`  | `hf.deepseek-ai/DeepSeek-V3.1`        |
| `minimax`     | `hf.MiniMaxAI/MiniMax-M2`             |

**Using Aliases:**

```bash
fast-agent --model kimi
fast-agent --model deepseek31
```

### MCP Server Connections

`HF_TOKEN` is **automatically** applied when connecting to HuggingFace MCP servers - no additional configuration needed!

**Supported domains:**
- `hf.co` / `huggingface.co` - Uses `Authorization: Bearer {HF_TOKEN}`
- `*.hf.space` - Uses `X-HF-Authorization: Bearer {HF_TOKEN}`

**Examples:**

```yaml
# fastagent.config.yaml
mcp:
  servers:
    huggingface:
      url: "https://huggingface.co/mcp?login"
      # HF_TOKEN automatically applied!
```

```bash
# Command line - HF_TOKEN automatically applied
fast-agent go --url https://hf.co/mcp
fast-agent go --url https://my-space.hf.space/mcp
```
