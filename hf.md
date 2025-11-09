# Hugging Face Integration

`fast-agent` provides comprehensive integration with Hugging Face, supporting both MCP server connections and model inference via [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index).

## Authentication with HF_TOKEN

Set the `HF_TOKEN` environment variable to enable authentication:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

This single token is used for:
1. **MCP Server Connections** - Automatic authentication for HuggingFace MCP servers
2. **Model Inference** - Authentication for HuggingFace Inference API

## MCP Server Connections

### Automatic Authentication

`fast-agent` automatically adds authentication headers when connecting to HuggingFace MCP servers. No additional configuration required!

**Supported Domains:**
- `hf.co` (official short domain)
- `huggingface.co` (main domain)
- `*.hf.space` (HuggingFace Spaces)

**Configuration:**

```yaml
# In fastagent.config.yaml
mcp:
  servers:
    huggingface:
      url: "https://huggingface.co/mcp?login"
      # HF_TOKEN automatically applied - no auth config needed!

    my-space:
      url: "https://my-custom-space.hf.space/mcp"
      # HF_TOKEN automatically applied here too!
```

**Command Line:**

```bash
# HF_TOKEN automatically applied to HuggingFace URLs
fast-agent go --url https://hf.co/mcp
fast-agent go --url https://huggingface.co/mcp?login
fast-agent go --url https://my-space.hf.space/mcp
```

**Authentication Headers:**
- `hf.co` / `huggingface.co` → `Authorization: Bearer {HF_TOKEN}`
- `*.hf.space` → `X-HF-Authorization: Bearer {HF_TOKEN}`

**Security Features:**
- Only applies authentication to legitimate HuggingFace domains
- Never overrides existing Authorization headers
- Validates `.hf.space` domains to prevent spoofing attacks
- Space name validation (non-empty, valid format)

## Model Inference

Use models via HuggingFace Inference Providers with the `hf.` prefix.

### Model Syntax

**Format:** `hf.<model_name>[:provider]`

- If no provider is specified, the model is auto-routed to an available provider
- Provider can be specified explicitly: `groq`, `fireworks-ai`, `cerebras`, etc.

### Examples

```bash
# Auto-routed models
fast-agent --model hf.openai/gpt-oss-120b
fast-agent --model hf.moonshotai/kimi-k2-instruct-0905
fast-agent --model hf.deepseek-ai/deepseek-v3.1

# Explicit provider selection
fast-agent --model hf.moonshotai/kimi-k2-instruct-0905:groq
fast-agent --model hf.deepseek-ai/deepseek-v3.1:fireworks-ai
fast-agent --model hf.openai/gpt-oss-120b:cerebras
```

### Model Aliases

For convenience, several model aliases are pre-configured and tested to work with Structured Outputs and Tool Use. Note that functionality may vary between providers.

| Alias         | Full Model Name                       |
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
fast-agent --model kimi          # Auto-routed Kimi K2
fast-agent --model deepseek31    # Auto-routed DeepSeek v3.1
fast-agent --model gpt-oss       # Auto-routed GPT-OSS-120b
```

### Configuration

Add HuggingFace settings to your `fastagent.config.yaml` or `fastagent.secrets.yaml`:

```yaml
huggingface:
  api_key: "${HF_TOKEN}"  # Reference HF_TOKEN environment variable
  base_url: "https://router.huggingface.co/v1"  # Default, override only if needed
  default_provider: "groq"  # Optional: default provider for all HF models
```

**Authentication Priority:**
1. `huggingface.api_key` in config file
2. `HF_TOKEN` environment variable
3. Error if neither is set

**Provider Selection Priority:**
1. Explicit in model string (e.g., `model:groq`)
2. `huggingface.default_provider` in config
3. `HF_DEFAULT_PROVIDER` environment variable
4. Auto-routing by HuggingFace

### Python API

```python
from fast_agent import FastAgent

fast = FastAgent("HF Example")

@fast.agent(
    instruction="You are a helpful assistant",
    model="hf.moonshotai/kimi-k2-instruct-0905:groq"
)
async def main():
    async with fast.run() as agent:
        response = await agent("What is the capital of France?")
        print(response)
```

**Using Aliases:**

```python
@fast.agent(
    instruction="You are a helpful assistant",
    model="kimi"  # Uses alias
)
```

## Environment Variables

| Variable               | Purpose                              | Required |
|------------------------|--------------------------------------|----------|
| `HF_TOKEN`             | HuggingFace authentication token     | Yes      |
| `HF_DEFAULT_PROVIDER`  | Default inference provider           | No       |

## Model Capabilities

The following models are verified with specific capabilities:

| Model                              | Context  | Output  | Structured Output | Reasoning |
|------------------------------------|----------|---------|-------------------|-----------|
| `moonshotai/kimi-k2-instruct-0905` | 262K     | 16K     | ✓ (object mode)   | ✗         |
| `openai/gpt-oss-120b`              | 131K     | 32K     | ✓ (object mode)   | ✗         |
| `openai/gpt-oss-20b`               | 131K     | 32K     | ✓ (object mode)   | ✗         |
| `deepseek-ai/deepseek-v3.1`        | 163K     | 8K      | ✗                 | ✗         |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 262K     | 8K      | ✗                 | ✗         |
| `zai-org/GLM-4.6`                  | 202K     | 8K      | ✓ (object mode)   | ✓ (tags)  |
| `MiniMaxAI/MiniMax-M2`             | 202K     | 8K      | ✓ (object mode)   | ✓ (tags)  |

**Notes:**
- Context and output values are in tokens
- Structured Output modes: `schema` (JSON Schema), `object` (generic JSON)
- Reasoning modes: `tags` (enclosed in `<thinking>` tags), `openai` (native reasoning)

## Complete Example

```yaml
# fastagent.config.yaml
mcp:
  servers:
    huggingface:
      url: "https://huggingface.co/mcp?login"
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]

huggingface:
  api_key: "${HF_TOKEN}"
  default_provider: "groq"
```

```python
# agent.py
import asyncio
from fast_agent import FastAgent

fast = FastAgent("HF Agent")

@fast.agent(
    "researcher",
    "Research topics and provide comprehensive summaries",
    servers=["huggingface", "fetch"],
    model="kimi"  # Uses Kimi K2 via default provider (groq)
)
async def main():
    async with fast.run() as agent:
        await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
# Set token and run
export HF_TOKEN="hf_your_token_here"
uv run agent.py
```

## Troubleshooting

### Authentication Errors

**Problem:** `401 Unauthorized` when connecting to HF MCP servers

**Solution:**
```bash
# Verify token is set
echo $HF_TOKEN

# Test with command line
fast-agent go --url https://hf.co/mcp
```

### Provider Not Available

**Problem:** Model request fails with provider unavailable

**Solution:**
- Try a different provider explicitly: `hf.model:groq`, `hf.model:fireworks-ai`
- Check HuggingFace provider status: https://huggingface.co/docs/inference-providers
- Use auto-routing by omitting provider suffix

### Token Scope

**Problem:** MCP connection fails despite valid token

**Solution:**
- Verify token has required scopes (check HuggingFace token settings)
- Some MCP servers may require specific permissions
- Try regenerating token with appropriate scopes

## Security Notes

- **Never commit `HF_TOKEN`** to version control
- Store in `fastagent.secrets.yaml` (add to `.gitignore`)
- Use environment variable references: `api_key: "${HF_TOKEN}"`
- Token is only sent to legitimate HuggingFace domains
- Anti-spoofing validation prevents token leakage to malicious domains

## Resources

- [HuggingFace Inference Providers](https://huggingface.co/docs/inference-providers/en/index)
- [HuggingFace MCP Documentation](https://huggingface.co/docs/hub/mcp)
- [Create HF Token](https://huggingface.co/settings/tokens)
- [fast-agent Documentation](https://fast-agent.ai)
