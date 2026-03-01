# hf-inference-acp

Hugging Face inference agent with ACP (Agent Client Protocol) support, powered by fast-agent-mcp.

## Installation

```bash
uvx hf-inference-acp
```

## What is this?

This package provides an ACP-compatible agent for Hugging Face Inference API. It allows you to use Hugging Face's Inference Providers through any ACP-compatible client (like Toad).

## Features

- **Setup Mode**: Configure Hugging Face credentials and model settings
- **Hugging Face Mode**: AI assistant powered by Hugging Face Inference API
- **HuggingFace MCP Server**: Built-in integration with Hugging Face's MCP server for accessing models, datasets, and spaces

## Quick Start

1. Run the agent:

   ```bash
   uvx hf-inference-acp
   ```

2. If `HF_TOKEN` is not set, you'll start in **Setup** mode with these commands:

   - `/login` - Get instructions for HuggingFace authentication
   - `/set-model <model>` - Set the default model
   - `/check` - Verify your configuration

3. Once authenticated (HF_TOKEN is set), you'll automatically start in **Hugging Face** mode.

4. In **Hugging Face** mode, use `/connect` to connect to the Hugging Face MCP server for model/dataset search tools.

## Curated Model Aliases

`/set-model` supports short aliases from fast-agent's curated Hugging Face list, including:

- `kimi`
- `glm`
- `minimax`
- `deepseek32`
- `kimi25`
- `qwen35` (thinking profile)
- `qwen35instruct` (instruct profile)

Qwen 3.5 aliases resolve to `hf.Qwen/Qwen3.5-397B-A17B:novita` with curated sampling defaults:

- `qwen35`: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`
- `qwen35instruct`: `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`

## Configuration

Configuration is stored at `~/.config/hf-inference/hf.config.yaml`:

```yaml
default_model: hf.moonshotai/Kimi-K2-Instruct-0905

mcp:
  servers:
    huggingface:
      url: "https://huggingface.co/mcp?login"
```

## Authentication

Set your HuggingFace token using one of these methods:

1. **Environment variable**:

   ```bash
   export HF_TOKEN=your_token_here
   ```

2. **HuggingFace CLI**:
   ```bash
   huggingface-cli login
   ```

Get your token from: https://huggingface.co/settings/tokens

## License

Apache License 2.0 - See the [main repository](https://github.com/evalstate/fast-agent) for details.

## More Information

For full documentation and the main project, visit: https://github.com/evalstate/fast-agent
