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
