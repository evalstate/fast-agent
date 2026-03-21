---
title: mcp-ui and fast-agent
---

## Using mcp-ui and `fast-agent`

**`fast-agent`** supports  [mcp-ui](https://mcpui.dev/) embedded components, and makes them accessible for usage and testing. 

## Installing `fast-agent`

To install **`fast-agent`**, first download and install the [`uv`](https://docs.astral.sh/uv/) package manager.

Next, install (or upgrade) with:

```bash
uv tool install -U fast-agent-mcp
```

Next, configure your API Keys. This guide assumes that you have `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables set.

Check your installation by running the Model Context Protocol everything server with:

```bash
fast-agent --npx @modelcontextprotocol/server-everything
```

Use the `fast-agent check` command to diagnose any issues.

## Using `mcp-ui`

Download the mcp-ui examples, and start the TypeScript [demo server](https://github.com/idosal/mcp-ui/blob/main/examples/typescript-server-demo/README.md):

To connect to the demo server with `gpt-5-mini` with low reasoning effort use:

```bash
fast-agent --url http://localhost:3000 --model=gpt-5-mini.low
```

![type:video](pics/mcp-ui-output.mp4)


**`fast-agent`** presents the mcp-ui content as links beneath the assistant message. HTML components are stored in the fast-agent environment directory (default `.fast-agent/ui`). Use `fast-agent --env <path>` or `environment_dir` in the config file to relocate this folder.

If you want to test multiple models in parallel - for example to compare behaviour - you can specify more than one model and run in parallel:

```bash
# run the test server with both gpt-5-mini and sonnet
fast-agent --url http://localhost:3000 --model=gpt-5-mini.low,sonnet
```

To run with a prompt and exit:

```bash
fast-agent --url http://localhost:3000 --model=haiku -m "run all three tools"
```

If you want to pass authorization headers, you use the --auth option:

```bash
fast-agent --url https://huggingface.co/mcp --auth $HF_TOKEN
```

## Advanced Configuration

To create configuration files for advanced configuration, use the `fast-agent scaffold` command. 

This allows you to configure some `mcp-ui` settings, or configure servers to use custom headers.

The following options are available:

```yaml title="fastagent.config.yaml"
# mcp-ui config options

# Base directory for fast-agent runtime data
# environment_dir: ".fast-agent"

# Where to write MCP-UI HTML files (relative to CWD if not absolute)
mcp_ui_output_dir: ".fast-agent/ui"  

# "disabled", "enabled" or "auto" to automatically open links in browser
mcp_ui_mode: enabled

mcp:
  servers:
      example:
        transport: http
        url: https://huggingface.co/mcp
        ## custom headers below
        headers: 
          custom_header: value
```

## Client Spoofing

Some MCP Servers adjust their tools or behaviour based on the connecting client (for exampling enabling mcp-ui). You can specify the name and version to present to the MCP Server:

```yaml title="fastagent.config.yaml"
  servers:
      example:
        transport: http
        url: https://huggingface.co/mcp
        implementation:
            name: claude-code
            version: 1.0.99
```
