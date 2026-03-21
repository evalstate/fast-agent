# Agent Client Protocol

**`fast-agent`** has comprehensive support for Zed Industries [Agent Client Protocol](https://zed.dev/acp). 

Why use **`fast-agent`**?:

- Robust, native LLM Provider infrastructure, with Streaming and Structured outputs.
- Comprehensive MCP and Agent Skills support, including Tool Progress Notifications and Sampling.
- Build custom, multi-agent experiences in a few lines of code.

## Features

| Feature | Support | Notes |
|---------|---------|-------|
| Modes   | ✅ | Each defined Agent appears as a Modes |
| Tool / Workflow Progress | ✅ | MCP Tool Progress and Agent Workflow Progress updates |
| Agent Plan | ✅ | Iterative Planner reports progress using [Agent Plan](https://agentclientprotocol.com/protocol/agent-plan) |
| Cancellation | ✅  | LLM Streaming Cancellation |
| Multimodal | ✅ | Support for Images  | 
| Slash Commands | ✅ | Save, Load, Status and Clear/Clear Last message |
| File System / Terminal | ✅ | Start with `-x` option to enable access to Client terminal |
| MCP Servers | ⚠️ | Add via command line switches or configuration file |
| Sessions | ⚠️ | Use `save` and `load` slash commands. Plan to implement with [Session List](https://agentclientprotocol.com/rfds/session-list) |


## Getting Started

## Customizing Agents

See [Customizing Agents](customizing_agents.md) for ACP-aware agents, slash commands, and controlling the ACP Mode display name/description.

### No Install Quick Start:
To try it out straight away with your Client, set an API Key environment variable and add:

**Hugging Face**

export HF_TOKEN=hf_.......

`uvx fast-agent-acp@latest --model <your_model> [e.g. kimi]` 

**Open AI**

export OPENAI_API_KEY=......

`uvx fast-agent-acp@latest  --model <your_model> [e.g. gpt-5-mini.low]` 

**Anthropic**

export ANTHROPIC_API_KEY=......

`uvx fast-agent-acp@latest --model <your_model> e.g. [sonnet]` 

Tip: Use `uvx fast-agent-acp check` to help diagnose issues.

The [default system prompt](../agents/instructions.md) will read `AGENTS.md` if present. Use `/status system` to check.

Note: OAuth keys are stored in your keyring, so `check` may prompt to read the credential store.

An example Zed configuration is:

```json
...
"agent_servers": {
    "fast-agent-uvx": {
        "command": "uvx",
        "args": [
        "fast-agent-acp@latest",
        "--model",
        "kimi",
        "-x",
        "--url",
        "https://huggingface.co/mcp"
        ],
        "env": { "HF_TOKEN": "hf_xxxxxxxxxxx" }
    }
}

```

### Hugging Face Inference ACP (`hf-inference-acp`)

`hf-inference-acp` is a dedicated ACP agent for Hugging Face Inference Providers (built on `fast-agent-mcp`).

- Run: `uvx hf-inference-acp@latest`
- Setup mode (when `HF_TOKEN` is missing) includes: `/set-model` (lists suggested models when called with no args), `/login`, `/check`
- `/set-model` accepts `alias`, `hf.<org>/<model>[:provider]`, or `<org>/<model>` (auto-adds `hf.` and can display available providers)

### Installing 

`uv tool install -U fast-agent-mcp`

The ACP Server can then be started with the `fast-agent-acp` command. Custom agents can be started with `uv run <agent.py> --transport acp`.

For example:

`fast-agent-acp -x --model kimi --url https://huggingface.co/mcp --auth ${HF_TOKEN}` 

Starts an ACP Agent, with shell access and access to the Hugging Face MCP Server.

Documentation in Progress.

## Shell and File Access

**`fast-agent`** adds the read and write tools from the Client to enable "follow-along" functionality.

## Permissions

Tool calls in ACP mode prompt for permission by default. You will see options for Allow Once / Always Allow / Reject Once / Never Allow.

- Disable prompts entirely with `fast-agent-acp --no-permissions` (all tools are allowed).
- Persistent “Always” decisions are stored in the fast-agent environment directory (default `.fast-agent/auths.md`) so you can audit or edit them later. The file is only created when you choose an “Always” option.
- “Once” decisions are remembered only for the current session and are not written to disk. Removing the permissions file clears any saved Always rules. Use `fast-agent --env <path>` or `environment_dir` in the config file to relocate the environment folder.

### `--noenv` in ACP mode

Use `fast-agent-acp --noenv` (or `fast-agent serve --transport acp --noenv`) for ephemeral runs.

- Session persistence and resume are disabled.
- Slash session operations are disabled.
- Permission-store writes (`auths.md`) are disabled.

Conflicts (fail fast):

- `--noenv` + `--env`
- `--noenv` + `--resume`
