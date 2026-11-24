---
title: fast-agent go command
description: Describes how to use the fast-agent go command to quickly connect to
  and test MCP Servers, including STDIO, SSE and Streaming support.
social:
  title: fast-agent go
  tagline: Quickly connect to MCP servers over STDIO, SSE, and streaming transports.
  description: Quickly connect to MCP servers over STDIO, SSE, and streaming transports.
  alt: fast-agent social card — fast-agent go
---


## `fast-agent go` command

The `go` command allows you to run an interactive agent directly from the command line without
creating a dedicated agent.py file.

### Usage

```bash
fast-agent go [OPTIONS]
```

### Options

- `--name TEXT`: Name for the agent (default: "fast-agent")
- `--instruction`, `-i <path or uri>`: File name, HTTP(S) URL, `file://` URI, or `hf://` URI for [System Prompt](../agents/instructions/)
- `--config-path`, `-c <path or uri>`: Path, HTTP(S) URL, `file://` URI, or `hf://` URI to config file
- `--servers <server1>,<server2>`: Comma-separated list of server names to enable from config
- `--url TEXT`: Comma-separated list of HTTP/SSE URLs to connect to directly
- `--auth TEXT`: Bearer token for authorization with URL-based servers
- `--client-metadata-url TEXT`: OAuth Client ID Metadata Document URL for URL-based servers when dynamic client registration is not available
- `--model`, `--models <model_string>`: Select a model for this run (e.g., haiku, sonnet, gpt-4)
- `--model`, `--models <model1>,<model2>,...`: Run one agent per model in parallel and print a side-by-side comparison of responses
- `--model-url <url>`: Auto-configure a vLLM, llama.cpp, or generic OpenAI-compatible model endpoint for this run (bare origin or `.../v1`)
- `--model-auth <token>`: Bearer token for `--model-url` (optional `Bearer ` prefix; kept in memory only and separate from MCP `--auth`). Prefer `--model-api-key-env` (or `HF_TOKEN` / config) to avoid exposing tokens in shell history or process lists
- `--model-api-key-env <ENV_VAR>`: Environment variable name containing the API key for `--model-url` (preferred over `--model-auth` for avoiding argv visibility)
- `--environment`, `-E <name>`: Select a named execution environment from config
- `--pack`, `--card-pack <name>`: Ensure a named card pack is installed in the selected environment before starting
- `--pack-registry <path or uri>`: Marketplace file path, HTTP(S) URL, `file://` URI, or `hf://` URI used to resolve `--pack` when it is not already installed
- `--agent-cards`, `--card <path or uri>`: Load AgentCards as runnable agents from a path, HTTP(S) URL, `file://` URI, or `hf://` URI (repeatable)
- `--card-tool <path or uri>`: Load AgentCards from a path, HTTP(S) URL, `file://` URI, or `hf://` URI and attach them as tools to the selected/default agent (repeatable)
- `--agent <name>`: Target a specific loaded agent by name for `--message`, `--prompt-file`, and initial interactive mode
- `--message`, `-m TEXT`: Message to send to the agent (skips interactive mode)
- `--attach`, `-a <path or url>`: Attach a local file or HTTP(S) URL to a one-shot `--message` or `--prompt-file` run (repeatable). With `--prompt-file`, attachments are added to the last user message.
- `--json-schema <path or uri>`: Path, HTTP(S) URL, `file://` URI, or `hf://` URI to a JSON Schema file used for one-shot structured output
- `--schema-model <module.path:ClassName>`: Pydantic `BaseModel` import path used for one-shot structured output
- `--structured-tool-policy <auto|always|defer|no_tools>`: Control whether tools may be used with one-shot structured output
- `--results <path>`: Write the resulting history to a file, or per-model suffixed files in comparison mode
- `--trajectory-output <path>`: Write a self-contained ATIF v1.8 JSON trajectory for a one-shot run
- `--trajectory-format atif`: Select the direct trajectory format (currently ATIF)

For Harbor-style one-shot execution, keep the Fast-Agent result artifact and
the interoperability trajectory separate:

```bash
fast-agent go \
  --message "$INSTRUCTION" \
  --workspace . \
  --home /logs/agent/fast-agent-home \
  --shell \
  --results /logs/agent/fast-agent-results.json \
  --trajectory-output /logs/agent/trajectory.json
```

The trajectory is one self-contained `ATIF-v1.8` JSON document. Recorded
agent-as-tool invocations are embedded in `subagent_trajectories`; Harbor does
not need to inspect Fast-Agent session files or private message channels.
- `--workspace <path>`: Override the workspace root; when `--home` is omitted, the home defaults to `<workspace>/.fast-agent`
- `--home <path>`: Use `<path>` as the `.fast-agent` home itself (cards load from `<path>/agent-cards/`); relative paths resolve under the selected workspace
- `--no-home`: Run in ephemeral mode (disable implicit home card loading, session persistence/resume, and permission-store side effects)
- `--isolated`: Embedded one-shot mode. Config, secrets, credentials and model references are still read from the home, but nothing is written there (no sessions, no log file, no telemetry), and shell/filesystem tools, subagents, skills, AgentCards, plugins and hooks are disabled regardless of config. Only MCP servers named with `--servers` or passed via `--url`/`--stdio`/`--npx`/`--uvx` are attached. Requires `--message` or `--prompt-file`; combining it with options that would re-enable any of the above (`--shell`, `--card`, `--skills-dir`, `--resume`, `--no-home`, ...) is an error
- `--resume <id|latest>`: Resume the latest session (or a specific session id) and restore its saved model, web-search/web-fetch state, Delegate/Orchestrate/Harness capability mode, and per-agent shell availability without opening the startup picker. Saved shell access is restored only in the same local workspace; `--no-shell` disables it and `--shell` explicitly enables it. ACP-originated sessions do not inherit shell access. Older sessions without saved shell state require `--shell`. A session already active in another process is rejected; use `fast-agent session fork <id>` to branch its latest committed checkpoint.
- `--subagents`: Enable the built-in `subagent` tool for the selected or generated agent
- `--subagent-model <model_string>`: Enable built-in subagents and force every one to use this model
- `--image-gen` / `--no-image-gen`: Enable or disable the built-in `generate_image` and `edit_image` tools. When enabled without a model, matches the main model's provider (`codexresponses.*` → `codexresponses.gpt-image-2.6-flare`, `openai.*` → `openai.gpt-image-2.6-flare`, `xai.*` → `xai.grok-imagine-image-2.0`), or falls back to the first available credential (Codex OAuth → `OPENAI_API_KEY` → xAI OAuth/`XAI_API_KEY`)
- `--image-gen-model <model_string>` (or `--image-gen=<model_string>`): Enable built-in image generation with a specific model or provider shorthand (e.g. `codexresponses.gpt-image-2.6-sunburst`, `openai.gpt-image-2.6-flare`, `xai.grok-imagine-image-2.0`, `codex`, `openai`, `xai`)
- `--prompt-file`, `-p <path or uri>`: Path, HTTP(S) URL, `file://` URI, or `hf://` URI to a prompt file to use (either text or JSON)
- `--skills-dir`, `--skills <path>`: Override the default skills directory
- `--stdio "<command> <options>"`: Run the command to attach a STDIO server (enclose arguments in quotes)
- `--npx "@package/name <options>"`: Run an NPX package as a STDIO server (enclose arguments in quotes)
- `--uvx "@package/name <options>"`: Run an UVX package as a STDIO server (enclose arguments in quotes)
- `--shell`, `-x`: Enable a local shell runtime and expose shell tools (canonical `shell` under PTC) (bash or pwsh)
- `-xx`: Enable both local shell access and built-in subagents
- `--no-shell`: Disable local shell/filesystem tools, even when skills or agent config request them
- `--ptc` / `--no-ptc`: Enable or disable [programmatic tool calling](config_file.md#programmatic-tool-calling-ptc), overriding the model string, YAML configuration and model default. Local POSIX environments only.
- `--ptc-all`: Enable programmatic tool calling with all function stubs upfront instead of deferring them behind `search_tools()`. Cannot be combined with `--no-ptc`.
- `--reload`: Enable manual AgentCard reloads with `/reload`
- `--watch`: Watch AgentCard paths and reload automatically
- `--timeout`, `-t <seconds>`: Maximum one-shot execution duration for `--message` or `--prompt-file` runs

--8<-- "_generated/ptc_model_defaults.md"

Global CLI options (apply to all subcommands):

- `--quiet`, `-q`: Disable progress display and logging
- `--verbose`, `-v`: Enable verbose mode

### Examples

Note - you may omit `go` when supplying command line options.

```bash
# Basic usage with interactive mode
fast-agent go --model=haiku

# Install or reuse a card pack, then launch immediately
fast-agent go --pack analyst --model haiku

# Resolve a pack from a specific marketplace file
fast-agent go --pack analyst --pack-registry ./marketplace.json --model haiku

# Multi-agent pack: target one specific agent
fast-agent go --pack analyst --agent planner --model haiku

# Basic usage with interactive mode (go omitted)
fast-agent --model haiku

# Use shell access and built-in subagents (go omitted)
fast-agent -xx --subagent-model haiku

# Enable built-in generate_image and edit_image tools (auto-selects image model)
fast-agent go --model codexresponses.gpt-6.2-sol --image-gen

# Pin a specific image generation model or provider shorthand
fast-agent go --model sonnet --image-gen=xai.grok-imagine-image-2.0

# Compare responses across multiple models (comparison mode)
fast-agent --models "kimi,gpt-5-mini?reasoning=low"

# Specifying servers from configuration
fast-agent go --servers=fetch,filesystem --model=haiku

# Directly connecting to HTTP/SSE servers via URLs
fast-agent go --url=http://localhost:8001/mcp,http://api.example.com/sse

# Connecting to an authenticated API endpoint
fast-agent go --url=https://api.example.com/mcp --auth=YOUR_API_TOKEN

# Run an NPX package directly
fast-agent --npx @modelcontextprotocol/server-everything 

# Non-interactive mode with a single message
fast-agent go --message="What is the weather today?" --model=haiku

# Unattended runs can also select a model through the environment
FAST_AGENT_MODEL=haiku fast-agent go --message="Summarize this report"

# Attach files or URLs to a one-shot message
fast-agent go --message "Summarize these" --attach ./report.pdf --attach https://example.com/chart.png --model=haiku

# Embedded run: home config is read, nothing is written, only the named MCP server is attached
fast-agent go --isolated --servers docs --message "Find the retry policy" --results ./out.json

# Machine-readable structured one-shot output
fast-agent go --no-home --model haiku --message "What is the weather in London?" --json-schema ./schema.json

# Use a prompt stored in Hugging Face Hub generic storage
fast-agent go --prompt-file hf://buckets/evalstate/home/demo.md --model=haiku

# Target one specific loaded agent when multiple agents are available
fast-agent go --agent-cards ./agents --agent researcher

# Send one message to a specific loaded agent and exit
fast-agent go --agent-cards ./agents --agent qa --message "run smoke checks"

# Using a prompt file
fast-agent go --prompt-file=my-prompt.txt --model=haiku

# Attach evidence to the last user message in a prompt file
fast-agent go --prompt-file=my-prompt.txt --attach ./evidence.pdf --model=haiku

# Specify a system prompt file
fast-agent go -i my_system_prompt.md

# Specify a skills directory (overrides default search paths)
fast-agent go --skills ~/my-skills/

# Provider LLM shell access (use at your own risk)
fast-agent go -x

# Run shell-enabled agents in a configured environment
fast-agent go --environment ubuntu -x

```

### Machine-readable JSON output

Use `--json-schema` when you need a strict scripting/automation contract from `fast-agent go`.

Rules:

- `--json-schema` must be combined with exactly one of:
  - `--message`
  - `--prompt-file`
- stdout contains only the final validated JSON document
- errors and diagnostics go to stderr
- this mode is not supported with multi-model fan-out

Recommended automation pattern:

- use `--no-home` to avoid implicit home side effects
- keep input one-shot with `--message` or `--prompt-file`
- provide a schema file that describes the exact payload you want

```bash
fast-agent go \
  --no-home \
  --model sonnet \
  --message "What is the weather in London?" \
  --json-schema ./schema.json
```

When you need exact JSON on stdout, prefer this mode over parsing chat output or exported histories.

### Comparison mode (multiple models)

Pass a comma-separated list to `--models` (or `--model`) to run one agent per model in parallel and compare responses side-by-side.

How it works:

- `--instruction` / `-i`, `--servers`, `--url`, and other connection options apply to every model agent.
- Each model string becomes a separate agent name in the output.
- Interactive mode (default): every prompt is sent to all models and results are shown in a comparison view.
- Non-interactive: use `--message` or `--prompt-file` to run once and print results for each model.
```bash
fast-agent go --models "sonnet,gpt-5-mini?reasoning=low"

# Route to one model agent directly (instead of side-by-side parallel output)
fast-agent go --models "sonnet,gpt-5-mini?reasoning=low" --agent sonnet --message "Summarize this"
```

### Agent targeting notes

- Use `--agent` when you loaded multiple agents (for example with `--agent-cards`).
- If `--instruction` points to a local file, fast-agent may derive an internal default agent name
  from the file name (for example `research.md` -> `research`). `--agent` still takes precedence
  for explicit targeting.
- Explicit targeting can include tool-only agents when needed for testing.

### AgentCards vs ToolCards

`tool-cards` are not a separate file format. They are still AgentCards.

- `--agent-cards`: load cards as normal runnable agents.
- `--card-tool`: load cards, then attach those loaded agents as tools to a parent agent.

By default, `fast-agent go` will auto-discover cards from your home when present:

- `<home>/agent-cards/` for runnable agents
- `<home>/tool-cards/` for cards to attach as tools

When `--no-home` is set, this implicit discovery is disabled. Explicit `--agent-cards` and
`--card-tool` values still work.

Cards loaded via `--agent-cards` / `--card-tool` can include `mcp_connect` entries;
those runtime MCP servers are resolved and attached automatically during startup.

See [AgentCards and ToolCards reference](../agents/defining/agent_cards/) for details and recommended layout.

### Card packs

Use `--pack` when you want fast-agent to install or reuse a named card pack in
the selected home and then launch immediately.

Typical flow:

1. Resolve the home from `--home`, `--workspace <path>/.fast-agent`, or normal fast-agent settings
2. Reuse the pack if it is already installed there
3. Otherwise resolve it from the configured marketplace (or `--pack-registry`)
4. Install it into the home
5. Continue through the normal `fast-agent go` startup path

Notes:

- `--pack` cannot be combined with `--no-home`
- `--model` is a fallback only; an explicit model declared in an AgentCard still takes precedence
- Some packs install multiple agents, so you may also want `--agent <name>`

### `--no-home` mode

Use `--no-home` when you want to run without implicit home side effects.

- Session persistence is disabled.
- Session resume is disabled.
- Default home card auto-loading is disabled.

Conflicts (fail fast):

- `--no-home` + `--home`
- `--no-home` + `--resume`
- `--no-home` + `--pack`

### URL Connection Details

The `--url` parameter allows you to connect directly to HTTP or SSE servers using URLs.

- URLs must have http or https scheme
- The transport type is determined by the URL path:
  - URLs ending with `/sse` are treated as SSE transport
  - URLs ending with `/mcp` or automatically appended with `/mcp` are treated as HTTP transport
- Server names are generated automatically based on the hostname, port, and path
- The URL-based servers are added to the agent's configuration and enabled

### Authentication

The `--auth` parameter provides authentication for URL-based MCP servers:

- When provided, it creates an `Authorization: Bearer TOKEN` header for all URL-based servers
- This is commonly used with API endpoints that require authentication
- Example: `fast-agent go --url=https://api.example.com/mcp --auth=12345abcde`

### Connecting directly to a model endpoint (`--model-url`)

Use `--model-url <url>` to auto-configure an OpenAI-compatible inference server without creating a config file or saved overlay:

```bash
# Dedicated Hugging Face Inference Endpoint (vLLM; requires https://)
fast-agent --model-url "https://<endpoint>.endpoints.huggingface.cloud"

# Read the endpoint API key from a specific environment variable (preferred over --model-auth)
fast-agent --model-url "https://<endpoint>.endpoints.huggingface.cloud" --model-api-key-env MY_ENDPOINT_KEY

# Pass a bearer token directly (optional "Bearer " prefix, kept in memory only)
fast-agent --model-url "https://<endpoint>.endpoints.huggingface.cloud" --model-auth "$HF_TOKEN"

# Combine with MCP --url / --auth without flag collision, and pass model query overrides
fast-agent --model-url http://localhost:8000 --model "?reasoning=medium" --url https://mcp.example.com/mcp --auth "$MCP_TOKEN"
```

How `--model-url` resolves the endpoint:

- **URL normalization and backend detection:** Accepts a bare origin or `.../v1` (schemeless `*.endpoints.huggingface.cloud` defaults to `https://`; embedded URL userinfo is rejected), probes `GET <base_url>/models`, and auto-detects **llama.cpp** (`owned_by == "llamacpp"` or `GET /props`), **vLLM** (`owned_by == "vllm"` or `max_model_len`, plus optional `GET /version`), or a **generic OpenAI-compatible** endpoint.
- **Multi-model selection:** When an endpoint serves multiple models, `--model-url` defaults to the first model marked `loaded`, or otherwise the first listed model. Pass `--model <model-id>` (or `--model "<model-id>?reasoning=medium"`) to choose a specific served model.
- **Model metadata and live limits:** Matches `(id, root, aliases)` against the built-in `ModelDatabase` while preserving the exact discovered wire model ID, and overrides `context_window` (and clamps live `max_output_tokens`) using live server truth (`max_model_len` on vLLM, `n_ctx` / `n_predict` on llama.cpp) inside an ephemeral in-memory overlay.
- **Fail-closed authentication precedence:**
  1. `--model-auth <token>` (in-memory only; never written to overlay YAML or logged; prefer `--model-api-key-env` to avoid argv/process-list visibility)
  2. `--model-api-key-env <ENV_VAR>` (fails closed if unset or empty)
  3. For `*.endpoints.huggingface.cloud` (requires `https://`): defaults provider to `hf` and resolves `HF_TOKEN` then merged `hf.api_key` settings (from `fastagent.secrets.yaml` or `fastagent.config.yaml`; never falls back to `~/.cache/huggingface/token`)
  4. For non-HF hosts: defaults provider to `generic`, using `GENERIC_API_KEY` on `https://` or loopback hosts (`localhost`, `127.0.0.1`, `::1`), or unauthenticated otherwise
- **Flag constraints:** `--model-url` cannot be combined with `--local`, `--base-url`, `--resume`, or comma-separated multi-model `--models` lists. `--model-auth` and `--model-api-key-env` are mutually exclusive and require `--model-url`.
