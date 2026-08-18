---
title: Using the TUI
description: Navigating and using fast-agent TUI features.
social:
  title: Using the TUI
  tagline: Navigating and using fast-agent TUI features.
  alt: fast-agent social card — Using the TUI
---

To start **`fast-agent`** interactively, use `fast-agent go`. 

When using common command line options, `go` is inferred:

```bash
# Start with a specific default model - note `go` does not need to be specified
fast-agent --model sonnet

# Start with a specific skills directory
fast-agent --skills-dir my-test-skill

# Connect to an MCP Server
fast-agent --url https://huggingface.co/mcp 
fast-agent --npx @modelcontextprotocol/server-everything
```

## Colours, Markdown Streaming and Scrollback

**`fast-agent`** streams reasoning, assistant responses and tool calls to the console, rendering markdown while protecting the scrollback buffer.

ANSI colours are used throughout to match your existing preferences. OSC133 and prominent `final response` markers are used to assist scrollback navigation. 

The `apply_patch` tool (supplied, and exposed by default to Codex and `GPT-5.2`+ models when shell file-edit tools are enabled) has highlighting applied during streaming.

Tools can be labelled as generating python code for syntax highlighting (especially useful when integrating with [Pydantic Monty](https://github.com/pydantic/monty))

Shell tool previews also highlight recognized heredoc bodies using the language
of the stdin interpreter. This includes direct interpreters such as
`python -`, repository-standard wrappers such as `uv run python -`, and
TypeScript executed with `pnpm exec tsx -` (including `pnpm -C <dir> exec`).
Highlighting is applied while the heredoc is still streaming.

## Shell Integration

You can run a shell command with `!` - for example `! git status`. When the active agent uses a local shell environment, commands run attached to your terminal, so interactive programs such as `! nano` work as expected. If the active agent uses a remote or sandbox environment, `!` runs in that environment; use `!!` to force a local shell command instead.

You can enter an interactive shell by typing `!` ++return++ (`!!` ++return++ for a local shell). Child shells get `FAST_AGENT_SHELL_CHILD=1`. Type `exit` to return to `fast-agent`.

File names and paths can be automatically completed with either ++tab++ or ++ctrl+space++.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/tui/tui-shell.cast"
  data-fa-asciinema-cols="96"
  data-fa-asciinema-rows="22"
  data-fa-asciinema-poster="npt:0:03"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
  data-fa-asciinema-autoplay="true"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

<!--
Cast asset:
- Source: docs/docs/assets/tui/tui-shell.cast
- Regenerate: uv run scripts/docs.py cast-build tui-shell
- Replay locally: asciinema play docs/docs/assets/tui/tui-shell.cast
-->

## File Previews

When the internal `read_text_file` tool is used, by default 5 lines of the file are displayed. Adjust this with `shell_execution.output_display_lines`, `SHELL_EXECUTION__OUTPUT_DISPLAY_LINES`, or `fast-agent config shell`.

Use `/history detail <turn>` to review the full contents of a specific turn and its tool calls.
When a stored MCP result includes `structuredContent`, its JSON is shown alongside the result's
content blocks.

## Output Review

By default, **`fast-agent`** truncates tool inputs and outputs. 

Use `/history review` to review the latest turn in full, or `/history review <turn>` to select a
specific turn.

## Inspecting Tool Schemas

Use `/tools` to list tools available to the active agent. Use `/tool <name>` (or
`/tools <name>`) to inspect a tool declaration:

- the input JSON Schema is always shown;
- when an MCP tool declares an `outputSchema`, fast-agent identifies it as a
  structured output schema and shows the complete schema.

```text
/tools
/tool hf__hf_whoami
```

The recording connects to the Hugging Face MCP server as `hf` and inspects its
`hf_whoami` declaration.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/tui/mcp-tool-schema.cast"
  data-fa-asciinema-cols="110"
  data-fa-asciinema-rows="32"
  data-fa-asciinema-poster="npt:0:08"
  data-fa-asciinema-speed="0.85"
  data-fa-asciinema-idle-time-limit="3"
  data-fa-asciinema-fit="width"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

<!--
Cast asset:
- Source: docs/docs/assets/tui/mcp-tool-schema.cast
- Regenerate: uv run scripts/docs.py cast-build mcp-tool-schema
- Replay locally: asciinema play docs/docs/assets/tui/mcp-tool-schema.cast
-->

## Image Viewer

Images received from the Assistant or tool calls are rendered to the console on the final turn. Local images that you attach to a user message are previewed in the user panel beneath the attachment link text.

The recording below uses Hugging Face's live MCP server. It shows progress
notifications during image generation, renders the returned image as terminal
cells, and then opens `/mcp` to inspect the same modern connection with a
60-segment, one-second-resolution activity timeline.

!!! note "Recording format"
    The image in this asciinema capture uses halfblock rendering so it can be recorded as plain terminal cells. In a real terminal, `fast-agent` can use higher-resolution terminal image protocols when your terminal supports them.
    

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/tui/hf-image-generation.cast"
  data-fa-asciinema-cols="120"
  data-fa-asciinema-rows="34"
  data-fa-asciinema-poster="npt:0:42.3"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

<!--
Cast asset:
- Source: docs/docs/assets/tui/hf-image-generation.cast
- Regenerate: uv run scripts/docs.py cast-build hf-image-generation
- Replay locally: asciinema play docs/docs/assets/tui/hf-image-generation.cast
-->

## Paste and Attach Images / Documents

You can attach images and documents using `/attach` or by using the `^<uri|file>` syntax. The indicator in the status bar shows a count of attachments, and is green if they are found, red if there is an error. Press ++f10++ to clear all attachments.

You can paste images directly with ++alt+v++. In terminals that reserve that chord, ++ctrl+alt+v++ is also bound.

Local image attachments, including pasted clipboard images, are displayed inline after your message when terminal image rendering is enabled. Remote image URLs remain as links.

## Agent and Model Feature Toggles

Use the function keys in the prompt to cycle agent and model runtime features:

| Key    | Action                                               |
| ------ | ---------------------------------------------------- |
| ++f5++ | Cycle Standard → Delegate → Orchestrate → Harness-only |
| ++f6++ | Cycle reasoning effort                               |
| ++f7++ | Cycle text verbosity                                 |
| ++f8++ | Toggle or cycle web search                           |
| ++f9++ | Toggle or cycle web fetch                            |

Delegate enables the built-in subagent tool. Orchestrate also enables the
parent agent's harness tools for allow-listed commands and resources.
Harness-only keeps those harness tools enabled without subagent delegation. The
toolbar shows these capabilities as `↳⌘`, with active capabilities highlighted.
Agent modes apply only to compatible agents and cannot override an explicit
subagent disable. Model toggles apply when the selected model/provider supports
the feature.

## Durable background processes

Persistent local background commands are supervised independently of the
fast-agent invocation that started them. On POSIX systems with fast-agent home
enabled, their records and output are stored under
`.fast-agent/processes/` (or the configured fast-agent home).

Use:

```text
/process
/process --history
/process attach <process-id>
```

`/process` shows processes already managed by the current runtime and durable
processes discovered from earlier invocations. `attach` adopts management and
output observation in the current runtime; it does not reconnect terminal
input. Once attached, the model-facing `process` tool can inspect output, wait,
or request that the supervisor stop the process.

The session active when a process starts is recorded as provenance. Attaching
from another session adds a non-owning association: deleting, forking, or
leaving a session does not stop the process. Startup reports available durable
processes and records whose supervisor is no longer available.

Durable supervision is currently local and POSIX-only. Session-scoped commands,
remote execution environments, Windows, and `--no-home` retain their existing
process lifecycle behavior.

Fast-agent retains the newest 100 completed durable process records and
automatically removes older terminal records. Running records are never removed
by retention cleanup. If the process store cannot be created or fails its
private-directory checks, fast-agent logs a warning and continues with ordinary
shell execution while durable management is disabled.

## Status Bar

Run `/help status` in the interactive prompt for this legend. The bar reads from
left to right:

```text
status bar
├─ Agent
│  └─ <name>  active agent
├─ Activity
│  ├─ ↻  managed shell processes: dim idle, yellow active, red near the limit
│  ├─ ↳  subagent delegation: green enabled, dim disabled
│  └─ ⌘  harness tools: green enabled, dim disabled
├─ Model
│  ├─ T V D  text, vision, and document support
│  │  └─ green supported; reversed white unsupported; red related content error
│  ├─ ▲ / ▲1…▲9 / ▲+  no draft attachments / count / ten or more
│  │  └─ green usable; red missing, unknown, or unsupported
│  ├─ ⣀…⣿ (paired: ⢀…⢸ ⡀…⡇)  reasoning, then verbosity gauges
│  │  └─ fuller and green → yellow → red mean higher; dim inactive; blue auto
│  ├─ ∞<model>  plan (OAuth login/monthly token plan)
│  ├─ ▼<model>  overlay
│  ├─ »  service tier: dim standard, blue flex, red fast
│  ├─ ⊕  web search: green enabled, dim disabled
│  └─ ⇣  web fetch: green enabled, dim disabled
├─ Context
│  └─ <percent> used, or a zero-padded turn count when usage is unavailable
├─ Mode
│  └─ NRM normal input; MLT multiline input
└─ Right side
   ├─ <working directory> / fast-agent <version>
   ├─ ◀  notifications, sampling, elicitation, warnings, or tool updates
   └─ transient copy notice
```

Unsupported controls are omitted.

## Prompt Shortcuts

| Key            | Action                                                                   |
| -------------- | ------------------------------------------------------------------------ |
| ++ctrl+enter++ | Submit in multiline mode                                                 |
| ++ctrl+space++ | Open completion menu                                                     |
| ++tab++        | Complete path/command, or cycle completions                              |
| ++shift+tab++  | Cycle completions backwards; otherwise cycle service tier when available |
| ++ctrl+t++     | Toggle multiline mode                                                    |
| ++ctrl+e++     | Edit the current buffer in `$EDITOR`                                     |
| ++ctrl+y++     | Copy the last assistant or shell output                                  |
| ++ctrl+l++     | Redraw the screen                                                        |
| ++ctrl+u++     | Clear the input buffer                                                   |
| ++ctrl+c++     | Cancel the current operation; press twice quickly to exit                |
| ++ctrl+d++     | End the prompt session                                                   |


## Markdown Theming

Markdown element colours are themeable with `logger.theme_file`; fenced-code rendering uses `logger.code_theme`.

The default Rich theme is equivalent to:

```ini title="fast-agent-theme.ini"
[styles]
markdown.h1 = bold yellow underline
markdown.h2 = yellow underline
markdown.h3 = bold yellow
markdown.h4 = italic yellow
markdown.h5 = italic yellow
markdown.h6 = dim yellow

markdown.link = bright_blue underline
markdown.link_url = bright_blue underline

markdown.code = bright_green on black
markdown.block_quote = blue

markdown.table.border = bold dim white
markdown.table.header = bright_yellow
markdown.hr = yellow dim
```

Save a modified copy and point `logger.theme_file` at it to override these styles.

## Changing Settings

Use `fast-agent config` to configure your preferences:

- `fast-agent config display` edits console display, markdown rendering, streaming, and prompt mark settings.
- `fast-agent config shell` edits shell execution and file preview settings.

!!! tip "Environment variables"
    The table below lists the matching environment variable for each setting. In general, any nested setting can be overridden by uppercasing the path and joining segments with double underscores; for example, `logger.code_theme` becomes `LOGGER__CODE_THEME`.

--8<-- "docs/docs/_generated/tui_runtime_reference.md"

## Detailed Configuration Reference

See the [Configuration Reference](../ref/config_file/) for the full set of settings.
