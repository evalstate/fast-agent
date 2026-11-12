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

## Terminal program status (OSC 7501)

Interactive fast-agent sessions emit the
[Program Status Protocol (OSC 7501)](https://www.superlogical.com/rex/docs/build/program-status)
by default. Supporting terminals and managers, including herdr-core and the new
Herdr app, can show when fast-agent is idle, working, or waiting for human input.
The report includes a session title and a compact context/model and cost summary
(or token counts when pricing is unavailable). Metadata changes retain the current
state; exiting clears the report.

This is independent of OSC 133 scrollback markers and `logger.enable_prompt_marks`.
No Herdr installation, environment detection, or capability query is required.
Unsupported terminals normally ignore these sequences; multiplexers such as tmux
may discard them rather than forward them.

To disable OSC 7501 output, set this environment variable before starting fast-agent:

```bash
export FAST_AGENT_DISABLE_PROGRAM_STATUS=1
```

In PowerShell:

```powershell
$env:FAST_AGENT_DISABLE_PROGRAM_STATUS = "1"
```

Values `1`, `true`, `yes`, and `on` disable reporting (case-insensitive). There is
no YAML setting. The opt-out does not disable OSC 133 or the legacy Herdr TUI's
socket integration.

Status output is restricted to interactive prompt sessions with terminal stdout;
it is skipped for redirected output and `TERM=dumb`. ACP/MCP protocol sessions
do not activate it. Raw terminal recordings may retain the reports, including
session titles derived from prompts; base64 encoding is not encryption.

The new Herdr app recognizes agents from their status reports. Displaying the
usage summary in its agents panel requires a custom mod reading the terminal's
`report.msg`; the old Herdr TUI's `$context`/`$cost` row configuration does not
carry over.

## Terminal images

Image rendering automatically uses reported terminal graphics support, including
Kitty graphics inside Herdr. Herdr can report Kitty support even when the outer
terminal cannot display it. For Herdr inside Foot or Windows Terminal, select
half-cell rendering:

```yaml
logger:
  terminal_images:
    backend: halfcell
```

Alternatively, select it for a terminal environment:

```bash
export LOGGER__TERMINAL_IMAGES__BACKEND=halfcell
```

In PowerShell:

```powershell
$env:LOGGER__TERMINAL_IMAGES__BACKEND = "halfcell"
```

Explicit configuration takes precedence over this environment variable; leave
`backend` unset in YAML to select it via the environment. Restart fast-agent
after changing these settings. With Herdr inside Ghostty or Kitty, leave the
backend at its automatic default or explicitly select `kitty`.

## Shell Integration

You can run a shell command with `!` - for example `! git status`. When the active agent uses a local shell environment, commands run attached to your terminal, so interactive programs such as `! nano` work as expected. If the active agent uses a remote or sandbox environment, `!` runs in that environment; use `!!` to force a local shell command instead.

Use `!!!` to run a command in the active agent environment and insert its output into the next prompt, where you can edit it before sending.

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

In Herdr app, **Ctrl+V** in a terminal on another machine uploads a clipboard image
to that machine and pastes its path. fast-agent recognizes a paste containing only
existing absolute image paths and adds attachments automatically when the model
supports images. When it does not, the path is inserted as text with a hint to
switch models with `/model`; submitting a file path on its own explains how to
`/attach` it instead of reporting an unknown command.
This also works in Herdr-managed remote terminals: run an updated fast-agent on
the remote machine. Ordinary text pastes and paths inside slash/shell commands
remain unchanged. Alt+V / Ctrl+Alt+V instead reads the clipboard on the machine
running fast-agent; it does not forward your desktop clipboard over SSH.

Automatic image rendering supports Herdr's Kitty graphics, including terminals
that return their graphics capability reply after their general terminal reply.
`logger.terminal_images.backend: kitty` remains an explicit override; `halfcell`
opts into lower-resolution character-cell graphics.

Local image attachments, including pasted clipboard images, are displayed inline after your message when terminal image rendering is enabled. Remote image URLs remain as links.

## Terminal integration diagnostics

```bash
fast-agent check terminal           # Probe capabilities on an interactive terminal
fast-agent check terminal --image   # Also draw a portable resolution test chart
fast-agent check terminal --json    # Scriptable report; no terminal probes or graphics
```

The check reports TTY availability, configured image backend, the actual selected
renderer, reported Kitty/Sixel support, cell-pixel geometry, Herdr core terminal ID
when supplied, and eligibility for OSC 7501 status reporting. `TERM_PROGRAM` is
shown only as a hint, not treated as proof of capabilities. A cell size can be a
library fallback if the terminal supplies none.

`--image` uses the configured renderer and a viewport-fitted chart of small text,
fine stripes and a gradient, without system font dependencies. It requires an
interactive terminal and cannot be combined with `--json`. Images are not drawn
by default. No model calls, clipboard access, uploads or lifecycle status writes
are performed; model vision support is explicitly left unchecked. In redirected
output and JSON mode, active graphics probing is skipped and unknown fields are null.

### Agent lifecycle status

Interactive fast-agent reports OSC 7501 status to compatible terminals, including
herdr-core. A turn or direct shell command starts as `working`. Human elicitation
uses `blocked` (`kind=question`; OAuth callback pastes use `kind=auth`) and restores
the previous state when it ends. Successful completion reports `idle`; Herdr decides
whether to show its unseen-completion `done` indicator.

Questions asked before the first prompt (the one-time recommended-plugins offer,
the startup model picker, and missing shell directory creation) also report
`blocked`, so an agent started with `hc run` shows that it is waiting for a person
instead of reporting no status. Once answered, the record is withdrawn until the
prompt reports `idle`.

Uncaught turn failures, exhausted-retry system errors, assistant `ERROR`/`TIMEOUT`
stop reasons, and nonzero direct-shell exits report `error`. The error remains while
waiting at the next prompt, including metadata/cost refreshes, and the next operation
clears it. Ctrl+C and task cancellation return to `idle`; conventional interrupt or
termination statuses (130, 143, -2 and -15) are treated as cancellation. A failed
individual tool call does not fail an agent turn that continues running. Token/iteration
limits and safety stops are not automatically classified as execution errors.

`hc wait` returns on either failure or completion; inspect its `status` rather than
assuming a returned wait means success. The old Herdr socket protocol has no error
state and retains its previous idle behavior; the richer status travels over OSC 7501.

### Typing ahead and messages sent while working

While a turn is running, fast-agent holds terminal input instead of echoing it into
the output. Submitted text is queued in arrival order, using the active prompt mode:
Enter submits in normal mode; in multiline mode (Ctrl+T), Enter inserts a newline
and Ctrl+J submits. Plain messages are consumed at the next tool-loop
continuation, after the complete tool batch and before the next model call.
This lets the agent read a worker report after `hc wait` returns, without first
finishing its answer. Consumed messages remain part of the current conversation
turn and are not replayed into the next prompt.

Commands (`/`, `!`, `@`, `#`), resource/attachment mentions and pasted image paths
stay on the normal prompt path after the turn ends. Later messages cannot overtake
these deferred inputs. Text without the active submit key remains a draft. Messages
arriving after the last continuation boundary become the next turn; `hc wait` returning
`reason: message` does not guarantee that all message bytes have arrived yet.

`hc send` pastes text and presses Enter, so it assumes normal input mode. With
multiline mode enabled, its message remains a draft. Multiline message content
still works in normal mode through paste or Ctrl+J newlines followed by Enter.

While input is held, only the count (`⧗ 1 queued`) appears in the live streaming
header or progress display and the OSC 7501 status message. Submitted input
remaining at turn completion keeps the status working through the prompt handoff;
an unfinished draft does not. Consumed messages appear before the next model response.

Press ++escape++ (or Ctrl+C) to cancel the current turn; `hc interrupt` sends
Escape. Messages already queued are still sent after the cancellation. Forms such as
elicitation receive input normally while they are open.

This applies to interactive POSIX terminals. To keep the previous behavior, set:

```bash
export FAST_AGENT_DISABLE_TURN_INPUT=1
```

During a terminal's first 10 seconds, `hc 0.1.17` requires 800 ms of quiet output
before sending, waiting at most 5 seconds per attempt. Continuous progress redraws
can therefore cause “still starting: its output has not settled” even after the
agent has accepted an earlier turn. This startup check happens before input reaches
fast-agent and is bypassed once the terminal is 10 seconds old; it is not a
restriction on established sessions. `--quiet` avoids this startup interaction.
See [hc's startup check](https://github.com/herdrdev/herdr-core/blob/v0.1.17/crates/herdr-core/src/input.rs#L215-L246).

The opt-in real-terminal regression suite covers this separately from input queueing:

```bash
FAST_AGENT_TEST_HC=1 uv run pytest tests/integration/ui/test_hc_queued_input.py -v -rx
```

### Starting fast-agent with `hc`

fast-agent does not take its first message as a positional argument
(`hc run --bg fast-agent "task"` fails with an explanation). Either start it and
send the task once it reports `idle`:

```bash
hc run --bg --name helper fast-agent
hc wait helper
hc send helper "Review src/x.py and report with hc send $HC_TERMINAL_ID"
```

or use an `hc` template (`~/.config/herdr-core/templates/<name>.md`) whose
`command` starts fast-agent; `hc` waits for `idle` and sends the template body plus
your task as the first message:

```markdown
---
name: fa-reviewer
description: fast-agent code reviewer
command: fast-agent go --model sonnet
---
Review the requested files. Read-only.
```

`hc run` starts programs with herdr-core's environment, not your shell's: use an
absolute path to run a development or virtual-environment build.

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
the feature. Web search, web fetch, and the agent capability mode are saved in
the session checkpoint and restored on resume.

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
/process terminate <process-id>
```

`/process` shows processes already managed by the current runtime and durable
processes discovered from earlier invocations. `attach` adopts management and
output observation in the current runtime; it does not reconnect terminal
input. Once attached, the model-facing `process` tool can inspect output, wait,
or request that the supervisor stop the process.

`terminate` stops a process already managed by the current runtime or sends a
stop request to a discoverable durable process without attaching it. Durable
termination remains supervisor-mediated; unavailable records are never
signalled directly.

The session active when a process starts is recorded as provenance. Attaching
from another session adds a non-owning association: deleting, forking, or
leaving a session does not stop the process. Startup reports available durable
processes and identifies their associated sessions when those sessions still
exist. Resuming a session reattaches its active durable processes and warns
about associated unavailable records. Discovery marks records unavailable once
both the supervisor and child process have disappeared, or when a stale launch
never published either process ID. Their provenance and retained output remain
in history until normal terminal-record retention removes them.

Durable supervision is currently local and POSIX-only. Session-scoped commands,
remote execution environments, Windows, and `--no-home` retain their existing
process lifecycle behavior.

Fast-agent retains the newest 100 completed durable process records and
automatically removes older terminal records. Active records are not removed by
retention cleanup. Each stdout, stderr, and combined durable output log is
capped by `shell_execution.durable_output_max_bytes`; output beyond the cap is
still drained but is reported as dropped and is unavailable for later readback.
If the process store cannot be created or fails its private-directory checks,
fast-agent logs a warning and continues with ordinary shell execution while
durable management is disabled.

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
