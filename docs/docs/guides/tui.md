---
title: Using the TUI
description: Navigating and using fast-agent TUI features.
social:
  title: Using the TUI
  tagline: Navigating and using fast-agent TUI features.
  alt: fast-agent social card — Using the TUI
---

## Colours, Markdown Streaming and Scrollback

**`fast-agent`** streams reasoning, assistant responses and tool calls to the console, rendering markdown while protecting the scrollback buffer.

ANSI colours are used throughout to match your existing preferences. OSC133 and prominent `final response` markers are used to assist scrollback navigation. 

The `apply_patch` tool (supplied, and exposed by default to Codex and `GPT-5.2`+ models when shell file-edit tools are enabled) has highlighting applied during streaming.

Tools can be labelled as generating python code for syntax highlighting (especially useful when integrating with [Pydantic Monty](https://github.com/pydantic/monty))

## Shell Integration

You can run a shell command with `!` - for example `! git status`. You can enter an interactive shell by typing `!` ++return++. Child shells get `FAST_AGENT_SHELL_CHILD=1`. Type `exit` to return to `fast-agent`.

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

Use `/history detail` to review the full contents of a turn and tool calls. 

## Image Viewer

Images received from the Assistant or tool calls are rendered to the console on the final turn. Local images that you attach to a user message are previewed in the user panel beneath the attachment link text.

!!! note "Recording format"
    The image in this asciinema capture uses halfblock rendering so it can be recorded as plain terminal cells. In a real terminal, `fast-agent` can use higher-resolution terminal image protocols when your terminal supports them.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/tui/hf-image-generation.cast"
  data-fa-asciinema-cols="120"
  data-fa-asciinema-rows="34"
  data-fa-asciinema-poster="npt:0:36"
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
- Replay locally: asciinema play docs/docs/assets/tui/hf-image-generation.cast
-->

## Paste and Attach Images / Documents

You can attach images and documents using `/attach` or by using the `^<uri|file>` syntax. The indicator in the status bar shows a count of attachments, and is green if they are found, red if there is an error. Press ++f10++ to clear all attachments.

You can paste images directly with ++alt+v++. In terminals that reserve that chord, ++ctrl+alt+v++ is also bound.

Local image attachments, including pasted clipboard images, are displayed inline after your message when terminal image rendering is enabled. Remote image URLs remain as links.

## Model Feature Toggles

Use the function keys in the prompt to cycle model-specific runtime features:

| Key    | Action                     |
| ------ | -------------------------- |
| ++f6++ | Cycle reasoning effort     |
| ++f7++ | Cycle text verbosity       |
| ++f8++ | Toggle or cycle web search |
| ++f9++ | Toggle or cycle web fetch  |

These toggles apply when the selected model/provider supports the feature.

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

markdown.table.border = yellow
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
