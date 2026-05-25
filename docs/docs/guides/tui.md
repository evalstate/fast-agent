---
title: Using the TUI
description: Navigating and using fast-agent TUI features.
social:
  title: Using the TUI
  tagline: Navigating and using fast-agent TUI features.
  alt: fast-agent social card — Using the TUI
---

## Introduction


### Colours, Markdown Streaming and Scrollback

**`fast-agent`** streams reasoning, assistant responses and tool calls to the console, rendering markdown while protecting the scrollback buffer.

ANSI colours are used throughout to match your existing preferences. OSC133 and prominent `final response` markers are used to assist scrollback navigation. 

The `apply_patch` tool (supplied, and exposed by default to > `GPT-5.2` models) has highlighting applied during streaming.

Tools can be labelled as generating python code for syntax highlighting (especially useful when integrating with [Pydantic Monty](https://github.com/pydantic/monty))

Markdown element colours are themeable with `logger.theme_file` and fenced-code rendering uses `logger.code_theme`.
Use `LOGGER__THEME_FILE` and `LOGGER__CODE_THEME` to set these from the environment.

Use `/history detail` to review the full contents. 

### Shell Integration

You can run a shell command with `!` - for example `! git status`. You can enter an interactive shell by typing `!` ++return++. Child shells get `FAST_AGENT_SHELL_CHILD=1`. Type `exit` to return to `fast-agent`.

File names and paths can be automatically completed with either ++tab++ or ++ctrl+space++.

The recording below shows `fast-agent` starting with a model, a normal prompt, and a shell command run from inside the TUI.

<div class="fa-terminal-demo">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/asciinema-player.css">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/catppuccin.css">
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div id="tui-shell-player"></div>
</div>

<script src="../../assets/vendor/asciinema-player/asciinema-player.min.js"></script>
<script>
  (function () {
    var override = "auto";

    function siteTheme() {
      var scheme = document.documentElement.getAttribute("data-md-color-scheme");
      if (scheme === "slate") {
        return "dark";
      }
      if (scheme === "default") {
        return "light";
      }
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }

    function selectedMode() {
      return override === "auto" ? siteTheme() : override;
    }

    function currentTheme() {
      return selectedMode() === "dark" ? "fast-agent-dark" : "fast-agent-light";
    }

    function updateButtons() {
      document.querySelectorAll("[data-fa-terminal-theme]").forEach(function (button) {
        var active = button.getAttribute("data-fa-terminal-theme") === override;
        button.toggleAttribute("aria-pressed", active);
      });
    }

    function bindButtons() {
      document.querySelectorAll("[data-fa-terminal-theme]").forEach(function (button) {
        if (button.dataset.bound === "true") {
          return;
        }
        button.dataset.bound = "true";
        button.addEventListener("click", function () {
          override = button.getAttribute("data-fa-terminal-theme") || "auto";
          updateButtons();
          renderTuiShellCast(true);
        });
      });
      updateButtons();
    }

    function renderTuiShellCast(force) {
      var target = document.getElementById("tui-shell-player");
      if (!target || !window.AsciinemaPlayer) {
        return;
      }
      bindButtons();
      var theme = currentTheme();
      if (!force && target.dataset.loaded === "true" && target.dataset.theme === theme) {
        return;
      }
      target.dataset.loaded = "true";
      target.dataset.theme = theme;
      target.innerHTML = "";
      window.AsciinemaPlayer.create("../../assets/tui/tui-shell.cast", target, {
        cols: 96,
        rows: 22,
        preload: true,
        poster: "npt:0:03",
        speed: 1,
        idleTimeLimit: 1.3,
        fit: "width",
        theme: theme
      });
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () { renderTuiShellCast(false); });
    } else {
      renderTuiShellCast(false);
    }
    if (window.document$ && window.document$.subscribe) {
      window.document$.subscribe(function () { renderTuiShellCast(false); });
    }
    new MutationObserver(function () { renderTuiShellCast(false); }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"]
    });
  })();
</script>

<!--
Cast asset:
- Source: docs/docs/assets/tui/tui-shell.cast
- Regenerate: uv run scripts/docs.py cast-build tui-shell
- Replay locally: asciinema play docs/docs/assets/tui/tui-shell.cast
-->

### File Previews

When the internal `read_text_file` tool is used, by default 5 lines of the file are displayed. Adjust this with `shell_execution.output_display_lines`, `SHELL_EXECUTION__OUTPUT_DISPLAY_LINES`, or `fast-agent config shell`.

### Image Viewer

Images received from the Assistant or tool calls are rendered to the console on the final turn. Local images that you attach to a user message are previewed in the user panel beneath the attachment link text.

Configure image rendering with `logger.terminal_images` or the matching nested environment variables:

```yaml
logger:
  terminal_images:
    enabled: true
    backend: auto       # auto, textual-image, kitty, sixel, halfcell, unicode, none
    width: 80%
    height: auto
    render_assistant: true
```

For example, use `LOGGER__TERMINAL_IMAGES__ENABLED=false` to disable terminal image rendering or `LOGGER__TERMINAL_IMAGES__BACKEND=kitty` to choose a backend.

The recording below is a review capture of image generation through the Hugging Face MCP server.
It was recorded with `FAST_AGENT_KEYRING_NOTICE=0` so the OS keyring access notice does not appear
in the cast, and with `LOGGER__TERMINAL_IMAGES__ENABLED=true` so terminal image output is enabled:

```bash
export FAST_AGENT_KEYRING_NOTICE=0
export LOGGER__TERMINAL_IMAGES__ENABLED=true
uv run fast-agent -x --model codexplan --url https://huggingface.co/mcp
```

Prompt:

```text
generate an image of a sunflower
```

In this environment the terminal preview is captured by asciinema as SIXEL-style terminal frames
(`SIXEL IMAGE (...) +++++...`) rather than as a separate image asset. The Markdown image link and
source URL remain visible in the recording.

<div class="fa-terminal-demo">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/asciinema-player.css">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/catppuccin.css">
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-image-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-image-terminal-theme="light">Light</button>
    <button type="button" data-fa-image-terminal-theme="dark">Dark</button>
  </div>
  <div id="tui-image-generation-player"></div>
</div>

<script>
  (function () {
    var override = "auto";

    function siteTheme() {
      var scheme = document.documentElement.getAttribute("data-md-color-scheme");
      if (scheme === "slate") {
        return "dark";
      }
      if (scheme === "default") {
        return "light";
      }
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }

    function selectedMode() {
      return override === "auto" ? siteTheme() : override;
    }

    function currentTheme() {
      return selectedMode() === "dark" ? "fast-agent-dark" : "fast-agent-light";
    }

    function updateButtons() {
      document.querySelectorAll("[data-fa-image-terminal-theme]").forEach(function (button) {
        var active = button.getAttribute("data-fa-image-terminal-theme") === override;
        button.toggleAttribute("aria-pressed", active);
      });
    }

    function bindButtons() {
      document.querySelectorAll("[data-fa-image-terminal-theme]").forEach(function (button) {
        if (button.dataset.bound === "true") {
          return;
        }
        button.dataset.bound = "true";
        button.addEventListener("click", function () {
          override = button.getAttribute("data-fa-image-terminal-theme") || "auto";
          updateButtons();
          renderTuiImageCast(true);
        });
      });
      updateButtons();
    }

    function renderTuiImageCast(force) {
      var target = document.getElementById("tui-image-generation-player");
      if (!target || !window.AsciinemaPlayer) {
        return;
      }
      bindButtons();
      var theme = currentTheme();
      if (!force && target.dataset.loaded === "true" && target.dataset.theme === theme) {
        return;
      }
      target.dataset.loaded = "true";
      target.dataset.theme = theme;
      target.innerHTML = "";
      window.AsciinemaPlayer.create(
        "../../assets/tui/hf-image-generation.cast",
        target,
        {
          cols: 120,
          rows: 34,
          preload: true,
          poster: "npt:1:24",
          speed: 1,
          idleTimeLimit: 1.3,
          fit: "width",
          theme: theme
        }
      );
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () { renderTuiImageCast(false); });
    } else {
      renderTuiImageCast(false);
    }
    if (window.document$ && window.document$.subscribe) {
      window.document$.subscribe(function () { renderTuiImageCast(false); });
    }
    new MutationObserver(function () { renderTuiImageCast(false); }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"]
    });
  })();
</script>

### Paste and Attach Images / Documents

You can attach images and documents using `/attach` or by using the `^<uri|file>` syntax. The indicator in the status bar shows a count of attachments, and is green if they are found, red if there is an error. Press ++f10++ to clear all attachments.

You can paste images directly with ++alt+v++. In terminals that reserve that chord, ++ctrl+alt+v++ is also bound.

Local image attachments, including pasted clipboard images, are displayed inline after your message when terminal image rendering is enabled. Remote image URLs remain as links.

### Model Feature Toggles

Use the function keys in the prompt to cycle model-specific runtime features:

| Key    | Action                     |
| ------ | -------------------------- |
| ++f6++ | Cycle reasoning effort     |
| ++f7++ | Cycle text verbosity       |
| ++f8++ | Toggle or cycle web search |
| ++f9++ | Toggle or cycle web fetch  |

These toggles apply when the selected model/provider supports the feature.

### Prompt Shortcuts

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


### Changing Settings

Use `fast-agent config` to configure your preferences:

- `fast-agent config display` edits console display, markdown rendering, streaming, and prompt mark settings.
- `fast-agent config shell` edits shell execution and file preview settings.

You can also set any config value through environment variables by joining nested setting names with double underscores.

--8<-- "docs/docs/_generated/tui_runtime_reference.md"

### Detailed Configuration Reference

See the [Configuration Reference](../ref/config_file/) for the full set of settings.
