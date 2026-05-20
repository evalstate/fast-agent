---
title: A2A Getting Started
description: Connect fast-agent to a remote Agent2Agent (A2A) server, stream responses, and inspect file/data parts.
---

# A2A Getting Started

fast-agent can connect to remote [Agent2Agent (A2A)](https://a2a-protocol.org/)
agents as first-class agents. The quickest path is the `--a2a` command-line
shortcut, which creates a temporary `type: a2a` AgentCard for the current run.

This guide uses the deterministic fake A2A server included in the fast-agent test
suite. That keeps the examples copy/pasteable and gives us a repeatable docs +
test pipeline.

!!! note "Client-only scope"
    This page covers fast-agent acting as an A2A **client**. Serving a fast-agent
    agent as an A2A server is planned separately.

## 1. Start the fake A2A server

From the fast-agent repository root, run:

```bash
--8<-- "docs/docs/a2a/snippets/start-fake-server.sh"
```

The fake server exposes:

| Endpoint | URL |
|---|---|
| AgentCard | `http://127.0.0.1:41242/.well-known/agent-card.json` |
| JSON-RPC | `http://127.0.0.1:41242/a2a/jsonrpc` |
| HTTP+JSON | `http://127.0.0.1:41242/a2a/rest` |

Keep this server running in one terminal, then use a second terminal for the
client commands below.

## 2. Connect from the CLI and stream a response

```bash
--8<-- "docs/docs/a2a/snippets/cli-stream-command.sh"
```

Expected output:

```text
--8<-- "docs/docs/a2a/snippets/cli-stream-output.txt"
```

The `--a2a` value is normally the remote A2A agent's base URL. fast-agent resolves
its AgentCard from `/.well-known/agent-card.json`, selects the requested
transport, sends the message, and prints the final aggregated response.

Transport names accepted by fast-agent are:

| Canonical | Useful aliases |
|---|---|
| `JSONRPC` | `jsonrpc`, `json-rpc`, `rpc` |
| `HTTP+JSON` | `http`, `http+json`, `rest` |
| `GRPC` | `grpc` |

## 3. Receive file, URL, and data parts

The fake server can also return non-text A2A parts:

```bash
--8<-- "docs/docs/a2a/snippets/cli-files-command.sh"
```

Expected output:

````text
--8<-- "docs/docs/a2a/snippets/cli-files-output.txt"
````

Current rendering behavior:

- text parts render as normal assistant text;
- URL parts render as Markdown links;
- data parts render as fenced JSON;
- raw bytes render as a safe filename/media-type/byte-count placeholder.

## 4. Use an AgentCard instead of `--a2a`

For persistent configuration, create a card like this:

```yaml
--8<-- "docs/docs/a2a/snippets/agent-card.yaml"
```

Then run:

```bash
uv run fast-agent -x --agent-cards ./fake-a2a.yaml --agent fake_remote
```

Use AgentCards when you want the connection checked in, shared, or combined with
other configured agents.

## 5. Connect inside the TUI

You can connect to A2A agents after the TUI has started:

```text
/a2a connect http://127.0.0.1:41242 --transport JSONRPC --name fake_remote
```

Useful diagnostics:

```text
--8<-- "docs/docs/a2a/snippets/tui-session.txt"
```

The `/a2a` command group currently includes:

```text
/a2a list
/a2a status [agent]
/a2a card [agent]
/a2a transport [agent]
/a2a reset [agent]
/a2a connect <url> [--transport JSONRPC|HTTP+JSON|GRPC] [--name NAME]
```

## Demo recording

The repeatable docs pipeline can generate an asciinema recording for the TUI
streaming/files flow. The committed `.cast` file is embedded below and can also
be downloaded for local replay.

<div class="a2a-terminal-demo">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/asciinema-player.css">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/catppuccin.css">
  <div class="a2a-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-a2a-terminal-theme="auto">Auto</button>
    <button type="button" data-a2a-terminal-theme="light">Light</button>
    <button type="button" data-a2a-terminal-theme="dark">Dark</button>
  </div>
  <div id="a2a-streaming-files-player"></div>
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
      document.querySelectorAll("[data-a2a-terminal-theme]").forEach(function (button) {
        var active = button.getAttribute("data-a2a-terminal-theme") === override;
        button.toggleAttribute("aria-pressed", active);
      });
    }

    function bindButtons() {
      document.querySelectorAll("[data-a2a-terminal-theme]").forEach(function (button) {
        if (button.dataset.bound === "true") {
          return;
        }
        button.dataset.bound = "true";
        button.addEventListener("click", function () {
          override = button.getAttribute("data-a2a-terminal-theme") || "auto";
          updateButtons();
          renderA2ACast(true);
        });
      });
      updateButtons();
    }

    function renderA2ACast(force) {
      var target = document.getElementById("a2a-streaming-files-player");
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
        "../../assets/a2a/a2a-streaming-files.cast",
        target,
        {
          cols: 104,
          rows: 27,
          preload: true,
          poster: "npt:0:03",
          speed: 1,
          idleTimeLimit: 1.3,
          fit: "width",
          theme: theme
        }
      );
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () { renderA2ACast(false); });
    } else {
      renderA2ACast(false);
    }
    if (window.document$ && window.document$.subscribe) {
      window.document$.subscribe(function () { renderA2ACast(false); });
    }
    new MutationObserver(function () { renderA2ACast(false); }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"]
    });
  })();
</script>

If the player does not load, [download the A2A streaming/files cast](../assets/a2a/a2a-streaming-files.cast)
and replay it locally with:

```bash
asciinema play docs/docs/assets/a2a/a2a-streaming-files.cast
```

## Regenerate these examples

The page snippets and cast are generated from the same fake server used by the
integration tests:

```bash
uv run scripts/a2a_docs_pipeline.py generate
uv run scripts/a2a_docs_pipeline.py check
```

To refresh the terminal recording as well, install `asciinema` and `tmux`, then
run:

```bash
uv run scripts/a2a_docs_pipeline.py record
```
