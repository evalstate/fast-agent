---
title: A2A Getting Started
description: Connect fast-agent to remote Agent2Agent (A2A) servers and deploy fast-agent agents over A2A HTTP transports.
---

# A2A Getting Started

fast-agent can connect to remote [Agent2Agent (A2A)](https://a2a-protocol.org/)
agents as first-class agents, and can serve fast-agent agents over A2A HTTP
transports. The quickest client path is the `--a2a` command-line shortcut, which
creates a temporary `type: a2a` AgentCard for the current run.

For focused reference material, see:

- [Use as Client](client.md);
- [Serve as A2A Server](server.md);
- [API Usage](api.md);
- [Protocol Compliance](protocol-compliance.md).

This guide uses the deterministic fake A2A server included in the fast-agent test
suite. That keeps the examples copy/pasteable and gives us a repeatable docs +
test pipeline.

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

If you forget the fake server prompts, send:

```text
help
```

The fake server responds with its available demo commands. This is separate from
fast-agent's local `/a2a help`, which lists TUI-side A2A commands.

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

For a longer manual streaming test, use the same server with:

```bash
uv run fast-agent -x \
  --a2a http://127.0.0.1:41242 \
  --a2a-transport JSONRPC \
  --message "please long stream" \
  --quiet
```

In the TUI, send:

```text
please long stream
```

The fake server emits a multi-step "remote analysis" artifact over several
streaming updates, which is useful for checking the live renderer rather than
only the final assistant turn. The stream uses `Step 1 — ...` text instead of a
Markdown ordered list so the TUI preserves the step labels exactly.

Transport names accepted by fast-agent are:

| Canonical | Useful aliases |
|---|---|
| `JSONRPC` | `jsonrpc`, `json-rpc`, `rpc` |
| `HTTP+JSON` | `http`, `http+json`, `rest` |

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
- raw non-image bytes are preserved as blob resources when received by an A2A
  server, and remote raw response bytes render as a safe
  filename/media-type/byte-count placeholder in the fast-agent client.

## 4. Continue an `INPUT_REQUIRED` task

A2A agents can pause a task and ask the client for more input. fast-agent maps
that state to a normal assistant turn, keeps the remote `task_id`, and sends the
next user message back to the same task.

With the fake server, type this in the TUI:

```text
need input
blue
```

The first turn receives:

```text
A2A task TASK_STATE_INPUT_REQUIRED: Please provide the missing value.
```

The second turn is sent with the pending A2A task id and completes the task:

```text
input received: blue
```

Use `/a2a status` between those turns to inspect the preserved `Context`, `Task`,
and `Last state` fields. `/a2a reset` starts a fresh remote context and clears any
pending task.

## 5. Use an AgentCard instead of `--a2a`

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

## 6. Connect inside the TUI

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
/a2a connect <url> [--transport JSONRPC|HTTP+JSON] [--name NAME]
```

## 7. Serve fast-agent over A2A

Use `fast-agent serve a2a` when you want another A2A client to call a fast-agent
agent. The A2A server exposes both HTTP transports:

| Endpoint | URL |
|---|---|
| AgentCard | `http://127.0.0.1:41241/.well-known/agent-card.json` |
| JSON-RPC | `http://127.0.0.1:41241/a2a/jsonrpc` |
| HTTP+JSON | `http://127.0.0.1:41241/a2a/rest` |

Example with an AgentCard bundle:

```bash
uv run fast-agent serve a2a \
  --host 127.0.0.1 \
  --port 41241 \
  --agent-cards ./agents \
  --model codexresponses.gpt-5.4-mini
```

The same runtime wiring used by normal fast-agent agents is available inside the
served agent: configured MCP servers, tools, skills, hooks, and AgentCard-loaded
agents are initialized through the regular fast-agent path before the A2A server
starts.

The generated A2A AgentCard lists each loaded fast-agent agent as an A2A skill.
By default, incoming A2A messages are routed to the fast-agent default agent. API
clients can route to a specific loaded agent by adding message metadata:

```json
{
  "metadata": {
    "agent": "researcher"
  }
}
```

`fast_agent_agent` is accepted as an equivalent metadata key.

See [Protocol Compliance](protocol-compliance.md) for the supported A2A 1.0
surface and known gaps.

### Server sessions

A2A `context_id` is optional in the protocol request. The A2A SDK server
generates one when the client omits it. `fast-agent serve a2a` also honors the
normal `--instance-scope` option:

- `shared` reuses the primary fast-agent instance for all A2A contexts;
- `connection` uses the A2A `context_id` as the server-side instance key;
- `request` creates and disposes a fresh fast-agent instance per message.

The served agent's own `use_history` setting still controls whether prior turns
are sent to the model inside the selected instance scope.

Clients should preserve and reuse the returned `context_id` for conversational
continuity. The fast-agent A2A client does this automatically when history is
enabled, and intentionally starts completed no-history turns with a fresh
context.

### API behavior

The A2A server maps incoming A2A parts into the same `PromptMessageExtended`
shape used by normal fast-agent agents:

- text parts become `TextContent`;
- URL parts become `ResourceLink` where the URL is valid;
- raw image bytes become `ImageContent`;
- other raw bytes become `EmbeddedResource` values with `BlobResourceContents`;
- data parts become formatted JSON text.

Responses are mapped back to A2A artifact parts and completed with
`TASK_STATE_COMPLETED`. Provider credential failures are reported as
`TASK_STATE_AUTH_REQUIRED`. Cancellations are reported as `TASK_STATE_CANCELED`.
When a fast-agent response has `stop_reason=LlmStopReason.PAUSE`, the A2A server
reports `TASK_STATE_INPUT_REQUIRED` with the response text as the status message.
The task remains resumable; clients should send the next user message with the
same A2A `task_id` and `context_id`. The fast-agent A2A client preserves both
automatically, as shown in step 4.

## Demo recording

The repeatable docs pipeline can generate an asciinema recording for the TUI
streaming/files/input-required flow. The committed `.cast` file is embedded below
and can also be downloaded for local replay.

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
