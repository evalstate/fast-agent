---
title: Skills over MCP
social:
  title: Skills over MCP
  tagline: Install Agent Skills from MCP servers that implement SEP-2640.
  description: Install Agent Skills from MCP servers that implement SEP-2640.
  alt: fast-agent social card - Skills over MCP
---

`fast-agent` supports the draft 
[SEP-2640: Skills over MCP Extension](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/93d7a9ddb20d4b3594f4a1be7508ee47f0718f17/seps/2640-skills-extension.md).

When a connected MCP server advertises this capability, `fast-agent` shows it as
an MCP-backed skills registry. Opening `/skills registry` reads
`skill://index.json` and lists installable `skill-md` or archive entries that
include a valid `sha256:` digest. Installing a selected entry downloads the
artifact, verifies its SHA-256, then writes the skill into the normal managed
skills directory. Installed skills then behave like other local skills and
include MCP server provenance plus the verified artifact digest in their sidecar
metadata.

## Trying it

Run or connect to a SEP-2640-enabled MCP server. This example uses the hosted
Hugging Face MCP Server:

```text
/mcp connect --name hf https://huggingface.co/mcp
/mcp
/skills registry
/skills registry hf
/skills available
/skills add <number|name>
```

`/mcp` shows when SEP-2640 Skills over MCP is enabled and points you to
`/skills registry` to select the MCP server as the current install source.
Listings show `integrity: SHA256 checked` for installable MCP skills.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/tui/skills-over-mcp.cast"
  data-fa-asciinema-cols="96"
  data-fa-asciinema-rows="22"
  data-fa-asciinema-poster="npt:0:02"
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
- Source: docs/docs/assets/tui/skills-over-mcp.cast
- Regenerate: uv run scripts/docs.py cast-build skills-over-mcp
- Replay locally: asciinema play docs/docs/assets/tui/skills-over-mcp.cast
-->

## Current scope

This implementation uses MCP as a registry for installation. It does not expose
MCP-served skill resources directly to the model, and it does not make active
skills read supporting files from the MCP server. That deeper resource-loading
workflow is planned separately.

`/skills update` can compare the installed artifact digest with the current MCP
registry digest and apply a verified update when the server publishes a newer
artifact. The top-level `fast-agent skills` CLI remains marketplace/file/GitHub
oriented; select MCP registries from an interactive session after connecting the
MCP server.

Thanks to [olaservo](https://github.com/olaservo) for contributing this feature.