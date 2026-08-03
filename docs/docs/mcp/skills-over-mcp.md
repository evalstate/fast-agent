---
title: Skills over MCP
social:
  title: Skills over MCP
  tagline: Install Agent Skills from MCP servers that implement SEP-2640.
  description: Install Agent Skills from MCP servers that implement SEP-2640.
  alt: fast-agent social card - Skills over MCP
---

`fast-agent` supports the current draft
[SEP-2640: Skills Extension](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/d7490ecd1a250f7bc8c3ebb0d65450dfec274bad/seps/2640-skills-extension.md)
as a verified MCP-to-local skill installer.

When a connected MCP server advertises this capability, `fast-agent` shows it as
an MCP-backed skills registry. Opening `/skills registry` calls the paginated
`skills/list` method. Installing a listed skill refreshes its entry with
`skills/get`, downloads every file named by its `resources` manifest through
`resources/read`, verifies every SHA-256 digest, and writes the complete skill
into the normal managed skills directory. Installed skills include the
host-assigned MCP server identity, skill URI, and verified resource set in their
sidecar metadata.

Skill names are labels rather than identifiers. `fast-agent` preserves
same-named entries in an MCP listing and uses their URIs to disambiguate them.
The local managed skills directory can contain only one installed skill with a
given name. Because a server may return a partial or empty list, you can also
install a skill omitted from the listing when you know its URI:

```text
/skills add skill://acme/example/SKILL.md
```

The selected MCP server confirms that URI through `skills/get`. Skills that omit
`resources` are shown in listings but cannot be installed, because their content
cannot be verified or bound to an update revision.

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
Listings show `integrity: SHA256 checked` when the server supplies a complete
resource manifest.

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

This implementation uses MCP as a verified installation source. It does not
expose MCP-served skill resources directly to the model or retain an active MCP
resource reader after installation. Installed content is an explicit local copy,
not a transparent MCP cache.

`/skills update` calls `skills/get` and compares the complete resource-set
revision with the installed revision. Any file addition, removal, URI change, or
digest change creates a new revision; updates re-fetch and verify the entire set.
The top-level `fast-agent skills` CLI remains marketplace/file/GitHub oriented;
select MCP registries from an interactive session after connecting the MCP
server.

Thanks to [olaservo](https://github.com/olaservo) for contributing this feature.
