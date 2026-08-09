---
title: Skills over MCP
social:
  title: Skills over MCP
  tagline: Install skills from servers compatible with SEP-2640 Draft d7490ecd.
  description: Install skills from servers compatible with SEP-2640 Draft d7490ecd.
  alt: fast-agent social card - Skills over MCP
---

## Compatibility

`fast-agent` is compatible with the
[SEP-2640: Skills Extension](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/d7490ecd1a250f7bc8c3ebb0d65450dfec274bad/seps/2640-skills-extension.md)
**Draft** at revision
`d7490ecd1a250f7bc8c3ebb0d65450dfec274bad`
(`io.modelcontextprotocol/skills`). This is draft compatibility, not support for
a ratified MCP standard.

It supports that revision's `skills/list` and `skills/get` resource-manifest
flow, reading declared files with `resources/read`. Legacy servers that publish
`skill://index.json` entries as `skill-md` or archive artifacts are unsupported.

When a connected MCP server advertises this capability, `fast-agent` shows it as
an MCP-backed skills registry. Opening `/skills registry` calls the paginated
`skills/list` method. Installing a listed skill refreshes its entry with
`skills/get`, downloads every file named by its `resources` manifest through
`resources/read`, and requires each downloaded file to match its declared
SHA-256 digest before writing the local copy. Sidecar metadata records the
host-assigned MCP server identity, skill URI, and resource manifest.

!!! warning "Integrity is not trust"

    A successful SHA-256 check means the downloaded bytes match the manifest
    supplied by that MCP server. It does not authenticate the server or
    publisher, endorse the skill, or establish that its instructions are safe.
    Connect only to MCP servers you trust, and review installed skill content
    before enabling or using it.

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
has no complete digest manifest against which to integrity-check it or compute
an update revision.

## SDK status

The pinned MCP Python SDK does not yet provide typed SEP-2640 request and result
models. `fast-agent` therefore uses local, provisional wire models for
`skills/list`, `skills/get`, and the draft's optional
`resources/directory/read` method. Those internal models may change when the SDK
adds support or the draft changes.

## Trying it

Run or connect to a server compatible with the pinned SEP-2640 draft above.
This example uses the hosted Hugging Face MCP Server:

```text
/mcp connect --name hf https://huggingface.co/mcp
/mcp
/skills registry
/skills available --registry hf
/skills search datasets --registry hf
/skills add <number|name> --registry hf
```

`/mcp` shows when a server advertises the
`io.modelcontextprotocol/skills` extension and points you to the one-shot
`/skills available --registry <server>` browse command. Use
`/skills registry <server>` when you want to make that server the active source
for subsequent skills commands.

Listings show `integrity: SHA-256 manifest; checked on install` when the server
supplies a complete resource manifest. A listing has not yet checked the served
file bytes.

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

This implementation uses MCP as an integrity-checked local-copy installer. It
does not expose MCP-served skill resources directly to the model or retain an
active MCP resource reader after installation. Installed content is an explicit
local copy, not a transparent MCP cache.

`/skills update` calls `skills/get` and compares the complete resource-set
revision with the installed revision. Any file addition, removal, URI change, or
digest change creates a new revision. Updates re-fetch the complete resource set
and SHA-256-check every file against the refreshed server manifest; this is not
publisher verification.
The top-level `fast-agent skills` CLI remains marketplace/file/GitHub oriented;
select MCP registries from an interactive session after connecting the MCP
server.

Thanks to [olaservo](https://github.com/olaservo) for contributing this feature.
