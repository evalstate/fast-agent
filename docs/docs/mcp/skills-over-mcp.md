---
title: Skills over MCP
social:
  title: Skills over MCP
  tagline: Install local skill copies over MCP with integrity checks.
  description: Install local skill copies over MCP with integrity checks.
  alt: fast-agent social card - Skills over MCP
---

## Compatibility

`fast-agent` implements the stable
[Skills Extension](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/seps/2640-skills-extension.md)
(`io.modelcontextprotocol/skills`) wire shapes for its local-copy installer.
This is mechanical wire compatibility, not full released-spec host conformance.

It uses `skills/list` and `skills/get`, reading declared files with
`resources/read`. Each resource requires a nonnegative integer `size` in bytes
alongside its `uri` and `digest`. Skill `resources` must be an array or
`"dynamic"`; omission is invalid. `skills/get` uses `CacheableResult`: cache hints
are recognized, but the caching lifecycle remains pending, with no new automatic
caching behavior. Legacy servers that publish
`skill://index.json` entries as `skill-md` or archive artifacts are unsupported.

When a connected MCP server advertises this capability, `fast-agent` shows it as
an MCP-backed skills registry. Opening `/skills registry` calls the paginated
`skills/list` method. Installing a listed skill refreshes its entry with
`skills/get`, downloads every file named by its `resources` manifest through
`resources/read`, and requires each downloaded file to match its declared
byte size and SHA-256 digest before writing the local copy. Sidecar metadata
records the host-assigned MCP server identity, skill URI, and resource manifest.

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

The selected MCP server confirms that URI through `skills/get`. Skills with
`resources: "dynamic"` are shown in listings, but installation is explicitly
unsupported: there is no complete manifest for integrity checks or update
revisions.

## SDK status

The pinned MCP Python SDK does not yet provide typed Skills Extension request
and result models. `fast-agent` therefore uses local wire models for
`skills/list`, `skills/get`, and `resources/directory/read`. Those internal
models may change when the SDK adds support.

## Trying it

Run or connect to a server using the stable Skills wire format described above.
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

This implementation uses MCP as an eager, integrity-checked local-copy installer.
It does not expose MCP-served skill resources directly to the model or retain an
active MCP resource reader after installation. Installed content is an explicit
local copy, not a transparent MCP cache. Runtime origin tracking and approval
limitations are unchanged; these wire fixes do not add origin-bound execution
or per-use approval enforcement.

`/skills update` calls `skills/get` and compares the complete resource-set
revision with the installed revision. Any file addition, removal, URI change, or
digest or declared-size change creates a new revision. Updates re-fetch the
complete resource set and check every file's size and SHA-256 digest against the
refreshed server manifest; this is not publisher verification.
The top-level `fast-agent skills` CLI remains marketplace/file/GitHub oriented;
select MCP registries from an interactive session after connecting the MCP
server.

Thanks to [olaservo](https://github.com/olaservo) for contributing this feature.
