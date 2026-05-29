# Skills-over-MCP demo

This example runs a tiny MCP server that publishes Agent Skills through the
Skills-over-MCP discovery resource:

```text
skill://index.json
```

The server exposes:

- a concrete `pirate` skill at `skill://pirate/SKILL.md`
- a reference file at `skill://pirate/references/example.md`
- a template skill namespace at `skill://docs/{product}/SKILL.md`

## Start with the Server enabled

```bash
fast-agent --stdio "uv run skill_server.py"
```

## Run the scripted demo

From this directory:

```bash
uv run example.py
```

The demo uses the passthrough model to call `read_skill` directly for the
MCP-served skill and one associated resource.

`example.py` prints human-readable demo output. If a client expects MCP
JSON-RPC on stdout it will try to parse that text as protocol messages and
fail.

## Run just the MCP server

If you want to connect another MCP client directly to the Skills-over-MCP
server, point it at the server script instead:

```bash
uv run skill_server.py
```

For fast-agent config, this is already wired in `fast-agent.yaml` as:

```yaml
mcp:
  targets:
    - name: skill_demo
      target: "uv run skill_server.py"
```

## Try it interactively

From this directory:

```bash
uv run fast-agent --env . go
```

Useful commands:

```text
/skills
/skills preview pirate
/skills templates
/skills resolve 1 product=alpha
/skills preview alpha
/skills disable pirate
/skills enable pirate
```

The `pirate` and resolved `alpha` skills are served by the MCP server, not
loaded from local disk. Their content is returned through `read_skill` wrapped
as untrusted MCP-provided input.

## Disable Skills-over-MCP for the server

Set `mcp_skills: false` on the server to keep the MCP target connected while
suppressing `skill://index.json` discovery:

```yaml
mcp:
  targets:
    - name: skill_demo
      target: "uv run skill_server.py"
      mcp_skills: false
```
