# smart

Minimal card pack with:

- `smart` (default smart coordinator)
- `ripgrep_search` (tool-only search subagent)
- `before_tool_call` hook that strips invalid `-R` / `--recursive` flags from ripgrep commands

## Install locally (example)

If this folder is inside a git repo and listed in a marketplace entry, install with:

```bash
fast-agent cards add smart
```

For direct testing, point `fast-agent` at the cards after copying them into `.fast-agent/`.
