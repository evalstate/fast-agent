---
name: smart
type: smart
description: |
  Smart coordinator that delegates repository discovery and code search to
  the ripgrep_search subagent.
default: true
agents:
  - ripgrep_search
use_history: true
---

{{internal:smart_prompt}}

When codebase discovery or content lookup is needed, prefer delegating to
`ripgrep_search` and then synthesize concise, path-cited results.
