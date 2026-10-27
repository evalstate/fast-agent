---
type: agent
name: ptc_eval
description: Agent used by the PTC evals; enables every tool family PTC has to handle.
default: true
servers:
  - orders
  - media
shell: true
subagents: true
harness_tools: true
---
You are a precise assistant. Answer with the facts asked for and nothing else.

{{env}}
