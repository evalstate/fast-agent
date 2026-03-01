---
type: agent
model: $system.demo
skills: []
servers:
  - mcp_sessions_required
---

Global session-required gatekeeper demo.

Try:

- "Call whoami."
- "Call echo with text hello."

The server enforces active data-layer sessions before allowing tool calls.
Use `/mcp` to inspect session metadata.
