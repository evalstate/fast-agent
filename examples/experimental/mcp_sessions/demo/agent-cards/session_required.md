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

The server enforces session establishment before allowing tool calls.
Use `/mcp` to inspect cookie/session state.
