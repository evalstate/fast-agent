---
type: agent
model: $system.demo
skills: []
default: true
servers:
  - mcp_sessions_probe
---

Session lifecycle probe demo.

Use tools normally via natural language prompts. Good starter requests:

- "Call session_probe with action=status and note=first."
- "Now call session_probe with action=revoke and note=clear."
- "Call session_probe with action=new and note=rotate."

Use `/mcp` to inspect `exp sess` and cookie state.
