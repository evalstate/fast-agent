---
type: agent
model: $system.demo
skills: []
servers:
  - mcp_sessions_selective
---

Selective session policy demo (public + session-only tools).

Try:

- "Call public_echo with text hi."
- "Get the current session counter value."
- "Reset the session."
- "Try session_counter_get again (should fail with session not found)."
- "Start a session labeled demo."
- "Increment the session counter."
