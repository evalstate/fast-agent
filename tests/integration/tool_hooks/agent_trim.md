---
type: agent
name: trim_agent
model: passthrough
trim_tool_history: true
function_tools:
  - tools.py:dummy_tool
instruction: Agent with history trimming enabled.
---
