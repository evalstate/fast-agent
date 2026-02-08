---
type: agent
name: hooks_agent
model: passthrough
function_tools:
  - tools.py:dummy_tool
tool_hooks:
  after_turn_complete: hooks.py:track_after_turn_complete
instruction: Agent with custom hooks.
---
