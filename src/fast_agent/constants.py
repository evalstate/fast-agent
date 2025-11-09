"""
Global constants for fast_agent with minimal dependencies to avoid circular imports.
"""

# Canonical tool name for the human input/elicitation tool
HUMAN_INPUT_TOOL_NAME = "__human_input"
MCP_UI = "mcp-ui"
REASONING = "reasoning"
FAST_AGENT_ERROR_CHANNEL = "fast-agent-error"
FAST_AGENT_REMOVED_METADATA_CHANNEL = "fast-agent-removed-meta"
FAST_AGENT_TIMING = "fast-agent-timing"
FAST_AGENT_TOOL_TIMING = "fast-agent-tool-timing"
# should we have MAX_TOOL_CALLS instead to constrain by number of tools rather than turns...?
DEFAULT_MAX_ITERATIONS = 20
"""Maximum number of User/Assistant turns to take"""

DEFAULT_AGENT_INSTRUCTION = """You are a helpful AI Agent.

{{serverInstructions}}

{{agentSkills}}

The current date is {{currentDate}}."""
