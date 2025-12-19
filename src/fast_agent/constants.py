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
DEFAULT_MAX_ITERATIONS = 99
"""Maximum number of User/Assistant turns to take"""

DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT = 8192
"""Baseline byte limit for ACP terminal output when no model info exists."""

TERMINAL_OUTPUT_TOKEN_RATIO = 0.25
"""Target fraction of model max output tokens to budget for terminal output."""

TERMINAL_OUTPUT_TOKEN_HEADROOM_RATIO = 0.2
"""Leave headroom for tool wrapper text and other turn data."""

TERMINAL_AVG_BYTES_PER_TOKEN = 4
"""Conservative bytes-per-token estimate for mapping token budgets to byte limits."""

MAX_TERMINAL_OUTPUT_BYTE_LIMIT = 32768
"""Hard cap on default ACP terminal output to avoid oversized tool payloads."""

DEFAULT_AGENT_INSTRUCTION = """You are a helpful AI Agent.

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}

The current date is {{currentDate}}."""

DEFAULT_SKILLS_PATHS = [
    ".fast-agent/skills",
    ".claude/skills",
]

CONTROL_MESSAGE_SAVE_HISTORY = "***SAVE_HISTORY"
