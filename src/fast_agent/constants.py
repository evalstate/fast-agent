"""
Global constants for fast_agent with minimal dependencies to avoid circular imports.
"""

# Canonical tool name for the human input/elicitation tool
HUMAN_INPUT_TOOL_NAME = "__human_input"
MCP_UI = "mcp-ui"
REASONING = "reasoning"
ANTHROPIC_THINKING_BLOCKS = "anthropic-thinking-raw"
"""Raw Anthropic thinking blocks with signatures for tool use passback."""
OPENAI_REASONING_ENCRYPTED = "openai-reasoning-encrypted"
"""Encrypted OpenAI reasoning items for stateless Responses API passback."""
FAST_AGENT_ERROR_CHANNEL = "fast-agent-error"
FAST_AGENT_REMOVED_METADATA_CHANNEL = "fast-agent-removed-meta"
FAST_AGENT_TIMING = "fast-agent-timing"
FAST_AGENT_TOOL_TIMING = "fast-agent-tool-timing"
FAST_AGENT_USAGE = "fast-agent-usage"

FORCE_SEQUENTIAL_TOOL_CALLS = False
"""Force tool execution to run sequentially even when multiple tool calls are present."""
# should we have MAX_TOOL_CALLS instead to constrain by number of tools rather than turns...?
DEFAULT_MAX_ITERATIONS = 99
"""Maximum number of User/Assistant turns to take"""

DEFAULT_STREAMING_TIMEOUT = 300.0
"""Default streaming timeout in seconds for provider streaming responses."""

DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT = 8192
"""Baseline byte limit for ACP terminal output when no model info exists."""

TERMINAL_OUTPUT_TOKEN_RATIO = 0.25
"""Target fraction of model max output tokens to budget for terminal output."""

TERMINAL_OUTPUT_TOKEN_HEADROOM_RATIO = 0.2
"""Leave headroom for tool wrapper text and other turn data."""

# Empirical observation from real shell outputs (135 samples, avg 3.33 bytes/token)
TERMINAL_BYTES_PER_TOKEN = 3.3
"""Bytes-per-token estimate for terminal output limits and display."""

MAX_TERMINAL_OUTPUT_BYTE_LIMIT = 32768
"""Hard cap on default ACP terminal output to avoid oversized tool payloads."""

DEFAULT_AGENT_INSTRUCTION = """You are a helpful AI Agent.

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}

The current date is {{currentDate}}."""


SMART_AGENT_INSTRUCTION = """You are a smart fast-agent helper.

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}

Fast-agent environment paths:
- Environment root: {{environmentDir}}
- Agent cards: {{environmentAgentCardsDir}}
- Tool cards: {{environmentToolCardsDir}}

Use the smart tool to load AgentCards temporarily when you need extra agents.
Use validate to check AgentCard files before running them.

The current date is {{currentDate}}."""


DEFAULT_ENVIRONMENT_DIR = ".fast-agent"

DEFAULT_SKILLS_PATHS = [
    f"{DEFAULT_ENVIRONMENT_DIR}/skills",
    ".claude/skills",
]

CONTROL_MESSAGE_SAVE_HISTORY = "***SAVE_HISTORY"

FAST_AGENT_SHELL_CHILD_ENV = "FAST_AGENT_SHELL_CHILD"
"""Environment variable set when running fast-agent shell commands."""

SHELL_NOTICE_PREFIX = "[yellow][bold]Agents have shell[/bold][/yellow]"
