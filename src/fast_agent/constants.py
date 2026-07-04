"""
Global constants for fast_agent with minimal dependencies to avoid circular imports.
"""

from dataclasses import dataclass

# Canonical tool name for the human input/elicitation tool
HUMAN_INPUT_TOOL_NAME = "__human_input"
MCP_UI = "mcp-ui"
REASONING = "reasoning"
REASONING_LABEL = "Reasoning"
"""UI label for reasoning effort configuration."""
ANTHROPIC_THINKING_BLOCKS = "anthropic-thinking-raw"
"""Raw Anthropic thinking blocks with signatures for tool use passback."""
ANTHROPIC_ASSISTANT_RAW_CONTENT = "anthropic-assistant-raw-content"
"""Raw Anthropic assistant content blocks in provider order for exact history replay."""
ANTHROPIC_SERVER_TOOLS_CHANNEL = "anthropic-server-tools"
"""Raw Anthropic server-tool blocks (server_tool_use + *_tool_result) for history passback."""
ANTHROPIC_CITATIONS_CHANNEL = "anthropic-citations"
"""Extracted citation metadata from Anthropic text blocks for source rendering."""
ANTHROPIC_CONTAINER_CHANNEL = "anthropic-container"
"""Anthropic code-execution container metadata for multi-turn request reuse."""
OPENAI_REASONING_ENCRYPTED = "openai-reasoning-encrypted"
"""Encrypted OpenAI reasoning items for stateless Responses API passback."""
OPENAI_ASSISTANT_MESSAGE_ITEMS = "openai-assistant-message-items"
"""Raw OpenAI assistant message items for Responses history passback, including phase."""
OPENAI_MCP_LIST_TOOLS_ITEMS = "openai-mcp-list-tools-items"
"""Raw OpenAI mcp_list_tools output items for Responses history passback."""
FAST_AGENT_ERROR_CHANNEL = "fast-agent-error"
FAST_AGENT_ALERT_CHANNEL = "fast-agent-alert"
FAST_AGENT_SAFETY_DETAILS = "fast-agent-safety-details"
"""Normalized provider safety/refusal metadata for display and diagnostics."""
FAST_AGENT_REMOVED_METADATA_CHANNEL = "fast-agent-removed-meta"
FAST_AGENT_URL_ELICITATION_CHANNEL = "fast-agent-url-elicitation"
FAST_AGENT_TIMING = "fast-agent-timing"
FAST_AGENT_TOOL_METADATA = "fast-agent-tool-metadata"
FAST_AGENT_TOOL_TIMING = "fast-agent-tool-timing"
FAST_AGENT_USAGE = "fast-agent-usage"
FAST_AGENT_SYNTHETIC_FINAL_CHANNEL = "fast-agent-synthetic-final"
FAST_AGENT_PENDING_MEDIA_ATTACHMENTS = "fast-agent-pending-media-attachments"
"""Content blocks staged by attach_media for injection as user input on the next LLM call."""
FAST_AGENT_COMPACTION_CHANNEL = "fast-agent-compaction"
"""Metadata channel marking a compaction summary message (prompt used, counts, timestamps)."""

FORCE_SEQUENTIAL_TOOL_CALLS = False
"""Force tool execution to run sequentially even when multiple tool calls are present."""


def should_parallelize_tool_calls(tool_call_count: int) -> bool:
    """Return True when tool calls should run in parallel (and show per-call IDs)."""
    return (not FORCE_SEQUENTIAL_TOOL_CALLS) and tool_call_count > 1


# should we have MAX_TOOL_CALLS instead to constrain by number of tools rather than turns...?
DEFAULT_MAX_ITERATIONS = 199
"""Maximum number of User/Assistant turns to take"""

DEFAULT_STREAMING_TIMEOUT = 120.0
"""Default idle timeout in seconds between provider streaming events."""

DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT = 8192
"""Baseline byte limit for ACP terminal output when no model info exists."""

TERMINAL_OUTPUT_TOKEN_RATIO = 0.83
"""Target fraction of model max output tokens to budget for terminal output (~2/3 after headroom)."""

TERMINAL_OUTPUT_TOKEN_HEADROOM_RATIO = 0.2
"""Leave headroom for tool wrapper text and other turn data."""

# Empirical observation from real shell outputs (135 samples, avg 3.33 bytes/token)
TERMINAL_BYTES_PER_TOKEN = 3.3
"""Bytes-per-token estimate for terminal output limits and display."""

MAX_TERMINAL_OUTPUT_TOKEN_LIMIT = 10_000
"""Hard cap on default ACP terminal output tokens."""

MAX_TERMINAL_OUTPUT_BYTE_LIMIT = int(MAX_TERMINAL_OUTPUT_TOKEN_LIMIT * TERMINAL_BYTES_PER_TOKEN)
"""Hard cap on default ACP terminal output (~10k tokens with TERMINAL_BYTES_PER_TOKEN=3.3)."""

DEFAULT_AGENT_INSTRUCTION = """You are a helpful AI Agent.

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}

Mermaid diagrams between code fences are supported.

{{model_specific}}

The current date is {{currentDate}}."""


SMART_AGENT_INSTRUCTION = "{{internal:smart_prompt}}"

DEFAULT_GO_AGENT_TYPE = "agent"
"""Default agent type for CLI single-model runs ("smart" or "agent")."""

DEFAULT_SERVE_AGENT_TYPE = "agent"
"""Default agent type for CLI serve/acp single-model runs ("smart" or "agent")."""


DEFAULT_HOME_DIR = ".fast-agent"

DEFAULT_SKILLS_PATHS = [
    f"{DEFAULT_HOME_DIR}/skills",
    ".agents/skills",
    ".claude/skills",
]

CONTROL_MESSAGE_SAVE_HISTORY = "***SAVE_HISTORY"

FAST_AGENT_SHELL_CHILD_ENV = "FAST_AGENT_SHELL_CHILD"
"""Environment variable set when running fast-agent shell commands."""

FAST_AGENT_RUNTIME_HOME = "FAST_AGENT_RUNTIME_HOME"
"""Resolved active fast-agent home exported to shell commands and automation."""


@dataclass(frozen=True)
class DocumentedEnvVar:
    """Environment variable that is part of a documented fast-agent surface."""

    symbol: str
    value: str
    purpose: str
    surface: str


DOCUMENTED_ENV_VARS = (
    DocumentedEnvVar(
        symbol="FAST_AGENT_SHELL_CHILD_ENV",
        value=FAST_AGENT_SHELL_CHILD_ENV,
        purpose="Set to `1` in child shells opened from the TUI with `!`.",
        surface="tui",
    ),
    DocumentedEnvVar(
        symbol="FAST_AGENT_RUNTIME_HOME",
        value=FAST_AGENT_RUNTIME_HOME,
        purpose="Resolved active fast-agent home exported to shell commands and automation.",
        surface="runtime",
    ),
)

SHELL_NOTICE_PREFIX = "[yellow][bold]Agents have shell[/bold][/yellow]"
