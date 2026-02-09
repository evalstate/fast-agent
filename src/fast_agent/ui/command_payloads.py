from dataclasses import dataclass
from typing import Literal, TypeGuard


class CommandBase:
    kind: str


@dataclass(frozen=True, slots=True)
class ShowUsageCommand(CommandBase):
    kind: Literal["show_usage"] = "show_usage"


@dataclass(frozen=True, slots=True)
class ShowSystemCommand(CommandBase):
    kind: Literal["show_system"] = "show_system"


@dataclass(frozen=True, slots=True)
class ShowMarkdownCommand(CommandBase):
    kind: Literal["show_markdown"] = "show_markdown"


@dataclass(frozen=True, slots=True)
class ShowMcpStatusCommand(CommandBase):
    kind: Literal["show_mcp_status"] = "show_mcp_status"


McpConnectMode = Literal["url", "npx", "uvx", "stdio"]


@dataclass(frozen=True, slots=True)
class McpListCommand(CommandBase):
    kind: Literal["mcp_list"] = "mcp_list"


@dataclass(frozen=True, slots=True)
class McpConnectCommand(CommandBase):
    target_text: str
    parsed_mode: McpConnectMode
    server_name: str | None
    timeout_seconds: float | None
    trigger_oauth: bool | None
    reconnect_on_disconnect: bool | None
    force_reconnect: bool
    error: str | None
    kind: Literal["mcp_connect"] = "mcp_connect"


@dataclass(frozen=True, slots=True)
class McpDisconnectCommand(CommandBase):
    server_name: str | None
    error: str | None
    kind: Literal["mcp_disconnect"] = "mcp_disconnect"


@dataclass(frozen=True, slots=True)
class ListToolsCommand(CommandBase):
    kind: Literal["list_tools"] = "list_tools"


@dataclass(frozen=True, slots=True)
class ListPromptsCommand(CommandBase):
    kind: Literal["list_prompts"] = "list_prompts"


@dataclass(frozen=True, slots=True)
class ListSkillsCommand(CommandBase):
    kind: Literal["list_skills"] = "list_skills"


@dataclass(frozen=True, slots=True)
class ShowHistoryCommand(CommandBase):
    agent: str | None
    kind: Literal["show_history"] = "show_history"


@dataclass(frozen=True, slots=True)
class ClearCommand(CommandBase):
    kind: Literal["clear_history", "clear_last"]
    agent: str | None


@dataclass(frozen=True, slots=True)
class SkillsCommand(CommandBase):
    action: str
    argument: str | None
    kind: Literal["skills_command"] = "skills_command"


@dataclass(frozen=True, slots=True)
class SelectPromptCommand(CommandBase):
    prompt_name: str | None
    prompt_index: int | None
    kind: Literal["select_prompt"] = "select_prompt"


@dataclass(frozen=True, slots=True)
class SwitchAgentCommand(CommandBase):
    agent_name: str
    kind: Literal["switch_agent"] = "switch_agent"


@dataclass(frozen=True, slots=True)
class HashAgentCommand(CommandBase):
    """Send a message to an agent and return the response to the input buffer."""

    agent_name: str
    message: str
    kind: Literal["hash_agent"] = "hash_agent"


@dataclass(frozen=True, slots=True)
class SaveHistoryCommand(CommandBase):
    filename: str | None
    kind: Literal["save_history"] = "save_history"


@dataclass(frozen=True, slots=True)
class LoadHistoryCommand(CommandBase):
    filename: str | None
    error: str | None
    kind: Literal["load_history"] = "load_history"


@dataclass(frozen=True, slots=True)
class LoadPromptCommand(CommandBase):
    filename: str | None
    error: str | None
    kind: Literal["load_prompt"] = "load_prompt"


@dataclass(frozen=True, slots=True)
class HistoryRewindCommand(CommandBase):
    turn_index: int | None
    error: str | None
    kind: Literal["history_rewind"] = "history_rewind"


@dataclass(frozen=True, slots=True)
class HistoryReviewCommand(CommandBase):
    turn_index: int | None
    error: str | None
    kind: Literal["history_review"] = "history_review"


@dataclass(frozen=True, slots=True)
class HistoryFixCommand(CommandBase):
    agent: str | None
    kind: Literal["history_fix"] = "history_fix"


@dataclass(frozen=True, slots=True)
class LoadAgentCardCommand(CommandBase):
    filename: str | None
    add_tool: bool
    remove_tool: bool
    error: str | None
    kind: Literal["load_agent_card"] = "load_agent_card"


@dataclass(frozen=True, slots=True)
class ReloadAgentsCommand(CommandBase):
    kind: Literal["reload_agents"] = "reload_agents"


@dataclass(frozen=True, slots=True)
class AgentCommand(CommandBase):
    agent_name: str | None
    add_tool: bool
    remove_tool: bool
    dump: bool
    error: str | None
    kind: Literal["agent_command"] = "agent_command"


@dataclass(frozen=True, slots=True)
class ListSessionsCommand(CommandBase):
    show_help: bool = False
    kind: Literal["list_sessions"] = "list_sessions"


@dataclass(frozen=True, slots=True)
class CreateSessionCommand(CommandBase):
    session_name: str | None
    kind: Literal["create_session"] = "create_session"


@dataclass(frozen=True, slots=True)
class SwitchSessionCommand(CommandBase):
    session_name: str
    kind: Literal["switch_session"] = "switch_session"


@dataclass(frozen=True, slots=True)
class ResumeSessionCommand(CommandBase):
    session_id: str | None
    kind: Literal["resume_session"] = "resume_session"


@dataclass(frozen=True, slots=True)
class TitleSessionCommand(CommandBase):
    title: str
    kind: Literal["title_session"] = "title_session"


@dataclass(frozen=True, slots=True)
class ForkSessionCommand(CommandBase):
    title: str | None
    kind: Literal["fork_session"] = "fork_session"


@dataclass(frozen=True, slots=True)
class ClearSessionsCommand(CommandBase):
    target: str | None
    kind: Literal["clear_sessions"] = "clear_sessions"


@dataclass(frozen=True, slots=True)
class PinSessionCommand(CommandBase):
    value: str | None
    target: str | None
    kind: Literal["pin_session"] = "pin_session"


@dataclass(frozen=True, slots=True)
class ShellCommand(CommandBase):
    """Execute a shell command directly."""

    command: str
    kind: Literal["shell_command"] = "shell_command"


@dataclass(frozen=True, slots=True)
class ModelReasoningCommand(CommandBase):
    value: str | None
    kind: Literal["model_reasoning"] = "model_reasoning"


@dataclass(frozen=True, slots=True)
class ModelVerbosityCommand(CommandBase):
    value: str | None
    kind: Literal["model_verbosity"] = "model_verbosity"


@dataclass(frozen=True, slots=True)
class UnknownCommand(CommandBase):
    command: str
    kind: Literal["unknown_command"] = "unknown_command"


CommandPayload = (
    ShowUsageCommand
    | ShowSystemCommand
    | ShowMarkdownCommand
    | ShowMcpStatusCommand
    | McpListCommand
    | McpConnectCommand
    | McpDisconnectCommand
    | ListToolsCommand
    | ListPromptsCommand
    | ListSkillsCommand
    | ShowHistoryCommand
    | ClearCommand
    | SkillsCommand
    | SelectPromptCommand
    | SwitchAgentCommand
    | HashAgentCommand
    | SaveHistoryCommand
    | LoadHistoryCommand
    | LoadPromptCommand
    | HistoryRewindCommand
    | HistoryReviewCommand
    | HistoryFixCommand
    | LoadAgentCardCommand
    | ReloadAgentsCommand
    | AgentCommand
    | ListSessionsCommand
    | CreateSessionCommand
    | SwitchSessionCommand
    | ResumeSessionCommand
    | TitleSessionCommand
    | ForkSessionCommand
    | ClearSessionsCommand
    | PinSessionCommand
    | ShellCommand
    | ModelReasoningCommand
    | ModelVerbosityCommand
    | UnknownCommand
)


def is_command_payload(value: object) -> TypeGuard[CommandPayload]:
    return isinstance(value, CommandBase)
