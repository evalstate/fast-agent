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
class SaveHistoryCommand(CommandBase):
    filename: str | None
    kind: Literal["save_history"] = "save_history"


@dataclass(frozen=True, slots=True)
class LoadHistoryCommand(CommandBase):
    filename: str | None
    error: str | None
    kind: Literal["load_history"] = "load_history"


CommandPayload = (
    ShowUsageCommand
    | ShowSystemCommand
    | ShowMarkdownCommand
    | ShowMcpStatusCommand
    | ListToolsCommand
    | ListPromptsCommand
    | ListSkillsCommand
    | ShowHistoryCommand
    | ClearCommand
    | SkillsCommand
    | SelectPromptCommand
    | SwitchAgentCommand
    | SaveHistoryCommand
    | LoadHistoryCommand
)


def is_command_payload(value: object) -> TypeGuard[CommandPayload]:
    return isinstance(value, CommandBase)
