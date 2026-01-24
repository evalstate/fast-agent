"""
Enhanced prompt functionality with advanced prompt_toolkit features.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from rich import print as rich_print

from fast_agent.agents.agent_types import AgentType
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.agents.workflow.router_agent import RouterAgent
from fast_agent.constants import (
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_REMOVED_METADATA_CHANNEL,
)
from fast_agent.core.exceptions import PromptExitError
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import available_reasoning_values
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.ui.command_payloads import (
    AgentCommand,
    ClearCommand,
    ClearSessionsCommand,
    CommandPayload,
    CreateSessionCommand,
    ForkSessionCommand,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryReviewCommand,
    HistoryRewindCommand,
    ListSessionsCommand,
    ListToolsCommand,
    LoadAgentCardCommand,
    LoadHistoryCommand,
    LoadPromptCommand,
    ModelReasoningCommand,
    ReloadAgentsCommand,
    ResumeSessionCommand,
    SaveHistoryCommand,
    SelectPromptCommand,
    ShellCommand,
    ShowHistoryCommand,
    ShowMarkdownCommand,
    ShowMcpStatusCommand,
    ShowSystemCommand,
    ShowUsageCommand,
    SkillsCommand,
    SwitchAgentCommand,
    TitleSessionCommand,
    UnknownCommand,
    is_command_payload,
)
from fast_agent.ui.mcp_display import render_mcp_status
from fast_agent.ui.model_display import format_model_display
from fast_agent.ui.reasoning_effort_display import render_reasoning_effort_gauge
from fast_agent.ui.shell_notice import format_shell_notice

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.types import PromptMessageExtended

# Get the application version
try:
    app_version = version("fast-agent-mcp")
except:  # noqa: E722
    app_version = "unknown"

# Map of agent names to their history
agent_histories = {}

# Store available agents for auto-completion
available_agents = set()

# Keep track of multi-line mode state
in_multiline_mode = False

# Track last copyable output (shell output or assistant response)
_last_copyable_output: str | None = None

# Track transient copy notice for the toolbar.
_copy_notice: str | None = None
_copy_notice_until: float = 0.0


def set_last_copyable_output(output: str) -> None:
    """Set the last copyable output for Ctrl+Y clipboard functionality."""
    global _last_copyable_output
    _last_copyable_output = output


def _show_system_cmd() -> ShowSystemCommand:
    return ShowSystemCommand()


def _show_usage_cmd() -> ShowUsageCommand:
    return ShowUsageCommand()


def _show_markdown_cmd() -> ShowMarkdownCommand:
    return ShowMarkdownCommand()


def _show_mcp_status_cmd() -> ShowMcpStatusCommand:
    return ShowMcpStatusCommand()


def _list_tools_cmd() -> ListToolsCommand:
    return ListToolsCommand()


def _switch_agent_cmd(agent_name: str) -> SwitchAgentCommand:
    return SwitchAgentCommand(agent_name=agent_name)


def _hash_agent_cmd(agent_name: str, message: str) -> HashAgentCommand:
    return HashAgentCommand(agent_name=agent_name, message=message)


def _default_shell_command() -> str:
    """Best-effort shell choice for interactive prompt commands."""
    if platform.system() == "Windows":
        for shell_name in ["pwsh", "powershell", "cmd"]:
            shell_path = shutil.which(shell_name)
            if shell_path:
                return shell_path
        return os.environ.get("COMSPEC", "cmd.exe")

    shell_env = os.environ.get("SHELL")
    if shell_env and Path(shell_env).exists():
        return shell_env

    for shell_name in ["bash", "zsh", "sh"]:
        shell_path = shutil.which(shell_name)
        if shell_path:
            return shell_path

    return "sh"


def _show_history_cmd(target_agent: str | None) -> ShowHistoryCommand:
    return ShowHistoryCommand(agent=target_agent)


def _clear_last_cmd(target_agent: str | None) -> ClearCommand:
    return ClearCommand(kind="clear_last", agent=target_agent)


def _clear_history_cmd(target_agent: str | None) -> ClearCommand:
    return ClearCommand(kind="clear_history", agent=target_agent)


def _save_history_cmd(filename: str | None) -> SaveHistoryCommand:
    return SaveHistoryCommand(filename=filename)


def _load_history_cmd(filename: str | None, error: str | None) -> LoadHistoryCommand:
    return LoadHistoryCommand(filename=filename, error=error)


def _load_prompt_cmd(filename: str | None, error: str | None) -> LoadPromptCommand:
    return LoadPromptCommand(filename=filename, error=error)


def _history_rewind_cmd(turn_index: int | None, error: str | None) -> HistoryRewindCommand:
    return HistoryRewindCommand(turn_index=turn_index, error=error)


def _history_review_cmd(turn_index: int | None, error: str | None) -> HistoryReviewCommand:
    return HistoryReviewCommand(turn_index=turn_index, error=error)


def _history_fix_cmd(target_agent: str | None) -> HistoryFixCommand:
    return HistoryFixCommand(agent=target_agent)


def _load_agent_card_cmd(
    filename: str | None, add_tool: bool, remove_tool: bool, error: str | None
) -> LoadAgentCardCommand:
    return LoadAgentCardCommand(
        filename=filename, add_tool=add_tool, remove_tool=remove_tool, error=error
    )


def _agent_cmd(
    agent_name: str | None, add_tool: bool, remove_tool: bool, dump: bool, error: str | None
) -> AgentCommand:
    return AgentCommand(
        agent_name=agent_name,
        add_tool=add_tool,
        remove_tool=remove_tool,
        dump=dump,
        error=error,
    )


def _reload_agents_cmd() -> ReloadAgentsCommand:
    return ReloadAgentsCommand()


def _select_prompt_cmd(prompt_index: int | None, prompt_name: str | None) -> SelectPromptCommand:
    return SelectPromptCommand(prompt_index=prompt_index, prompt_name=prompt_name)


def _skills_cmd(action: str, argument: str | None) -> SkillsCommand:
    return SkillsCommand(action=action, argument=argument)


def _extract_alert_flags_from_meta(blocks) -> set[str]:
    flags: set[str] = set()
    for block in blocks or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue
        if payload.get("type") != "fast-agent-removed":
            continue
        category = payload.get("category")
        match category:
            case "text":
                flags.add("T")
            case "document":
                flags.add("D")
            case "vision":
                flags.add("V")
    return flags


# Track whether help text has been shown globally
help_message_shown = False

# Track which agents have shown their info
_agent_info_shown = set()

# One-off notices to render at the top of the prompt UI
_startup_notices: list[str] = []


def queue_startup_notice(notice: str) -> None:
    if notice:
        _startup_notices.append(notice)


async def show_mcp_status(agent_name: str, agent_provider: "AgentApp | None") -> None:
    if agent_provider is None:
        rich_print("[red]No agent provider available[/red]")
        return

    try:
        agent = agent_provider._agent(agent_name)
    except Exception as exc:
        rich_print(f"[red]Unable to load agent '{agent_name}': {exc}[/red]")
        return

    await render_mcp_status(agent)


async def _display_agent_info_helper(agent_name: str, agent_provider: "AgentApp | None") -> None:
    """Helper function to display agent information."""
    # Only show once per agent
    if agent_name in _agent_info_shown:
        return

    try:
        # Get agent info from AgentApp
        if agent_provider is None:
            return
        agent = agent_provider._agent(agent_name)

        # Get counts TODO -- add this to the type library or adjust the way aggregator/reporting works
        server_count = 0
        if isinstance(agent, McpAgentProtocol):
            server_names = agent.aggregator.server_names
            server_count = len(server_names) if server_names else 0

        tools_result = await agent.list_tools()
        tool_count = (
            len(tools_result.tools) if tools_result and hasattr(tools_result, "tools") else 0
        )

        resources_dict = await agent.list_resources()
        resource_count = (
            sum(len(resources) for resources in resources_dict.values()) if resources_dict else 0
        )

        prompts_dict = await agent.list_prompts()
        prompt_count = sum(len(prompts) for prompts in prompts_dict.values()) if prompts_dict else 0

        skill_count = 0
        skill_manifests = getattr(agent, "_skill_manifests", None)
        if skill_manifests:
            try:
                skill_count = len(list(skill_manifests))
            except TypeError:
                skill_count = 0
        tool_children = _collect_tool_children(agent)

        # Handle different agent types
        if isinstance(agent, ParallelAgent):
            # Count child agents for parallel agents
            child_count = 0
            if agent.fan_out_agents:
                child_count += len(agent.fan_out_agents)
            if agent.fan_in_agent:
                child_count += 1

            if child_count > 0:
                child_word = "child agent" if child_count == 1 else "child agents"
                rich_print(
                    f"[dim]Agent [/dim][blue]{agent_name}[/blue][dim]:[/dim] {child_count:,}[dim] {child_word}[/dim]"
                )
        elif isinstance(agent, RouterAgent):
            # Count child agents for router agents
            child_count = 0
            if agent.agents:
                child_count = len(agent.agents)

            if child_count > 0:
                child_word = "child agent" if child_count == 1 else "child agents"
                rich_print(
                    f"[dim]Agent [/dim][blue]{agent_name}[/blue][dim]:[/dim] {child_count:,}[dim] {child_word}[/dim]"
                )
        else:
            content_parts = []

            if tool_children:
                child_count = len(tool_children)
                child_word = "child agent" if child_count == 1 else "child agents"
                content_parts.append(f"{child_count:,}[dim] {child_word}[/dim]")

            if server_count > 0:
                sub_parts = []
                if tool_count > 0:
                    tool_word = "tool" if tool_count == 1 else "tools"
                    sub_parts.append(f"{tool_count:,}[dim] {tool_word}[/dim]")
                if prompt_count > 0:
                    prompt_word = "prompt" if prompt_count == 1 else "prompts"
                    sub_parts.append(f"{prompt_count:,}[dim] {prompt_word}[/dim]")
                if resource_count > 0:
                    resource_word = "resource" if resource_count == 1 else "resources"
                    sub_parts.append(f"{resource_count:,}[dim] {resource_word}[/dim]")

                server_word = "Server" if server_count == 1 else "Servers"
                server_text = f"{server_count:,}[dim] MCP {server_word}[/dim]"
                if sub_parts:
                    server_text = (
                        f"{server_text}[dim] ([/dim]"
                        + "[dim], [/dim]".join(sub_parts)
                        + "[dim])[/dim]"
                    )
                content_parts.append(server_text)

            if skill_count > 0:
                skill_word = "skill" if skill_count == 1 else "skills"
                content_parts.append(
                    f"{skill_count:,}[dim] {skill_word}[/dim][dim] available[/dim]"
                )

            if content_parts:
                content = "[dim]. [/dim]".join(content_parts)
                rich_print(f"[dim]Agent [/dim][blue]{agent_name}[/blue][dim]:[/dim] {content}")
        #               await _render_mcp_status(agent)

        # Display Skybridge status (if aggregator discovered any)
        try:
            aggregator = agent.aggregator if isinstance(agent, McpAgentProtocol) else None
            display = getattr(agent, "display", None)
            if aggregator and display and hasattr(display, "show_skybridge_summary"):
                skybridge_configs = await aggregator.get_skybridge_configs()
                display.show_skybridge_summary(agent_name, skybridge_configs)
        except Exception:
            # Ignore Skybridge rendering issues to avoid interfering with startup
            pass

        # Mark as shown
        _agent_info_shown.add(agent_name)

    except Exception:
        # Silently ignore errors to not disrupt the user experience
        pass


async def _display_all_agents_with_hierarchy(
    available_agents: Iterable[str], agent_provider: "AgentApp | None"
) -> None:
    """Display all agents with tree structure for workflow agents."""
    agent_list = list(available_agents)
    # Track which agents are children to avoid displaying them twice
    child_agents = set()

    # First pass: identify all child agents
    for agent_name in agent_list:
        try:
            if agent_provider is None:
                continue
            agent = agent_provider._agent(agent_name)

            if isinstance(agent, ParallelAgent):
                if agent.fan_out_agents:
                    for child_agent in agent.fan_out_agents:
                        if child_agent.name:
                            child_agents.add(child_agent.name)
                if agent.fan_in_agent and agent.fan_in_agent.name:
                    child_agents.add(agent.fan_in_agent.name)
            elif isinstance(agent, RouterAgent):
                if agent.agents:
                    for child_agent in agent.agents:
                        if child_agent.name:
                            child_agents.add(child_agent.name)
            else:
                tool_children = _collect_tool_children(agent)
                for child_agent in tool_children:
                    child_name = getattr(child_agent, "name", None)
                    if child_name:
                        child_agents.add(child_name)
        except Exception:
            continue

    # Second pass: display agents (parents with children, standalone agents without children)
    for agent_name in sorted(agent_list):
        # Skip if this agent is a child of another agent
        if agent_name in child_agents:
            continue

        try:
            if agent_provider is None:
                continue
            agent = agent_provider._agent(agent_name)

            # Display parent agent
            await _display_agent_info_helper(agent_name, agent_provider)

            # If it's a workflow agent, display its children
            if agent.agent_type == AgentType.PARALLEL:
                await _display_parallel_children(agent, agent_provider)
            elif agent.agent_type == AgentType.ROUTER:
                await _display_router_children(agent, agent_provider)
            else:
                tool_children = _collect_tool_children(agent)
                if tool_children:
                    await _display_tool_children(tool_children, agent_provider)

        except Exception:
            continue


async def _display_parallel_children(parallel_agent, agent_provider: "AgentApp | None") -> None:
    """Display child agents of a parallel agent in tree format."""
    children = []

    # Collect fan-out agents
    if parallel_agent.fan_out_agents:
        for child_agent in parallel_agent.fan_out_agents:
            children.append(child_agent)

    # Collect fan-in agent
    if parallel_agent.fan_in_agent is not None:
        children.append(parallel_agent.fan_in_agent)

    # Display children with tree formatting
    for i, child_agent in enumerate(children):
        is_last = i == len(children) - 1
        prefix = "└─" if is_last else "├─"
        await _display_child_agent_info(child_agent, prefix, agent_provider)


async def _display_router_children(router_agent, agent_provider: "AgentApp | None") -> None:
    """Display child agents of a router agent in tree format."""
    children = []

    # Collect routing agents
    if router_agent.agents:
        children = list(router_agent.agents)

    # Display children with tree formatting
    for i, child_agent in enumerate(children):
        is_last = i == len(children) - 1
        prefix = "└─" if is_last else "├─"
        await _display_child_agent_info(child_agent, prefix, agent_provider)


async def _display_tool_children(tool_children, agent_provider: "AgentApp | None") -> None:
    """Display tool-exposed child agents in tree format."""
    for i, child_agent in enumerate(tool_children):
        is_last = i == len(tool_children) - 1
        prefix = "└─" if is_last else "├─"
        await _display_child_agent_info(child_agent, prefix, agent_provider)


def _collect_tool_children(agent) -> list[Any]:
    """Collect child agents exposed as tools."""
    children: list[Any] = []
    child_map = getattr(agent, "_child_agents", None)
    if isinstance(child_map, dict):
        children.extend(child_map.values())
    agent_tools = getattr(agent, "_agent_tools", None)
    if isinstance(agent_tools, dict):
        children.extend(agent_tools.values())

    seen: set[str] = set()
    unique_children: list[Any] = []
    for child in children:
        name = getattr(child, "name", None)
        if not name or name in seen:
            continue
        seen.add(name)
        unique_children.append(child)
    return unique_children


async def _display_child_agent_info(
    child_agent, prefix: str, agent_provider: "AgentApp | None"
) -> None:
    """Display info for a child agent with tree prefix."""
    try:
        # Get counts for child agent
        servers = await child_agent.list_servers()
        server_count = len(servers) if servers else 0

        tools_result = await child_agent.list_tools()
        tool_count = (
            len(tools_result.tools) if tools_result and hasattr(tools_result, "tools") else 0
        )

        resources_dict = await child_agent.list_resources()
        resource_count = (
            sum(len(resources) for resources in resources_dict.values()) if resources_dict else 0
        )

        prompts_dict = await child_agent.list_prompts()
        prompt_count = sum(len(prompts) for prompts in prompts_dict.values()) if prompts_dict else 0

        # Only display if child has MCP servers
        if server_count > 0:
            # Pluralization helpers
            server_word = "Server" if server_count == 1 else "Servers"
            tool_word = "tool" if tool_count == 1 else "tools"
            resource_word = "resource" if resource_count == 1 else "resources"
            prompt_word = "prompt" if prompt_count == 1 else "prompts"

            rich_print(
                f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue][dim]:[/dim] {server_count:,}[dim] MCP {server_word}, [/dim]{tool_count:,}[dim] {tool_word}, [/dim]{resource_count:,}[dim] {resource_word}, [/dim]{prompt_count:,}[dim] {prompt_word} available[/dim]"
            )
        else:
            # Show child even without MCP servers for context
            rich_print(
                f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue][dim]: No MCP Servers[/dim]"
            )

    except Exception:
        # Fallback: just show the name
        rich_print(f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue]")


class AgentCompleter(Completer):
    """Provide completion for agent names and common commands."""

    def __init__(
        self,
        agents: list[str],
        agent_types: dict[str, AgentType] | None = None,
        is_human_input: bool = False,
        current_agent: str | None = None,
        agent_provider: "AgentApp | None" = None,
    ) -> None:
        self.agents = agents
        self.current_agent = current_agent
        self.agent_provider = agent_provider
        # Map commands to their descriptions for better completion hints
        self.commands = {
            "mcp": "Show MCP server status",
            "history": "Show conversation history overview (or /history save|load|clear|rewind|review|fix)",
            "tools": "List tools",
            "model": "Update model settings (/model reasoning <value>)",
            "skills": "Manage skills (/skills, /skills add, /skills remove, /skills registry)",
            "prompt": "Load a Prompt File or use MCP Prompt",
            "system": "Show the current system prompt",
            "usage": "Show current usage statistics",
            "markdown": "Show last assistant message without markdown formatting",
            "resume": "Resume the last session or specified session id",
            "session": "Manage sessions (/session list|new|resume|title|fork|clear)",
            "card": "Load an AgentCard (add --tool to attach/remove as tool)",
            "agent": "Attach/remove an agent as a tool or dump an AgentCard",
            "reload": "Reload AgentCards from disk",
            "help": "Show commands and shortcuts",
            "EXIT": "Exit fast-agent, terminating any running workflows",
            "STOP": "Stop this prompting session and move to next workflow step",
        }
        if is_human_input:
            self.commands.pop("prompt", None)  # Remove prompt command in human input mode
            self.commands.pop("tools", None)  # Remove tools command in human input mode
            self.commands.pop("usage", None)  # Remove usage command in human input mode
        self.agent_types = agent_types or {}

    def _resolve_completion_search(self, partial: str) -> tuple[Path, str] | None:
        if partial:
            partial_path = Path(partial)
            if partial.endswith("/") or partial.endswith(os.sep):
                search_dir = partial_path
                prefix = ""
            else:
                search_dir = (
                    partial_path.parent if partial_path.parent != partial_path else Path(".")
                )
                prefix = partial_path.name
        else:
            search_dir = Path(".")
            prefix = ""

        if not search_dir.exists():
            return None

        return search_dir, prefix

    def _iter_file_completions(
        self,
        partial: str,
        *,
        file_filter: Callable[[Path], bool],
        file_meta: Callable[[Path], str],
        include_hidden_dirs: bool = False,
    ) -> Iterable[Completion]:
        resolved = self._resolve_completion_search(partial)
        if not resolved:
            return []

        search_dir, prefix = resolved
        completions: list[Completion] = []
        try:
            for entry in sorted(search_dir.iterdir()):
                name = entry.name
                is_hidden = name.startswith(".")
                if is_hidden and not (include_hidden_dirs and entry.is_dir()):
                    continue
                if not name.lower().startswith(prefix.lower()):
                    continue

                completion_text = name if search_dir == Path(".") else str(search_dir / name)

                if entry.is_dir():
                    completions.append(
                        Completion(
                            completion_text + "/",
                            start_position=-len(partial),
                            display=name + "/",
                            display_meta="directory",
                        )
                    )
                elif entry.is_file() and file_filter(entry):
                    completions.append(
                        Completion(
                            completion_text,
                            start_position=-len(partial),
                            display=name,
                            display_meta=file_meta(entry),
                        )
                    )
        except PermissionError:
            return []

        return completions

    def _complete_history_files(self, partial: str):
        """Generate completions for history files (.json and .md)."""

        def _history_filter(entry: Path) -> bool:
            return entry.name.endswith(".json") or entry.name.endswith(".md")

        def _history_meta(entry: Path) -> str:
            return "JSON history" if entry.name.endswith(".json") else "Markdown"

        yield from self._iter_file_completions(
            partial,
            file_filter=_history_filter,
            file_meta=_history_meta,
            include_hidden_dirs=True,
        )

    def _normalize_turn_preview(self, text: str, *, limit: int = 60) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            return "<no text>"
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1] + "…"

    def _iter_user_turns(self):
        if not self.agent_provider or not self.current_agent:
            return []
        try:
            agent_obj = self.agent_provider._agent(self.current_agent)
        except Exception:
            return []
        history = getattr(agent_obj, "message_history", [])
        turns: list[list[PromptMessageExtended]] = []
        current: list[PromptMessageExtended] = []
        saw_assistant = False

        for message in list(history):
            is_new_user = message.role == "user" and not message.tool_results
            if is_new_user:
                if not current:
                    current = [message]
                    saw_assistant = False
                    continue
                if not saw_assistant:
                    current.append(message)
                    continue
                turns.append(current)
                current = [message]
                saw_assistant = False
                continue
            if current:
                current.append(message)
                if message.role == "assistant":
                    saw_assistant = True

        if current:
            turns.append(current)

        user_turns = []
        for turn in turns:
            if not turn:
                continue
            first = turn[0]
            if first.role != "user" or first.tool_results:
                continue
            user_turns.append(first)
        return user_turns

    def _resolve_reasoning_values(self) -> list[str]:
        if not self.agent_provider or not self.current_agent:
            return []
        try:
            agent_obj = self.agent_provider._agent(self.current_agent)
        except Exception:
            return []
        llm = getattr(agent_obj, "llm", None) or getattr(agent_obj, "_llm", None)
        if not llm:
            return []
        return available_reasoning_values(getattr(llm, "reasoning_effort_spec", None))

    def _complete_history_rewind(self, partial: str):
        user_turns = self._iter_user_turns()
        if not user_turns:
            return
        partial_clean = partial.strip()
        for index in range(len(user_turns), 0, -1):
            message = user_turns[index - 1]
            index_str = str(index)
            if partial_clean and not index_str.startswith(partial_clean):
                continue
            content = getattr(message, "content", []) or []
            text = None
            if content:
                from fast_agent.mcp.helpers.content_helpers import get_text

                text = get_text(content[0])
            if not text or text == "<no text>":
                text = ""
            preview = self._normalize_turn_preview(text or "")
            yield Completion(
                index_str,
                start_position=-len(partial),
                display=f"turn {index_str}",
                display_meta=preview,
            )

    def _complete_session_ids(self, partial: str):
        """Generate completions for recent session ids."""
        from fast_agent.session import (
            display_session_name,
            get_session_history_window,
            get_session_manager,
        )

        manager = get_session_manager()
        sessions = manager.list_sessions()
        limit = get_session_history_window()
        if limit > 0:
            sessions = sessions[:limit]
        partial_lower = partial.lower()
        for session_info in sessions:
            session_id = session_info.name
            display_name = display_session_name(session_id)
            if partial and not (
                session_id.lower().startswith(partial_lower)
                or display_name.lower().startswith(partial_lower)
            ):
                continue
            display_time = session_info.last_activity.strftime("%Y-%m-%d %H:%M")
            metadata = session_info.metadata or {}
            summary = (
                metadata.get("title")
                or metadata.get("label")
                or metadata.get("first_user_preview")
                or ""
            )
            summary = " ".join(str(summary).split())
            if summary:
                summary = summary[:30]
                display_meta = f"{display_time} • {summary}"
            else:
                display_meta = display_time
            yield Completion(
                session_id,
                start_position=-len(partial),
                display=display_name,
                display_meta=display_meta,
            )

    def _complete_agent_card_files(self, partial: str):
        """Generate completions for AgentCard files (.md/.markdown/.yaml/.yml)."""
        card_extensions = {".md", ".markdown", ".yaml", ".yml"}

        def _card_filter(entry: Path) -> bool:
            return entry.suffix.lower() in card_extensions

        def _card_meta(_: Path) -> str:
            return "AgentCard"

        yield from self._iter_file_completions(
            partial,
            file_filter=_card_filter,
            file_meta=_card_meta,
            include_hidden_dirs=True,
        )

    def _complete_local_skill_names(self, partial: str):
        """Generate completions for local skill names and indices."""
        from fast_agent.skills.manager import get_manager_directory
        from fast_agent.skills.registry import SkillRegistry

        manager_dir = get_manager_directory()
        manifests = SkillRegistry.load_directory(manager_dir)
        if not manifests:
            return

        partial_lower = partial.lower()
        include_numbers = not partial or partial.isdigit()
        for index, manifest in enumerate(manifests, 1):
            name = manifest.name
            if name and (not partial or name.lower().startswith(partial_lower)):
                yield Completion(
                    name,
                    start_position=-len(partial),
                    display=name,
                    display_meta="local skill",
                )
            if include_numbers:
                index_text = str(index)
                if not partial or index_text.startswith(partial):
                    yield Completion(
                        index_text,
                        start_position=-len(partial),
                        display=index_text,
                        display_meta=name or "local skill",
                    )

    def _complete_skill_registries(self, partial: str):
        """Generate completions for configured skills registries."""
        from fast_agent.config import get_settings
        from fast_agent.skills.manager import (
            DEFAULT_SKILL_REGISTRIES,
            format_marketplace_display_url,
        )

        settings = get_settings()
        skills_settings = getattr(settings, "skills", None)
        configured_urls = (
            getattr(skills_settings, "marketplace_urls", None) if skills_settings else None
        ) or list(DEFAULT_SKILL_REGISTRIES)
        active_url = getattr(skills_settings, "marketplace_url", None) if skills_settings else None
        if active_url and active_url not in configured_urls:
            configured_urls.append(active_url)

        unique_urls: list[str] = []
        seen_urls = set()
        for url in configured_urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            unique_urls.append(url)

        partial_lower = partial.lower()
        include_numbers = not partial or partial.isdigit()
        include_urls = bool(partial) and not partial.isdigit()
        for index, url in enumerate(unique_urls, 1):
            display = format_marketplace_display_url(url)
            index_text = str(index)
            if include_numbers and index_text.startswith(partial):
                yield Completion(
                    index_text,
                    start_position=-len(partial),
                    display=index_text,
                    display_meta=display,
                )
            if include_urls and url.lower().startswith(partial_lower):
                yield Completion(
                    url,
                    start_position=-len(partial),
                    display=index_text,
                    display_meta=display,
                )

    def _complete_executables(self, partial: str, max_results: int = 100):
        """Complete executable names from PATH.

        Args:
            partial: The partial executable name to match.
            max_results: Maximum number of completions to yield (default 100).
                        Limits scan time on systems with large PATH.
        """
        seen = set()
        count = 0
        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            if count >= max_results:
                break
            try:
                for entry in Path(path_dir).iterdir():
                    if count >= max_results:
                        break
                    if entry.is_file() and os.access(entry, os.X_OK):
                        name = entry.name
                        if name.startswith(partial) and name not in seen:
                            seen.add(name)
                            count += 1
                            yield Completion(
                                name,
                                start_position=-len(partial),
                                display=name,
                                display_meta="executable",
                            )
            except (PermissionError, FileNotFoundError):
                pass

    def _complete_shell_paths(self, partial: str, delete_len: int, max_results: int = 100):
        """Complete file/directory paths for shell commands.

        Args:
            partial: The partial path to complete.
            delete_len: Number of characters to delete for the completion.
            max_results: Maximum number of completions to yield (default 100).
        """
        if partial:
            partial_path = Path(partial)
            if partial.endswith("/") or partial.endswith(os.sep):
                search_dir = partial_path
                prefix = ""
            else:
                search_dir = (
                    partial_path.parent if partial_path.parent != partial_path else Path(".")
                )
                prefix = partial_path.name
        else:
            search_dir = Path(".")
            prefix = ""

        if not search_dir.exists():
            return

        try:
            count = 0
            for entry in sorted(search_dir.iterdir()):
                if count >= max_results:
                    break
                name = entry.name
                if name.startswith(".") and not prefix.startswith("."):
                    continue
                if not name.lower().startswith(prefix.lower()):
                    continue

                completion_text = str(search_dir / name) if search_dir != Path(".") else name

                if entry.is_dir():
                    yield Completion(
                        completion_text + "/",
                        start_position=-delete_len,
                        display=name + "/",
                        display_meta="directory",
                    )
                else:
                    yield Completion(
                        completion_text,
                        start_position=-delete_len,
                        display=name,
                        display_meta="file",
                    )
                count += 1
        except PermissionError:
            pass

    def get_completions(self, document, complete_event):
        """Synchronous completions method - this is what prompt_toolkit expects by default"""
        text = document.text_before_cursor
        text_lower = text.lower()
        completion_requested = bool(complete_event and complete_event.completion_requested)

        # Shell completion mode - detect ! prefix
        if text.lstrip().startswith("!"):
            if not completion_requested:
                return
            # Text after "!" with leading/trailing whitespace stripped
            shell_text = text.lstrip()[1:].lstrip()
            if not shell_text:
                return

            if " " not in shell_text:
                # First token: complete executables
                yield from self._complete_executables(shell_text)
            else:
                # After first token: complete paths
                _, path_part = shell_text.rsplit(" ", 1)
                yield from self._complete_shell_paths(path_part, len(path_part))
            return

        # Sub-completion for /history load - show .json and .md files
        if text_lower.startswith("/history load "):
            partial = text[len("/history load ") :]
            yield from self._complete_history_files(partial)
            return

        if text_lower.startswith("/history rewind "):
            partial = text[len("/history rewind ") :]
            yield from self._complete_history_rewind(partial)
            return

        if text_lower.startswith("/history review "):
            partial = text[len("/history review ") :]
            yield from self._complete_history_rewind(partial)
            return

        if text_lower.startswith("/prompt load "):
            partial = text[len("/prompt load ") :]
            yield from self._complete_history_files(partial)
            return

        if text_lower.startswith("/history clear "):
            partial = text[len("/history clear ") :]
            subcommands = {
                "all": "Clear the full history",
                "last": "Remove the most recent message",
            }
            for subcmd, description in subcommands.items():
                if subcmd.startswith(partial.lower()):
                    yield Completion(
                        subcmd,
                        start_position=-len(partial),
                        display=subcmd,
                        display_meta=description,
                    )
            return

        if text_lower.startswith("/resume "):
            partial = text[len("/resume ") :]
            yield from self._complete_session_ids(partial)
            return

        if text_lower.startswith("/session resume "):
            partial = text[len("/session resume ") :]
            yield from self._complete_session_ids(partial)
            return

        if text_lower.startswith("/session clear "):
            partial = text[len("/session clear ") :]
            if "all".startswith(partial.lower()):
                yield Completion(
                    "all",
                    start_position=-len(partial),
                    display="all",
                    display_meta="Delete all sessions",
                )
            yield from self._complete_session_ids(partial)
            return

        if text_lower.startswith("/skills "):
            remainder = text[len("/skills ") :]
            if not remainder:
                remainder = ""
            parts = remainder.split(maxsplit=1)
            subcommands = {
                "list": "List local skills",
                "add": "Install a skill",
                "remove": "Remove a local skill",
                "registry": "Set skills registry",
            }
            if not parts or (len(parts) == 1 and not remainder.endswith(" ")):
                partial = parts[0] if parts else ""
                for subcmd, description in subcommands.items():
                    if subcmd.startswith(partial.lower()):
                        yield Completion(
                            subcmd,
                            start_position=-len(partial),
                            display=subcmd,
                            display_meta=description,
                        )
                return

            subcmd = parts[0].lower()
            argument = parts[1] if len(parts) > 1 else ""
            if subcmd in {"remove", "rm", "delete", "uninstall"}:
                yield from self._complete_local_skill_names(argument)
                return
            if subcmd in {"registry", "marketplace", "source"}:
                yield from self._complete_skill_registries(argument)
                return
            return

        if text_lower.startswith("/model "):
            remainder = text[len("/model ") :]
            if not remainder:
                remainder = ""
            parts = remainder.split(maxsplit=1)
            subcommands = {
                "reasoning": "Set reasoning effort (low/medium/high/budget/off)",
            }
            if not parts or (len(parts) == 1 and not remainder.endswith(" ")):
                partial = parts[0] if parts else ""
                for subcmd, description in subcommands.items():
                    if subcmd.startswith(partial.lower()):
                        yield Completion(
                            subcmd,
                            start_position=-len(partial),
                            display=subcmd,
                            display_meta=description,
                        )
                return

            subcmd = parts[0].lower()
            argument = parts[1] if len(parts) > 1 else ""
            if subcmd == "reasoning":
                for value in self._resolve_reasoning_values():
                    if value.startswith(argument.lower()):
                        yield Completion(
                            value,
                            start_position=-len(argument),
                            display=value,
                            display_meta="reasoning",
                        )
                return

            return

        if text_lower.startswith("/history "):
            partial = text[len("/history ") :]
            subcommands = {
                "show": "Show history overview",
                "save": "Save history to a file",
                "load": "Load history from a file",
                "clear": "Clear history (all or last)",
                "rewind": "Rewind to a previous user turn",
                "review": "Review a previous user turn in full",
            }
            for subcmd, description in subcommands.items():
                if subcmd.startswith(partial.lower()):
                    yield Completion(
                        subcmd,
                        start_position=-len(partial),
                        display=subcmd,
                        display_meta=description,
                    )
            return

        if text_lower.startswith("/session "):
            partial = text[len("/session ") :]
            subcommands = {
                "clear": "Delete a session (or all)",
                "list": "List recent sessions",
                "new": "Create a new session",
                "resume": "Resume a session",
                "title": "Set session title",
                "fork": "Fork current session",
            }
            for subcmd, description in subcommands.items():
                if subcmd.startswith(partial.lower()):
                    yield Completion(
                        subcmd,
                        start_position=-len(partial),
                        display=subcmd,
                        display_meta=description,
                    )
            return

        if text_lower.startswith("/card "):
            partial = text[len("/card ") :]
            yield from self._complete_agent_card_files(partial)
            return

        # Sub-completion for /agent - show available agent names (excluding current agent)
        if text_lower.startswith("/agent "):
            partial = text[len("/agent ") :].lstrip()
            # Strip leading @ if present
            if partial.startswith("@"):
                partial = partial[1:]
            for agent in self.agents:
                # Don't suggest attaching current agent to itself
                if agent == self.current_agent:
                    continue
                if agent.lower().startswith(partial.lower()):
                    agent_type = self.agent_types.get(agent, AgentType.BASIC).value
                    yield Completion(
                        agent,
                        start_position=-len(partial),
                        display=agent,
                        display_meta=agent_type,
                    )
            return

        # Complete commands
        if text_lower.startswith("/"):
            cmd = text_lower[1:]
            # Simple command completion - match beginning of command
            for command, description in self.commands.items():
                if command.lower().startswith(cmd):
                    yield Completion(
                        command,
                        start_position=-len(cmd),
                        display=command,
                        display_meta=description,
                    )

        # Complete agent names for agent-related commands
        elif text.startswith("@"):
            agent_name = text[1:]
            for agent in self.agents:
                if agent.lower().startswith(agent_name.lower()):
                    # Get agent type or default to "Agent"
                    agent_type = self.agent_types.get(agent, AgentType.BASIC).value
                    yield Completion(
                        agent,
                        start_position=-len(agent_name),
                        display=agent,
                        display_meta=agent_type,
                    )

        if completion_requested:
            if text and not text[-1].isspace():
                partial = text.split()[-1]
            else:
                partial = ""
            yield from self._complete_shell_paths(partial, len(partial))
            return

        # Complete agent names for hash commands (#agent_name message)
        elif text.startswith("#"):
            # Only complete if we haven't finished the agent name yet (no space after #agent)
            rest = text[1:]
            if " " not in rest:
                # Still typing agent name
                agent_name = rest
                for agent in self.agents:
                    if agent.lower().startswith(agent_name.lower()):
                        # Get agent type or default to "Agent"
                        agent_type = self.agent_types.get(agent, AgentType.BASIC).value
                        yield Completion(
                            agent + " ",  # Add space after agent name for message input
                            start_position=-len(agent_name),
                            display=agent,
                            display_meta=f"# {agent_type}",
                        )


# Helper function to open text in an external editor
def get_text_from_editor(initial_text: str = "") -> str:
    """
    Opens the user\'s configured editor ($VISUAL or $EDITOR) to edit the initial_text.
    Falls back to \'nano\' (Unix) or \'notepad\' (Windows) if neither is set.
    Returns the edited text, or the original text if an error occurs.
    """
    editor_cmd_str = os.environ.get("VISUAL") or os.environ.get("EDITOR")

    if not editor_cmd_str:
        if os.name == "nt":  # Windows
            editor_cmd_str = "notepad"
        else:  # Unix-like (Linux, macOS)
            editor_cmd_str = "nano"  # A common, usually available, simple editor

    # Use shlex.split to handle editors with arguments (e.g., "code --wait")
    try:
        editor_cmd_list = shlex.split(editor_cmd_str)
        if not editor_cmd_list:  # Handle empty string from shlex.split
            raise ValueError("Editor command string is empty or invalid.")
    except ValueError as e:
        rich_print(f"[red]Error: Invalid editor command string ('{editor_cmd_str}'): {e}[/red]")
        return initial_text

    # Create a temporary file for the editor to use.
    # Using a suffix can help some editors with syntax highlighting or mode.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".txt", encoding="utf-8"
        ) as tmp_file:
            if initial_text:
                tmp_file.write(initial_text)
                tmp_file.flush()  # Ensure content is written to disk before editor opens it
            temp_file_path = tmp_file.name
    except Exception as e:
        rich_print(f"[red]Error: Could not create temporary file for editor: {e}[/red]")
        return initial_text

    try:
        # Construct the full command: editor_parts + [temp_file_path]
        # e.g., [\'vim\', \'/tmp/somefile.txt\'] or [\'code\', \'--wait\', \'/tmp/somefile.txt\']
        full_cmd = editor_cmd_list + [temp_file_path]

        # Run the editor. This is a blocking call.
        subprocess.run(full_cmd, check=True)

        # Read the content back from the temporary file.
        with open(temp_file_path, "r", encoding="utf-8") as f:
            edited_text = f.read()

    except FileNotFoundError:
        rich_print(
            f"[red]Error: Editor command '{editor_cmd_list[0]}' not found. "
            f"Please set $VISUAL or $EDITOR correctly, or install '{editor_cmd_list[0]}'.[/red]"
        )
        return initial_text
    except subprocess.CalledProcessError as e:
        rich_print(
            f"[red]Error: Editor '{editor_cmd_list[0]}' closed with an error (code {e.returncode}).[/red]"
        )
        return initial_text
    except Exception as e:
        rich_print(
            f"[red]An unexpected error occurred while launching or using the editor: {e}[/red]"
        )
        return initial_text
    finally:
        # Always attempt to clean up the temporary file.
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                rich_print(
                    f"[yellow]Warning: Could not remove temporary file {temp_file_path}: {e}[/yellow]"
                )

    return edited_text.strip()  # Added strip() to remove trailing newlines often added by editors


class ShellPrefixLexer(Lexer):
    """Lexer that highlights shell (!) and comment (#) commands."""

    def lex_document(self, document):
        def get_line_tokens(line_number):
            line = document.lines[line_number]
            stripped = line.lstrip()
            if stripped.startswith("!"):
                return [("class:shell-command", line)]
            if stripped.startswith("#"):
                return [("class:comment-command", line)]
            return [("", line)]

        return get_line_tokens


class AgentKeyBindings(KeyBindings):
    agent_provider: "AgentApp | None" = None
    current_agent_name: str | None = None


def create_keybindings(
    on_toggle_multiline: Callable[[bool], None] | None = None,
    app: Any | None = None,
    agent_provider: "AgentApp | None" = None,
    agent_name: str | None = None,
) -> AgentKeyBindings:
    """Create custom key bindings."""
    kb = AgentKeyBindings()

    def _should_start_completion(text: str) -> bool:
        stripped = text.lstrip()
        if not stripped:
            return True
        if stripped.startswith("!"):
            return bool(stripped[1:].lstrip())
        if stripped.startswith(("/", "@", "#")):
            return True
        return True

    @kb.add("c-space")
    @kb.add("c-@")
    def _(event) -> None:
        text = event.current_buffer.document.text_before_cursor
        if not _should_start_completion(text):
            return
        event.current_buffer.start_completion()

    @kb.add("c-m", filter=Condition(lambda: not in_multiline_mode))
    def _(event) -> None:
        """Enter: accept input when not in multiline mode."""
        event.current_buffer.validate_and_handle()

    @kb.add("c-j", filter=Condition(lambda: not in_multiline_mode))
    def _(event) -> None:
        """Ctrl+J: Insert newline when in normal mode."""
        event.current_buffer.insert_text("\n")

    @kb.add("c-m", filter=Condition(lambda: in_multiline_mode))
    def _(event) -> None:
        """Enter: Insert newline when in multiline mode."""
        event.current_buffer.insert_text("\n")

    # Use c-j (Ctrl+J) as an alternative to represent Ctrl+Enter in multiline mode
    @kb.add("c-j", filter=Condition(lambda: in_multiline_mode))
    def _(event) -> None:
        """Ctrl+J (equivalent to Ctrl+Enter): Submit in multiline mode."""
        event.current_buffer.validate_and_handle()

    @kb.add("c-t")
    def _(event) -> None:
        """Ctrl+T: Toggle multiline mode."""
        global in_multiline_mode
        in_multiline_mode = not in_multiline_mode

        # Force redraw the app to update toolbar
        if event.app:
            event.app.invalidate()
        elif app:
            app.invalidate()

        # Call the toggle callback if provided
        if on_toggle_multiline:
            on_toggle_multiline(in_multiline_mode)

        # Instead of printing, we'll just update the toolbar
        # The toolbar will show the current mode

    @kb.add("c-l")
    def _(event) -> None:
        """Ctrl+L: Clear and redraw the terminal screen."""
        app_ref = event.app or app
        if app_ref and getattr(app_ref, "renderer", None):
            app_ref.renderer.clear()
            app_ref.invalidate()

    @kb.add("c-u")
    def _(event) -> None:
        """Ctrl+U: Clear the input buffer."""
        event.current_buffer.text = ""

    @kb.add("c-e")
    async def _(event) -> None:
        """Ctrl+E: Edit current buffer in $EDITOR."""
        current_text = event.app.current_buffer.text
        try:
            # Run the synchronous editor function in a thread
            edited_text = await event.app.loop.run_in_executor(
                None, get_text_from_editor, current_text
            )
            event.app.current_buffer.text = edited_text
            # Optionally, move cursor to the end of the edited text
            event.app.current_buffer.cursor_position = len(edited_text)
        except asyncio.CancelledError:
            rich_print("[yellow]Editor interaction cancelled.[/yellow]")
        except Exception as e:
            rich_print(f"[red]Error during editor interaction: {e}[/red]")
        finally:
            # Ensure the UI is updated
            if event.app:
                event.app.invalidate()

    # Store reference to agent provider and agent name for clipboard functionality
    kb.agent_provider = agent_provider
    kb.current_agent_name = agent_name

    @kb.add("c-y")
    async def _(event) -> None:
        """Ctrl+Y: Copy last output (shell or assistant) to clipboard."""
        global _last_copyable_output

        def _show_copy_notice() -> None:
            global _copy_notice, _copy_notice_until
            _copy_notice = "COPIED"
            _copy_notice_until = time.monotonic() + 2.0
            if event.app:
                event.app.invalidate()
            else:
                rich_print("\n[green]✓ Copied to clipboard[/green]")

        # Priority 1: Last shell/copyable output if set
        if _last_copyable_output:
            try:
                import pyperclip

                pyperclip.copy(_last_copyable_output)
                _show_copy_notice()
                return
            except Exception:
                pass

        # Priority 2: Fall back to last assistant message
        if kb.agent_provider and kb.current_agent_name:
            try:
                # Get the agent from AgentApp
                agent = kb.agent_provider._agent(kb.current_agent_name)

                # Find last assistant message
                for msg in reversed(agent.message_history):
                    if msg.role == "assistant":
                        content = msg.last_text()
                        import pyperclip

                        pyperclip.copy(content)
                        _show_copy_notice()
                        return
            except Exception:
                pass

    return kb


def parse_special_input(text: str) -> str | CommandPayload:
    stripped = text.lstrip()
    if stripped.startswith("/"):
        cmd_line = stripped.splitlines()[0]
    else:
        cmd_line = text

    # Command processing
    if cmd_line and cmd_line.startswith("/"):
        if cmd_line == "/":
            return ""
        cmd_parts = cmd_line[1:].strip().split(maxsplit=1)
        cmd = cmd_parts[0].lower()

        if cmd == "help":
            return "HELP"
        if cmd == "system":
            return _show_system_cmd()
        if cmd == "usage":
            return _show_usage_cmd()
        if cmd == "history":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return _show_history_cmd(None)
            try:
                tokens = shlex.split(remainder)
            except ValueError:
                candidate = remainder.strip()
                return _show_history_cmd(candidate or None)
            if not tokens:
                return _show_history_cmd(None)
            subcmd = tokens[0].lower()
            argument = " ".join(tokens[1:]).strip()
            if subcmd == "show":
                return _show_history_cmd(argument or None)
            if subcmd == "save":
                return _save_history_cmd(argument or None)
            if subcmd == "load":
                if not argument:
                    return _load_history_cmd(None, "Filename required for /history load")
                return _load_history_cmd(argument, None)
            if subcmd == "review":
                if not argument:
                    return _history_review_cmd(None, "Turn number required for /history review")
                try:
                    turn_index = int(argument)
                except ValueError:
                    return _history_review_cmd(None, "Turn number must be an integer")
                return _history_review_cmd(turn_index, None)
            if subcmd == "fix":
                return _history_fix_cmd(argument or None)
            if subcmd == "rewind":
                if not argument:
                    return _history_rewind_cmd(None, "Turn number required for /history rewind")
                try:
                    turn_index = int(argument)
                except ValueError:
                    return _history_rewind_cmd(None, "Turn number must be an integer")
                return _history_rewind_cmd(turn_index, None)
            if subcmd == "clear":
                tokens = argument.split(maxsplit=1) if argument else []
                action = tokens[0].lower() if tokens else "all"
                target_agent = tokens[1].strip() if len(tokens) > 1 else None
                if action == "last":
                    return _clear_last_cmd(target_agent)
                if action == "all":
                    return _clear_history_cmd(target_agent)
                return _clear_history_cmd(argument or None)
            return _show_history_cmd(remainder)
        if cmd == "markdown":
            return _show_markdown_cmd()
        if cmd in ("save_history", "save"):
            filename = cmd_parts[1].strip() if len(cmd_parts) > 1 and cmd_parts[1].strip() else None
            return _save_history_cmd(filename)
        if cmd in ("load_history", "load"):
            filename = cmd_parts[1].strip() if len(cmd_parts) > 1 and cmd_parts[1].strip() else None
            if not filename:
                return _load_history_cmd(None, "Filename required for /history load")
            return _load_history_cmd(filename, None)
        if cmd == "resume":
            session_id = (
                cmd_parts[1].strip() if len(cmd_parts) > 1 and cmd_parts[1].strip() else None
            )
            return ResumeSessionCommand(session_id=session_id)
        if cmd == "session":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return "SESSION_HELP"
            try:
                tokens = shlex.split(remainder)
            except ValueError:
                return "SESSION_HELP"
            if not tokens:
                return "SESSION_HELP"
            subcmd = tokens[0].lower()
            argument = remainder[len(tokens[0]) :].strip()
            if subcmd == "resume":
                session_id = argument if argument else None
                return ResumeSessionCommand(session_id=session_id)
            if subcmd == "list":
                return ListSessionsCommand()
            if subcmd == "new":
                return CreateSessionCommand(session_name=argument or None)
            if subcmd == "clear":
                return ClearSessionsCommand(target=argument or None)
            if subcmd == "title":
                if not argument:
                    return TitleSessionCommand(title="")
                return TitleSessionCommand(title=argument)
            if subcmd == "fork":
                title = argument if argument else None
                return ForkSessionCommand(title=title)
            return "SESSION_HELP"
        if cmd == "card":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return _load_agent_card_cmd(None, False, False, "Filename required for /card")
            try:
                tokens = shlex.split(remainder)
            except ValueError as exc:
                return _load_agent_card_cmd(None, False, False, f"Invalid arguments: {exc}")
            add_tool = False
            remove_tool = False
            filename = None
            for token in tokens:
                if token in {"tool", "--tool", "--as-tool", "-t"}:
                    add_tool = True
                    continue
                if token in {"remove", "--remove", "--rm"}:
                    remove_tool = True
                    add_tool = True
                    continue
                if filename is None:
                    filename = token
            if not filename:
                return _load_agent_card_cmd(
                    None, add_tool, remove_tool, "Filename required for /card"
                )
            return _load_agent_card_cmd(filename, add_tool, remove_tool, None)
        if cmd == "agent":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return _agent_cmd(
                    None,
                    False,
                    False,
                    False,
                    "Usage: /agent <name> --tool | /agent [name] --dump",
                )
            try:
                tokens = shlex.split(remainder)
            except ValueError as exc:
                return _agent_cmd(None, False, False, False, f"Invalid arguments: {exc}")
            add_tool = False
            remove_tool = False
            dump = False
            agent_name = None
            unknown: list[str] = []
            for token in tokens:
                if token in {"tool", "--tool", "--as-tool", "-t"}:
                    add_tool = True
                    continue
                if token in {"remove", "--remove", "--rm"}:
                    remove_tool = True
                    add_tool = True
                    continue
                if token in {"dump", "--dump", "-d"}:
                    dump = True
                    continue
                if agent_name is None:
                    agent_name = token[1:] if token.startswith("@") else token
                    continue
                unknown.append(token)
            if unknown:
                return _agent_cmd(
                    agent_name,
                    add_tool,
                    remove_tool,
                    dump,
                    f"Unexpected arguments: {', '.join(unknown)}",
                )
            if add_tool and dump:
                return _agent_cmd(
                    agent_name,
                    add_tool,
                    remove_tool,
                    dump,
                    "Use either --tool or --dump, not both",
                )
            if not add_tool and not dump:
                return _agent_cmd(
                    agent_name,
                    add_tool,
                    remove_tool,
                    dump,
                    "Usage: /agent <name> --tool | /agent [name] --dump",
                )
            if add_tool and not agent_name:
                return _agent_cmd(
                    agent_name,
                    add_tool,
                    remove_tool,
                    dump,
                    "Agent name is required for /agent --tool",
                )
            return _agent_cmd(agent_name, add_tool, remove_tool, dump, None)
        if cmd == "reload":
            return _reload_agents_cmd()
        if cmd in ("mcpstatus", "mcp"):
            return _show_mcp_status_cmd()
        if cmd == "prompt":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return _select_prompt_cmd(None, None)
            try:
                tokens = shlex.split(remainder)
            except ValueError:
                tokens = []
            if tokens:
                subcmd = tokens[0].lower()
                argument = remainder[len(tokens[0]) :].strip()
                if subcmd == "load":
                    if not argument:
                        return _load_prompt_cmd(None, "Filename required for /prompt load")
                    return _load_prompt_cmd(argument, None)
            if remainder.lower().endswith((".json", ".md")):
                return _load_prompt_cmd(remainder, None)
            if remainder.isdigit():
                return _select_prompt_cmd(int(remainder), None)
            return _select_prompt_cmd(None, remainder)
        if cmd == "tools":
            return _list_tools_cmd()
        if cmd == "model":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return ModelReasoningCommand(value=None)
            try:
                tokens = shlex.split(remainder)
            except ValueError:
                tokens = remainder.split(maxsplit=1)
            if not tokens:
                return ModelReasoningCommand(value=None)
            subcmd = tokens[0].lower()
            argument = remainder[len(tokens[0]) :].strip()
            if subcmd == "reasoning":
                return ModelReasoningCommand(value=argument or None)
            return UnknownCommand(command=cmd_line)
        if cmd == "skills":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return _skills_cmd("list", None)
            tokens = remainder.split(maxsplit=1)
            action = tokens[0].lower()
            argument = tokens[1].strip() if len(tokens) > 1 else None
            return _skills_cmd(action, argument)
        if cmd == "exit":
            return "EXIT"
        if cmd.lower() == "stop":
            return "STOP"

        return UnknownCommand(command=cmd_line)

    if cmd_line and cmd_line.startswith("@"):
        return _switch_agent_cmd(cmd_line[1:].strip())

    # Hash command: #agent_name message - send message to agent, return result to buffer
    if cmd_line and cmd_line.startswith("#"):
        rest = cmd_line[1:].strip()
        if " " in rest:
            # Split into agent name and message
            agent_name, message = rest.split(" ", 1)
            return _hash_agent_cmd(agent_name.strip(), message.strip())
        elif rest:
            # Just agent name, no message - return empty hash command (user will be prompted)
            return _hash_agent_cmd(rest.strip(), "")

    # Shell command: ! command - execute directly in shell
    if cmd_line and cmd_line.startswith("!"):
        command = cmd_line[1:].strip()
        if command:
            return ShellCommand(command=command)
        return ShellCommand(command=_default_shell_command())

    return text


async def get_enhanced_input(
    agent_name: str,
    default: str = "",
    show_default: bool = False,
    show_stop_hint: bool = False,
    multiline: bool = False,
    available_agent_names: list[str] | None = None,
    agent_types: dict[str, AgentType] | None = None,
    is_human_input: bool = False,
    toolbar_color: str = "ansiblue",
    agent_provider: "AgentApp | None" = None,
    pre_populate_buffer: str = "",
) -> str | CommandPayload:
    """
    Enhanced input with advanced prompt_toolkit features.

    Args:
        agent_name: Name of the agent (used for prompt and history)
        default: Default value if user presses enter
        show_default: Whether to show the default value in the prompt
        show_stop_hint: Whether to show the STOP hint
        multiline: Start in multiline mode
        available_agent_names: List of agent names for auto-completion
        agent_types: Dictionary mapping agent names to their types for display
        is_human_input: Whether this is a human input request (disables agent selection features)
        toolbar_color: Color to use for the agent name in the toolbar (default: "ansiblue")
        agent_provider: Optional AgentApp for displaying agent info
        pre_populate_buffer: Text to pre-populate in the input buffer for editing (one-off)

    Returns:
        User input string or parsed command payload
    """
    global in_multiline_mode, available_agents, help_message_shown

    # Update global state
    in_multiline_mode = multiline
    if available_agent_names:
        available_agents = set(available_agent_names)
    if agent_provider:
        try:
            available_agents = set(agent_provider.agent_names())
        except Exception:
            pass

    # Get or create history object for this agent
    if agent_name not in agent_histories:
        agent_histories[agent_name] = InMemoryHistory()

    # Define callback for multiline toggle
    def on_multiline_toggle(enabled) -> None:
        nonlocal session
        if hasattr(session, "app") and session.app:
            session.app.invalidate()

    # Define toolbar function that will update dynamically
    def get_toolbar():
        global _copy_notice
        if in_multiline_mode:
            mode_style = "ansired"  # More noticeable for multiline mode
            mode_text = "MULTILINE"
        #           toggle_text = "Normal"
        else:
            mode_style = "ansigreen"
            mode_text = "NORMAL"
        #            toggle_text = "Multiline"

        # No shortcut hints in the toolbar for now
        shortcuts = []

        # Only show relevant shortcuts based on mode
        shortcuts = [(k, v) for k, v in shortcuts if v]

        shortcut_text = " | ".join(f"{key}:{action}" for key, action in shortcuts)

        # Resolve model name, turn counter, and TDV from the current agent if available
        model_display = None
        tdv_segment = None
        turn_count = 0
        agent = None
        if agent_provider:
            try:
                agent = agent_provider._agent(agent_name)
            except Exception:
                agent = None

        if agent:
            for message in agent.message_history:
                if message.role == "user":
                    turn_count += 1

            # Resolve LLM reference safely (avoid assertion when unattached)
            llm = None
            try:
                llm = agent.llm
            except AssertionError:
                llm = getattr(agent, "_llm", None)
            except Exception as exc:
                print(f"[toolbar debug] agent.llm access failed for '{agent_name}': {exc}")

            model_name = None
            if llm:
                model_name = getattr(llm, "model_name", None)
                if not model_name:
                    model_name = getattr(
                        getattr(llm, "default_request_params", None), "model", None
                    )

            if not model_name:
                model_name = agent.config.model
            if not model_name and agent.config.default_request_params:
                model_name = agent.config.default_request_params.model
            if not model_name:
                try:
                    context = agent.context
                except Exception:
                    context = None
                if context and context.config:
                    model_name = context.config.default_model

            codex_suffix = ""
            reasoning_gauge = None
            if model_name:
                display_name = format_model_display(model_name) or model_name
                if llm and getattr(llm, "provider", None) == Provider.CODEX_RESPONSES:
                    codex_suffix = " <style bg='ansiyellow'>$</style>"
                if llm:
                    try:
                        reasoning_gauge = render_reasoning_effort_gauge(
                            llm.reasoning_effort,
                            llm.reasoning_effort_spec,
                        )
                    except Exception:
                        reasoning_gauge = None
                max_len = 25
                model_display = (
                    display_name[: max_len - 1] + "…"
                    if len(display_name) > max_len
                    else display_name
                )
            else:
                if isinstance(agent, ParallelAgent):
                    parallel_models: list[str] = []
                    for fan_out_agent in agent.fan_out_agents:
                        child_llm = None
                        try:
                            child_llm = fan_out_agent.llm
                        except AssertionError:
                            child_llm = getattr(fan_out_agent, "_llm", None)
                        except Exception:
                            child_llm = None

                        child_model_name = None
                        if child_llm:
                            child_model_name = getattr(child_llm, "model_name", None)
                            if not child_model_name:
                                child_model_name = getattr(
                                    getattr(child_llm, "default_request_params", None),
                                    "model",
                                    None,
                                )
                        if not child_model_name:
                            child_model_name = fan_out_agent.config.model
                        if (
                            not child_model_name
                            and fan_out_agent.config.default_request_params
                        ):
                            child_model_name = (
                                fan_out_agent.config.default_request_params.model
                            )
                        if child_model_name:
                            display_name = (
                                format_model_display(child_model_name) or child_model_name
                            )
                            parallel_models.append(display_name)

                    if parallel_models:
                        deduped_models = list(dict.fromkeys(parallel_models))
                        display_name = ",".join(deduped_models)
                        max_len = 25
                        model_display = (
                            display_name[: max_len - 1] + "…"
                            if len(display_name) > max_len
                            else display_name
                        )
                    else:
                        model_display = "parallel"

                if not model_display:
                    print(f"[toolbar debug] no model resolved for agent '{agent_name}'")
                    model_display = "unknown"

            # Build TDV capability segment based on model database
            info = None
            if llm:
                info = ModelInfo.from_llm(llm)
            if not info and model_name:
                info = ModelInfo.from_name(model_name)

            # Default to text-only if info resolution fails for any reason
            t, d, v = (True, False, False)
            if info:
                t, d, v = info.tdv_flags

            # Check for alert flags in user messages
            alert_flags: set[str] = set()
            error_seen = False
            for message in agent.message_history:
                if message.channels:
                    if message.channels.get(FAST_AGENT_ERROR_CHANNEL):
                        error_seen = True
                if message.role == "user" and message.channels:
                    meta_blocks = message.channels.get(FAST_AGENT_REMOVED_METADATA_CHANNEL, [])
                    alert_flags.update(_extract_alert_flags_from_meta(meta_blocks))

            if error_seen and not alert_flags:
                alert_flags.add("T")

            def _style_flag(letter: str, supported: bool) -> str:
                # Enabled uses the same color as NORMAL mode (ansigreen), disabled is dim
                if letter in alert_flags:
                    return f"<style fg='ansired' bg='ansiblack'>{letter}</style>"

                enabled_color = "ansigreen"
                if supported:
                    return f"<style fg='{enabled_color}' bg='ansiblack'>{letter}</style>"
                return f"<style fg='ansiblack' bg='ansiwhite'>{letter}</style>"

            tdv_segment = f"{_style_flag('T', t)}{_style_flag('D', d)}{_style_flag('V', v)}"
        else:
            model_display = None
            tdv_segment = None

        # Build dynamic middle segments: model (in green), turn counter, and optional shortcuts
        middle_segments = []
        if model_display:
            # Model chip + inline TDV flags
            if tdv_segment:
                gauge_segment = f" {reasoning_gauge}" if reasoning_gauge else ""
                middle_segments.append(
                    f"{tdv_segment} <style bg='ansigreen'>{model_display}</style>{gauge_segment}{codex_suffix}"
                )
            else:
                gauge_segment = f" {reasoning_gauge}" if reasoning_gauge else ""
                middle_segments.append(
                    f"<style bg='ansigreen'>{model_display}</style>{gauge_segment}{codex_suffix}"
                )

        # Add turn counter (formatted as 3 digits)
        middle_segments.append(f"{turn_count:03d}")

        if shortcut_text:
            middle_segments.append(shortcut_text)
        middle = " | ".join(middle_segments)

        # Version/app label in green (dynamic version)
        version_segment = f"fast-agent {app_version}"

        # Add notifications - prioritize active events over completed ones
        from fast_agent.ui import notification_tracker

        notification_segment = ""

        # Check for active events first (highest priority)
        active_status = notification_tracker.get_active_status()
        if active_status:
            event_type = active_status["type"].upper()
            server = active_status["server"]
            notification_segment = (
                f" | <style fg='ansired' bg='ansiblack'>◀ {event_type} ({server})</style>"
            )
        elif notification_tracker.get_count() > 0:
            # Show completed events summary when no active events
            counts_by_type = notification_tracker.get_counts_by_type()
            total_events = sum(counts_by_type.values()) if counts_by_type else 0

            if len(counts_by_type) == 1:
                event_type, count = next(iter(counts_by_type.items()))
                label_text = notification_tracker.format_event_label(event_type, count)
                notification_segment = f" | ◀ {label_text}"
            else:
                summary = notification_tracker.get_summary(compact=True)
                heading = "event" if total_events == 1 else "events"
                notification_segment = f" | ◀ {total_events} {heading} ({summary})"

        copy_notice = ""
        if _copy_notice:
            if time.monotonic() < _copy_notice_until:
                copy_notice = f" | <style fg='{mode_style}' bg='ansiblack'> {_copy_notice} </style>"
            else:
                # Expire the notice once the timer elapses.
                _copy_notice = None

        if middle:
            return HTML(
                f" <style fg='{toolbar_color}' bg='ansiblack'> {agent_name} </style> "
                f" {middle} | <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
                f"{version_segment}{notification_segment}{copy_notice}"
            )
        else:
            return HTML(
                f" <style fg='{toolbar_color}' bg='ansiblack'> {agent_name} </style> "
                f"Mode: <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
                f"{version_segment}{notification_segment}{copy_notice}"
            )

    # A more terminal-agnostic style that should work across themes
    custom_style = Style.from_dict(
        {
            "completion-menu.completion": "bg:#ansiblack #ansigreen",
            "completion-menu.completion.current": "bg:#ansiblack bold #ansigreen",
            "completion-menu.meta.completion": "bg:#ansiblack #ansiblue",
            "completion-menu.meta.completion.current": "bg:#ansibrightblack #ansiblue",
            "bottom-toolbar": "#ansiblack bg:#ansigray",
            "shell-command": "#ansired",
            "comment-command": "#ansiblue",
        }
    )
    erase_when_done = True

    # Create session with history and completions
    session = PromptSession(
        history=agent_histories[agent_name],
        completer=AgentCompleter(
            agents=list(available_agents) if available_agents else [],
            agent_types=agent_types or {},
            is_human_input=is_human_input,
            current_agent=agent_name,
            agent_provider=agent_provider,
        ),
        lexer=ShellPrefixLexer(),
        complete_while_typing=True,
        multiline=Condition(lambda: in_multiline_mode),
        complete_in_thread=True,
        mouse_support=False,
        bottom_toolbar=get_toolbar,
        style=custom_style,
        erase_when_done=erase_when_done,
    )

    # Create key bindings with a reference to the app
    bindings = create_keybindings(
        on_toggle_multiline=on_multiline_toggle,
        app=session.app,
        agent_provider=agent_provider,
        agent_name=agent_name,
    )
    session.app.key_bindings = bindings

    shell_agent = None
    shell_enabled = False
    shell_access_modes: tuple[str, ...] = ()
    shell_name: str | None = None
    shell_runtime = None
    if agent_provider:
        try:
            shell_agent = agent_provider._agent(agent_name)
        except Exception:
            shell_agent = None

    if isinstance(shell_agent, McpAgentProtocol):
        direct_shell_enabled = shell_agent.shell_runtime_enabled
        shell_access_modes = shell_agent.shell_access_modes

        sub_agent_shells = [
            child
            for child in _collect_tool_children(shell_agent)
            if isinstance(child, McpAgentProtocol) and child.shell_runtime_enabled
        ]
        if sub_agent_shells:
            if direct_shell_enabled:
                if "sub-agent" not in shell_access_modes:
                    shell_access_modes = (*shell_access_modes, "sub-agent")
            else:
                shell_access_modes = ("sub-agent",)
                if len(sub_agent_shells) == 1:
                    shell_runtime = sub_agent_shells[0].shell_runtime

        shell_enabled = direct_shell_enabled or bool(sub_agent_shells)
        if direct_shell_enabled:
            shell_runtime = shell_agent.shell_runtime

        # Get the detected shell name from the runtime
        if shell_enabled and shell_runtime:
            runtime_info = shell_runtime.runtime_info()
            shell_name = runtime_info.get("name")

    def _resolve_prompt_text() -> HTML:
        buffer_text = ""
        try:
            buffer_text = session.default_buffer.text
        except Exception:
            buffer_text = ""

        if buffer_text.lstrip().startswith("!"):
            arrow_segment = "<ansired>❯</ansired>"
        else:
            arrow_segment = "<ansibrightyellow>❯</ansibrightyellow>" if shell_enabled else "❯"

        prompt_text = f"<ansibrightblue>{agent_name}</ansibrightblue> {arrow_segment} "

        # Add default value display if requested
        if show_default and default and default != "STOP":
            prompt_text = f"{prompt_text} [<ansigreen>{default}</ansigreen>] "

        return HTML(prompt_text)

    # Only show hints at startup if requested
    if show_stop_hint:
        if default == "STOP":
            rich_print("Enter a prompt, [red]STOP[/red] to finish")
            if default:
                rich_print(f"Press <ENTER> to use the default prompt:\n[cyan]{default}[/cyan]")

    # Mention available features but only on first usage globally
    if not help_message_shown:
        if is_human_input:
            rich_print("[dim]Type /help for commands. Ctrl+T toggles multiline mode.[/dim]")
        else:
            rich_print(
                "[dim]Use '/' for commands, '!' for shell. '#' to query, '@' to switch agents\nCTRL+T multiline, CTRL+Y copy last message, CTRL+E external editor.[/dim]\n"
            )

        if shell_enabled:
            modes_display = ", ".join(shell_access_modes or ("direct",))
            shell_display = f"{modes_display}, {shell_name}" if shell_name else modes_display

            # Add working directory info
            if shell_runtime:
                working_dir = shell_runtime.working_directory()
                try:
                    # Try to show relative to cwd for cleaner display
                    working_dir_display = str(working_dir.relative_to(Path.cwd()))
                    if working_dir_display == ".":
                        # Show last 2 parts of the path (e.g., "source/fast-agent")
                        parts = Path.cwd().parts
                        if len(parts) >= 2:
                            working_dir_display = "/".join(parts[-2:])
                        elif len(parts) == 1:
                            working_dir_display = parts[0]
                        else:
                            working_dir_display = str(Path.cwd())
                except ValueError:
                    # If not relative to cwd, show absolute path
                    working_dir_display = str(working_dir)
                shell_display = f"{shell_display} | cwd: {working_dir_display}"

            rich_print(format_shell_notice(shell_access_modes, shell_runtime))

            # Display agent info right after help text if agent_provider is available
            if agent_provider and not is_human_input:
                if _startup_notices:
                    for notice in _startup_notices:
                        rich_print(notice)
                    _startup_notices.clear()

                # Display info for all available agents with tree structure for workflows
                await _display_all_agents_with_hierarchy(available_agents, agent_provider)

            # Show streaming status message
            if agent_provider:
                # Get logger settings from the agent's context (not agent_provider)
                logger_settings = None
                try:
                    active_agent = shell_agent
                    if active_agent is None:
                        active_agent = agent_provider._agent(agent_name)
                    try:
                        agent_context = active_agent.context
                    except Exception:
                        agent_context = None
                    if agent_context and agent_context.config:
                        logger_settings = agent_context.config.logger
                except Exception:
                    # If we can't get the agent or its context, logger_settings stays None
                    pass

                # Only show streaming messages if chat display is enabled AND we have logger_settings
                if logger_settings:
                    show_chat = getattr(logger_settings, "show_chat", True)

                    if show_chat:
                        # Check for parallel agents
                        has_parallel = any(
                            agent.agent_type == AgentType.PARALLEL
                            for agent in agent_provider._agents.values()
                        )

                        # Note: streaming may have been disabled by fastagent.py if parallel agents exist
                        # So we check has_parallel first to show the appropriate message
                        if has_parallel:
                            # Streaming is disabled due to parallel agents
                            rich_print(
                                "[dim]Markdown Streaming disabled (Parallel Agents configured)[/dim]"
                            )
                        else:
                            # Check if streaming is enabled
                            streaming_enabled = getattr(logger_settings, "streaming_display", True)
                            streaming_mode = getattr(logger_settings, "streaming", "markdown")
                            if streaming_enabled and streaming_mode != "none":
                                # Streaming is enabled - notify users since it's experimental
                                rich_print(
                                    f"[dim]Experimental: Streaming Enabled - {streaming_mode} mode[/dim]"
                                )

                        # Show model source if configured via env var or config file
                        model_source = (
                            getattr(agent_context.config, "model_source", None)
                            if agent_context and agent_context.config
                            else None
                        )
                        if model_source:
                            rich_print(f"[dim]Model selected via {model_source}[/dim]")

                        # Show HuggingFace model and provider info if applicable
                        try:
                            if active_agent and active_agent.llm:
                                get_hf_info = getattr(active_agent.llm, "get_hf_display_info", None)
                                if get_hf_info:
                                    hf_info = get_hf_info()
                                    model = hf_info.get("model", "unknown")
                                    provider = hf_info.get("provider", "auto-routing")
                                    rich_print(f"[dim]HuggingFace: {model} via {provider}[/dim]")
                        except Exception:
                            pass

        rich_print()
        help_message_shown = True

    # Process special commands

    # Determine what to use as the buffer's initial content:
    # - pre_populate_buffer takes priority (one-off, for # command results)
    # - otherwise use the default parameter
    buffer_default = pre_populate_buffer if pre_populate_buffer else default

    # Get the input - using async version
    try:
        result = await session.prompt_async(
            _resolve_prompt_text,
            default=buffer_default,
        )
        # Echo slash command input if the prompt was erased.
        if erase_when_done:
            stripped = result.lstrip()
            if stripped.startswith("/"):
                rich_print(f"[dim]{agent_name} ❯ {stripped.splitlines()[0]}[/dim]")
            elif stripped.startswith("!"):
                rich_print(f"[dim]{agent_name} ❯ {stripped.splitlines()[0]}[/dim]")
        return parse_special_input(result)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        return "STOP"
    except EOFError:
        # Handle Ctrl+D gracefully
        return "STOP"
    except Exception as e:
        # Log and gracefully handle other exceptions
        print(f"\nInput error: {type(e).__name__}: {e}")
        return "STOP"
    finally:
        # Ensure the prompt session is properly cleaned up
        # This is especially important on Windows to prevent resource leaks
        if session.app.is_running:
            session.app.exit()


async def get_selection_input(
    prompt_text: str,
    options: list[str] | None = None,
    default: str | None = None,
    allow_cancel: bool = True,
    complete_options: bool = True,
) -> str | None:
    """
    Display a selection prompt and return the user's selection.

    Args:
        prompt_text: Text to display as the prompt
        options: List of valid options (for auto-completion)
        default: Default value if user presses enter
        allow_cancel: Whether to allow cancellation with empty input
        complete_options: Whether to use the options for auto-completion

    Returns:
        Selected value, or None if cancelled
    """
    try:
        # Initialize completer if options provided and completion requested
        completer = WordCompleter(options) if options and complete_options else None

        # Create prompt session
        prompt_session = PromptSession(completer=completer)

        try:
            # Get user input
            selection = await prompt_session.prompt_async(prompt_text, default=default or "")

            # Handle cancellation
            if allow_cancel and not selection.strip():
                return None

            return selection
        finally:
            # Ensure prompt session cleanup
            if prompt_session.app.is_running:
                prompt_session.app.exit()
    except (KeyboardInterrupt, EOFError):
        return None
    except Exception as e:
        rich_print(f"\n[red]Error getting selection: {e}[/red]")
        return None


async def get_argument_input(
    arg_name: str,
    description: str | None = None,
    required: bool = True,
) -> str | None:
    """
    Prompt for an argument value with formatting and help text.

    Args:
        arg_name: Name of the argument
        description: Optional description of the argument
        required: Whether this argument is required

    Returns:
        Input value, or None if cancelled/skipped
    """
    # Format the prompt differently based on whether it's required
    required_text = "(required)" if required else "(optional, press Enter to skip)"

    # Show description if available
    if description:
        rich_print(f"  [dim]{arg_name}: {description}[/dim]")

    prompt_text = HTML(
        f"Enter value for <ansibrightcyan>{arg_name}</ansibrightcyan> {required_text}: "
    )

    # Create prompt session
    prompt_session = PromptSession()

    try:
        # Get user input
        arg_value = await prompt_session.prompt_async(prompt_text)

        # For optional arguments, empty input means skip
        if not required and not arg_value:
            return None

        return arg_value
    except (KeyboardInterrupt, EOFError):
        return None
    except Exception as e:
        rich_print(f"\n[red]Error getting input: {e}[/red]")
        return None
    finally:
        # Ensure prompt session cleanup
        if prompt_session.app.is_running:
            prompt_session.app.exit()


async def handle_special_commands(
    command: str | CommandPayload | None, agent_app: "AgentApp | bool | None" = None
) -> bool | CommandPayload:
    """
    Handle special input commands.

    Args:
        command: The command to handle, can be string or dictionary
        agent_app: Optional agent app reference

    Returns:
        True if command was handled, False if not, or a dict with action info
    """
    # Quick guard for empty or None commands
    if not command:
        return False

    # If command is already a command payload, it has been pre-processed.
    if is_command_payload(command):
        return cast("CommandPayload", command)

    global agent_histories, available_agents

    # Check for special string commands
    if command == "HELP":
        rich_print("\n[bold]Available Commands:[/bold]")
        rich_print("  /help          - Show this help")
        rich_print("  /system        - Show the current system prompt")
        rich_print("  /prompt <name> - Load a Prompt File or use MCP Prompt")
        rich_print("  /usage         - Show current usage statistics")
        rich_print("  /skills        - List local skills for the manager directory")
        rich_print("  /skills add    - Install a skill from the marketplace")
        rich_print("  /skills remove - Remove a skill from the manager directory")
        rich_print("  /model reasoning <value> - Set reasoning effort for the current model")
        rich_print("  /history [agent_name] - Show chat history overview")
        rich_print("  /history clear all [agent_name] - Clear conversation history (keeps templates)")
        rich_print("  /history clear last [agent_name] - Remove the most recent message from history")
        rich_print("  /markdown      - Show last assistant message without markdown formatting")
        rich_print("  /mcpstatus     - Show MCP server status summary for the active agent")
        rich_print("  /history save [filename] - Save current chat history to a file")
        rich_print(
            "      [dim]Tip: Use a .json extension for MCP-compatible JSON; any other extension saves Markdown.[/dim]"
        )
        rich_print(
            "      [dim]Default: Timestamped filename (e.g., 25_01_15_14_30-conversation.json)[/dim]"
        )
        rich_print("  /history load <filename> - Load chat history from a file")
        rich_print("  /history rewind <turn> - Rewind to a prior user turn")
        rich_print("  /history review <turn> - Review a prior user turn in full")
        rich_print("  /history fix [agent_name] - Remove the last pending tool call")
        rich_print("  /resume [id|number] - Resume the last or specified session")
        rich_print("  /session list - List recent sessions")
        rich_print("  /session new [title] - Create a new session")
        rich_print("  /session resume [id|number] - Resume the last or specified session")
        rich_print("  /session title <text> - Set the current session title")
        rich_print("  /session fork [title] - Fork the current session")
        rich_print("  /session clear <id|number|all> - Delete a session or all sessions")
        rich_print(
            "  /card <filename> [--tool [remove]] - Load an AgentCard (attach/remove as tool)"
        )
        rich_print("  /agent <name> --tool [remove] - Attach/remove an agent as a tool")
        rich_print("  /agent [name] --dump - Print an AgentCard to screen")
        rich_print("  /reload        - Reload AgentCards")
        rich_print("  @agent_name    - Switch to agent")
        rich_print("  #agent_name <msg> - Send message to agent, return result to input buffer")
        rich_print("  STOP           - Return control back to the workflow")
        rich_print("  EXIT           - Exit fast-agent, terminating any running workflows")
        rich_print("\n[bold]Keyboard Shortcuts:[/bold]")
        rich_print("  Enter          - Submit (normal mode) / New line (multiline mode)")
        rich_print("  Ctrl+Enter     - Always submit (in any mode)")
        rich_print("  Ctrl+T         - Toggle multiline mode")
        rich_print("  Ctrl+E         - Edit in external editor")
        rich_print("  Ctrl+Y         - Copy last assistant response to clipboard")
        rich_print("  Ctrl+L         - Redraw the screen")
        rich_print("  Ctrl+U         - Clear input")
        rich_print("  Up/Down        - Navigate history")
        return True

    elif command == "SESSION_HELP":
        rich_print(
            "[yellow]Usage: /session list | /session new [title] | /session resume [id|number] "
            "| /session title <text> | /session fork [title] | /session clear <id|number|all>[/yellow]"
        )
        return True

    elif isinstance(command, str) and command.upper() == "EXIT":
        raise PromptExitError("User requested to exit fast-agent session")


    elif command == "SHOW_USAGE":
        return _show_usage_cmd()

    elif command == "SHOW_SYSTEM":
        return _show_system_cmd()

    elif command == "MARKDOWN":
        return _show_markdown_cmd()

    elif command == "SELECT_PROMPT" or (
        isinstance(command, str) and command.startswith("SELECT_PROMPT:")
    ):
        # Handle prompt selection UI
        if agent_app:
            # If it's a specific prompt, extract the name
            prompt_name = None
            if isinstance(command, str) and command.startswith("SELECT_PROMPT:"):
                prompt_name = command.split(":", 1)[1].strip()

            # Return a dictionary with a select_prompt action to be handled by the caller
            return _select_prompt_cmd(None, prompt_name)
        else:
            rich_print(
                "[yellow]Prompt selection is not available outside of an agent context[/yellow]"
            )
            return True

    elif isinstance(command, str) and command.startswith("SWITCH:"):
        agent_name = command.split(":", 1)[1]
        if agent_name in available_agents:
            if agent_app:
                # The parameter can be the actual agent_app or just True to enable switching
                return _switch_agent_cmd(agent_name)
            else:
                rich_print("[yellow]Agent switching not available in this context[/yellow]")
        else:
            rich_print(f"[red]Unknown agent: {agent_name}[/red]")
        return True

    return False
