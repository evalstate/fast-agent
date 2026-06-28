"""Interactive prompt command completer."""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, runtime_checkable
from urllib.parse import unquote

from mcp.types import ResourceTemplate
from prompt_toolkit.completion import Completer, Completion

from fast_agent.agents.agent_types import AgentType
from fast_agent.command_actions.accessors import (
    lookup_agent,
    plugin_command_base_path_for_provider,
    plugin_commands_for_agent,
    plugin_commands_for_provider,
)
from fast_agent.command_actions.loader import load_plugin_command_completion_function
from fast_agent.command_actions.models import (
    PluginCommandActionSpec,
    PluginCommandAgentProtocol,
    PluginCommandCompletion,
    PluginCommandCompletionContext,
)
from fast_agent.commands.command_catalog import get_command_spec
from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.model_capabilities import (
    resolve_reasoning_effort_spec,
    resolve_task_budget_supported,
    resolve_text_verbosity_spec,
)
from fast_agent.config import get_settings
from fast_agent.llm.reasoning_effort import available_reasoning_values
from fast_agent.llm.text_verbosity import available_text_verbosity_values
from fast_agent.mcp.connect_targets import (
    connect_flag_name,
    connect_flag_requires_value_token,
)
from fast_agent.mcp.provider_management import provider_managed_base_url
from fast_agent.ui.prompt.attachment_tokens import (
    FILE_MENTION_SERVER,
    URL_MENTION_SERVER,
    encode_local_attachment_reference,
)
from fast_agent.ui.prompt.resource_mentions import template_argument_names
from fast_agent.utils.async_utils import run_coroutine
from fast_agent.utils.commandline import join_commandline, split_commandline
from fast_agent.utils.text import (
    casefold_text,
    starts_with_casefold,
    strip_casefold,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable, Iterator, Sequence

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.session.session_manager import SessionManager
    from fast_agent.types import PromptMessageExtended


CompletionResultT = TypeVar("CompletionResultT")
McpConnectCompletionContext = Literal["target", "flag", "flag_value", "new_token"]
JSON_FILE_SUFFIXES = frozenset({".json"})
HISTORY_FILE_SUFFIXES = frozenset({".json", ".md"})
AGENT_CARD_FILE_SUFFIXES = frozenset({".md", ".markdown", ".yaml", ".yml"})


def _path_has_suffix(path: Path, suffixes: frozenset[str]) -> bool:
    return strip_casefold(path.suffix) in suffixes


@dataclass(frozen=True, slots=True)
class McpConnectCompletionState:
    context: McpConnectCompletionContext
    target_count: int
    partial: str


@dataclass(frozen=True, slots=True)
class MentionArgumentSection:
    template_uri: str
    argument_text: str


@dataclass(frozen=True, slots=True)
class _McpServerTargetParts:
    management: object
    url: object
    command: object
    args: object


@dataclass(frozen=True, slots=True)
class _ConfiguredMcpServers:
    configured: set[str]
    attached: set[str]
    server_targets: dict[str, str]


@dataclass(frozen=True, slots=True)
class _ManagedCompletionEntry:
    index: int
    name: str
    is_managed: bool
    local_meta: str
    managed_meta: str

    @property
    def display_meta(self) -> str:
        return self.managed_meta if self.is_managed else self.local_meta


@runtime_checkable
class _ResourceServerCapabilities(Protocol):
    @property
    def resources(self) -> object | None: ...


@runtime_checkable
class _ConnectedServerStatus(Protocol):
    @property
    def is_connected(self) -> bool | None: ...


@runtime_checkable
class _ResourceServerStatus(_ConnectedServerStatus, Protocol):
    @property
    def server_capabilities(self) -> object | None: ...


def _current_completion_token(raw_tokens: list[str], *, trailing_space: bool) -> str:
    if trailing_space or not raw_tokens:
        return ""
    return raw_tokens[-1]


def _completed_completion_tokens(raw_tokens: list[str], *, trailing_space: bool) -> list[str]:
    if trailing_space:
        return raw_tokens
    return raw_tokens[:-1]


def _mcp_server_target_values(server_config: Any) -> Mapping[str, object]:
    if isinstance(server_config, Mapping):
        return server_config
    try:
        return vars(server_config)
    except TypeError:
        return {}


def _mcp_server_target_parts(server_config: Any) -> _McpServerTargetParts:
    values = _mcp_server_target_values(server_config)
    return _McpServerTargetParts(
        management=values.get("management"),
        url=values.get("url"),
        command=values.get("command"),
        args=values.get("args"),
    )


class CompleterHistoryAgent(Protocol):
    @property
    def message_history(self) -> list["PromptMessageExtended"]: ...


class CompleterLlmAgent(Protocol):
    @property
    def llm(self) -> FastAgentLLMProtocol | None: ...


class CompleterMcpServerRegistry(Protocol):
    @property
    def registry(self) -> dict[object, object]: ...


class CompleterMcpContext(Protocol):
    @property
    def server_registry(self) -> CompleterMcpServerRegistry: ...


class CompleterMcpAggregator(Protocol):
    @property
    def context(self) -> CompleterMcpContext: ...

    def list_attached_servers(self) -> Iterable[str]: ...

    def list_configured_detached_servers(self) -> Iterable[str]: ...

    async def collect_server_status(self) -> object: ...

    async def get_capabilities(self, server_name: object) -> object: ...


class CompleterMcpAgent(Protocol):
    @property
    def aggregator(self) -> CompleterMcpAggregator: ...


class CompleterResourceAgent(Protocol):
    async def list_resources(self, *, namespace: str) -> object: ...


class CompleterResourceAggregator(CompleterMcpAggregator, Protocol):
    async def list_resource_templates(self, server_name: str) -> object: ...

    async def complete_resource_argument(
        self,
        *,
        server_name: str,
        template_uri: str,
        argument_name: str,
        value: str,
        context_args: dict[str, str] | None = None,
    ) -> object: ...


class CompleterResourceMcpAgent(Protocol):
    @property
    def aggregator(self) -> CompleterResourceAggregator: ...


class CompleterResourceArgumentCompletion(Protocol):
    @property
    def values(self) -> object: ...


def _catalog_command_description(command_name: str) -> str:
    spec = get_command_spec(command_name)
    if spec is None:
        return command_name

    examples = [f"/{spec.command}"]
    examples.extend(spec.examples)
    if not spec.examples:
        examples.extend(f"/{spec.command} {action.action}" for action in spec.actions)
    return f"{spec.summary} ({', '.join(examples)})"


class AgentCompleter(Completer):
    """Provide completion for agent names and common commands."""

    _MENTION_RE = re.compile(r"(?:^|\s)(\^[^\s]*)$")

    @dataclass(frozen=True)
    class _MentionContext:
        token: str
        kind: str
        server_name: str | None
        partial: str
        template_uri: str | None
        argument_name: str | None
        argument_value: str
        context_args: dict[str, str]

    @dataclass(frozen=True)
    class _CacheEntry:
        created_at: float
        completions: tuple[Completion, ...]

    def __init__(
        self,
        agents: list[str],
        agent_types: dict[str, AgentType] | None = None,
        is_human_input: bool = False,
        current_agent: str | None = None,
        agent_provider: "AgentApp | None" = None,
        noenv_mode: bool = False,
        cwd: Path | None = None,
        session_manager: "SessionManager | None" = None,
    ) -> None:
        self.agents = agents
        self.current_agent = current_agent
        self.agent_provider = agent_provider
        self.noenv_mode = noenv_mode
        self.cwd = cwd
        self.session_manager = session_manager
        # Map commands to their descriptions for better completion hints
        self.commands = {
            "mcp": "Manage MCP runtime servers (/mcp list|connect|disconnect|reconnect)",
            "connect": "Alias for /mcp connect with target auto-detection",
            "history": (
                "Show conversation history overview "
                "(or /history show|detail|save|load|clear|rewind|fix)"
            ),
            "compact": "Compact history into a checkpoint summary (/compact preview|prompt)",
            "tools": "List tools",
            "model": _catalog_command_description("model"),
            "models": _catalog_command_description("models"),
            "check": _catalog_command_description("check"),
            "commands": "Show command map and detailed command help",
            "skills": _catalog_command_description("skills"),
            "cards": _catalog_command_description("cards"),
            "plugins": _catalog_command_description("plugins"),
            "prompt": "Load a Prompt File or use MCP Prompt",
            "attach": "Stage file paths or remote URL attachments for the next prompt",
            "system": "Show the current system prompt",
            "usage": "Show current usage statistics",
            "markdown": "Show last assistant message without markdown formatting",
            "resume": "Resume the last session or specified session id",
            "session": "Manage sessions (/session list|new|resume|title|fork|delete|pin|unpin|export)",
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
        self._add_plugin_commands()
        self.agent_types = agent_types or {}
        self._mention_cache: dict[tuple[Any, ...], AgentCompleter._CacheEntry] = {}
        self._mention_cache_ttl_seconds = 3.0
        self._completion_wait_timeout_seconds = 1.5
        try:
            self._owner_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._owner_loop = None

    def _add_plugin_commands(self) -> None:
        if self.agent_provider is None or self.current_agent is None:
            return

        commands = {}
        global_commands = plugin_commands_for_provider(self.agent_provider)
        if global_commands:
            commands.update(global_commands)

        agent = lookup_agent(self.agent_provider, self.current_agent)
        agent_commands = plugin_commands_for_agent(agent)
        if agent_commands:
            commands.update(agent_commands)

        for name, spec in commands.items():
            if name in self.commands:
                continue
            description = spec.description
            if spec.input_hint:
                description = f"{description} {spec.input_hint}"
            if spec.key:
                description = f"{description} (key: {spec.key})"
            self.commands[name] = description

    def _current_plugin_command_spec(
        self,
        command_name: str,
    ) -> tuple[PluginCommandActionSpec, Path | None] | None:
        if self.agent_provider is None or self.current_agent is None:
            return None

        agent = lookup_agent(self.agent_provider, self.current_agent)
        agent_commands = plugin_commands_for_agent(agent)
        if agent_commands:
            spec = agent_commands.get(command_name)
            if spec is not None:
                plugin_agent = cast("PluginCommandAgentProtocol", agent)
                source_path = plugin_agent.config.source_path
                return spec, source_path.parent if source_path is not None else None

        global_commands = plugin_commands_for_provider(self.agent_provider)
        if global_commands:
            spec = global_commands.get(command_name)
            if spec is not None:
                return spec, plugin_command_base_path_for_provider(self.agent_provider)

        return None

    def _plugin_command_argument_completions(self, text: str) -> list[Completion] | None:
        if not text.startswith("/") or " " not in text or self.current_agent is None:
            return None

        command_token, arguments = text[1:].split(" ", 1)
        resolved_command = self._current_plugin_command_spec(command_token)
        if resolved_command is None:
            return None
        spec, base_path = resolved_command
        if spec is None or spec.completer is None:
            return None

        agent = lookup_agent(self.agent_provider, self.current_agent)
        if agent is None:
            return None

        try:
            raw_tokens = split_commandline(arguments)
        except ValueError:
            raw_tokens = arguments.split()
        trailing_space = arguments.endswith((" ", "\t"))
        current_token = _current_completion_token(raw_tokens, trailing_space=trailing_space)
        completed_tokens = tuple(
            _completed_completion_tokens(raw_tokens, trailing_space=trailing_space)
        )

        try:
            plugin_agent = cast("PluginCommandAgentProtocol", agent)
            completer = load_plugin_command_completion_function(spec.completer, base_path=base_path)
            values = self._run_plugin_command_completion(
                lambda: cast(
                    "Coroutine[Any, Any, list[PluginCommandCompletion | str]]",
                    completer(
                        PluginCommandCompletionContext(
                            command_name=command_token,
                            arguments=arguments,
                            current_token=current_token,
                            completed_tokens=completed_tokens,
                            agent=plugin_agent,
                            settings=get_settings(),
                            session_cwd=self.cwd,
                        )
                    ),
                ),
            )
        except Exception:
            return []
        if values is None:
            return []

        completions: list[Completion] = []
        for value in values:
            item = (
                value
                if isinstance(value, PluginCommandCompletion)
                else PluginCommandCompletion(value=str(value))
            )
            display = item.display or item.value
            detail = item.detail or ""
            if current_token and not (
                starts_with_casefold(item.value, current_token)
                or current_token.casefold() in display.casefold()
                or current_token.casefold() in detail.casefold()
            ):
                continue
            completions.append(
                Completion(
                    item.value,
                    start_position=-len(current_token),
                    display=display,
                    display_meta=detail,
                )
            )
        return completions

    def _current_agent_has_web_tools_enabled(self) -> bool:
        return history_handlers.web_tools_enabled_for_agent(self._current_llm_agent())

    @dataclass(frozen=True)
    class _CompletionSearch:
        search_dir: Path
        prefix: str
        completion_prefix: str

    def _resolve_completion_search(self, partial: str) -> _CompletionSearch | None:
        raw_dir = ""
        prefix = ""
        explicit_current_dir = False
        if partial:
            path_separators = ("/", os.sep) if os.altsep is None else ("/", os.sep, os.altsep)
            if partial.endswith(path_separators):
                raw_dir = partial
                prefix = ""
            else:
                raw_dir, prefix = os.path.split(partial)
                explicit_current_dir = partial.startswith(f".{os.sep}") or (
                    os.altsep is not None and partial.startswith(f".{os.altsep}")
                )

        raw_dir = raw_dir or "."
        expanded_dir = Path(os.path.expandvars(raw_dir)).expanduser()
        if not expanded_dir.is_absolute():
            expanded_dir = (self.cwd or Path.cwd()) / expanded_dir
        expanded_dir = expanded_dir.resolve(strict=False)
        if not expanded_dir.exists() or not expanded_dir.is_dir():
            return None

        completion_prefix = ""
        if raw_dir not in {"", "."}:
            completion_prefix = raw_dir
            if not completion_prefix.endswith(("/", os.sep)):
                completion_prefix = f"{completion_prefix}{os.sep}"
        elif explicit_current_dir:
            completion_prefix = f".{os.sep}"

        return self._CompletionSearch(
            search_dir=expanded_dir,
            prefix=prefix,
            completion_prefix=completion_prefix,
        )

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

        search_dir = resolved.search_dir
        prefix = resolved.prefix
        completion_prefix = resolved.completion_prefix
        completions: list[Completion] = []
        try:
            for entry in sorted(search_dir.iterdir()):
                name = entry.name
                is_hidden = name.startswith(".")
                if is_hidden and not (include_hidden_dirs and entry.is_dir()):
                    continue
                if not starts_with_casefold(name, prefix):
                    continue

                completion_text = f"{completion_prefix}{name}" if completion_prefix else name

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
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return []

        return completions

    def _complete_history_files(self, partial: str):
        """Generate completions for history files (.json and .md)."""

        def _history_filter(entry: Path) -> bool:
            return _path_has_suffix(entry, HISTORY_FILE_SUFFIXES)

        def _history_meta(entry: Path) -> str:
            return "JSON history" if _path_has_suffix(entry, JSON_FILE_SUFFIXES) else "Markdown"

        yield from self._iter_file_completions(
            partial,
            file_filter=_history_filter,
            file_meta=_history_meta,
            include_hidden_dirs=True,
        )

    def _complete_prompt_files(self, partial: str):
        """Generate completions for prompt files."""

        def _prompt_filter(entry: Path) -> bool:
            return entry.is_file()

        def _prompt_meta(entry: Path) -> str:
            return (
                "JSON prompt" if _path_has_suffix(entry, JSON_FILE_SUFFIXES) else "Prompt template"
            )

        yield from self._iter_file_completions(
            partial,
            file_filter=_prompt_filter,
            file_meta=_prompt_meta,
            include_hidden_dirs=True,
        )

    def _normalize_turn_preview(self, text: str, *, limit: int = 60) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            return "<no text>"
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1] + "…"

    @staticmethod
    def _starts_user_turn(message: PromptMessageExtended) -> bool:
        return message.role == "user" and not message.tool_results

    @staticmethod
    def _first_user_prompt(turn: Sequence[PromptMessageExtended]) -> PromptMessageExtended | None:
        if not turn:
            return None
        first = turn[0]
        return first if AgentCompleter._starts_user_turn(first) else None

    @staticmethod
    def _append_message_to_turns(
        turns: list[list[PromptMessageExtended]],
        current: list[PromptMessageExtended],
        message: PromptMessageExtended,
        *,
        saw_assistant: bool,
    ) -> tuple[list[PromptMessageExtended], bool]:
        if not AgentCompleter._starts_user_turn(message):
            current.append(message)
            return current, saw_assistant or message.role == "assistant"
        if not current or not saw_assistant:
            current.append(message)
            return current, False
        turns.append(current)
        return [message], False

    def _iter_user_turns(self):
        agent_obj = self._current_history_agent()
        if agent_obj is None:
            return []
        turns: list[list[PromptMessageExtended]] = []
        current: list[PromptMessageExtended] = []
        saw_assistant = False

        for message in agent_obj.message_history:
            if current or self._starts_user_turn(message):
                current, saw_assistant = self._append_message_to_turns(
                    turns,
                    current,
                    message,
                    saw_assistant=saw_assistant,
                )

        if current:
            turns.append(current)

        return [first for turn in turns if (first := self._first_user_prompt(turn)) is not None]

    def _current_history_agent(self) -> CompleterHistoryAgent | None:
        if not self.agent_provider or not self.current_agent:
            return None
        try:
            agent_obj = self.agent_provider._agent(self.current_agent)
        except Exception:
            return None
        return cast("CompleterHistoryAgent", agent_obj)

    def _current_llm_agent(self) -> CompleterLlmAgent | None:
        if not self.agent_provider or not self.current_agent:
            return None
        try:
            agent_obj = self.agent_provider._agent(self.current_agent)
        except Exception:
            return None
        return cast("CompleterLlmAgent", agent_obj)

    def _current_agent_llm(self) -> FastAgentLLMProtocol | None:
        agent_obj = self._current_llm_agent()
        if agent_obj is None:
            return None
        return agent_obj.llm

    def _resolve_reasoning_values(self) -> list[str]:
        llm = self._current_agent_llm()
        if not llm:
            return []
        values = available_reasoning_values(resolve_reasoning_effort_spec(llm))
        if "auto" in values:
            values = ["adaptive" if value == "auto" else value for value in values]
        return values

    def _supports_task_budget_setting(self) -> bool:
        llm = self._current_agent_llm()
        if llm is None:
            return False
        return resolve_task_budget_supported(llm)

    def _resolve_task_budget_values(self) -> list[str]:
        if not self._supports_task_budget_setting():
            return []
        return ["off", "20k", "64k", "128k", "256k"]

    def _resolve_verbosity_values(self) -> list[str]:
        llm = self._current_agent_llm()
        if not llm:
            return []
        return available_text_verbosity_values(resolve_text_verbosity_spec(llm))

    def _supports_web_search_setting(self) -> bool:
        llm = self._current_agent_llm()
        if llm is None:
            return False
        return model_handlers.model_supports_web_search(llm)

    def _supports_x_search_setting(self) -> bool:
        llm = self._current_agent_llm()
        if llm is None:
            return False
        return model_handlers.model_supports_x_search(llm)

    def _supports_service_tier_setting(self) -> bool:
        llm = self._current_agent_llm()
        if llm is None:
            return False
        return model_handlers.model_supports_service_tier(llm)

    def _resolve_service_tier_values(self) -> list[str]:
        llm = self._current_agent_llm()
        if llm is None:
            return []
        return list(model_handlers.service_tier_command_values(llm))

    def _supports_web_fetch_setting(self) -> bool:
        llm = self._current_agent_llm()
        if llm is None:
            return False
        return model_handlers.model_supports_web_fetch(llm)

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
            yield Completion(
                index_str,
                start_position=-len(partial),
                display=f"turn {index_str}",
                display_meta=self._turn_preview(message),
            )

    def _turn_preview(self, message: "PromptMessageExtended") -> str:
        content = message.content or []
        text = ""
        if content:
            from fast_agent.mcp.helpers.content_helpers import get_text

            text = get_text(content[0])
        if not text or text == "<no text>":
            text = ""
        return self._normalize_turn_preview(text)

    def _complete_session_ids(self, partial: str, *, start_position: int | None = None):
        """Generate completions for recent session ids."""
        if self.noenv_mode:
            return

        from fast_agent.session import (
            apply_session_window,
            display_session_name,
            get_session_manager,
        )
        from fast_agent.session.formatting import extract_session_title

        manager = self.session_manager or get_session_manager()
        sessions = apply_session_window(manager.list_sessions())
        for session_info in sessions:
            session_id = session_info.name
            display_name = display_session_name(session_id)
            if partial and not (
                starts_with_casefold(session_id, partial)
                or starts_with_casefold(display_name, partial)
            ):
                continue
            display_time = session_info.last_activity.strftime("%Y-%m-%d %H:%M")
            summary = extract_session_title(session_info.metadata)
            if summary:
                summary = summary[:30]
                display_meta = f"{display_time} • {summary}"
            else:
                display_meta = display_time
            yield Completion(
                session_id,
                start_position=-len(partial) if start_position is None else start_position,
                display=display_name,
                display_meta=display_meta,
            )

    def _complete_agent_card_files(self, partial: str):
        """Generate completions for AgentCard files (.md/.markdown/.yaml/.yml)."""

        def _card_filter(entry: Path) -> bool:
            return _path_has_suffix(entry, AGENT_CARD_FILE_SUFFIXES)

        def _card_meta(_: Path) -> str:
            return "AgentCard"

        yield from self._iter_file_completions(
            partial,
            file_filter=_card_filter,
            file_meta=_card_meta,
            include_hidden_dirs=True,
        )

    @staticmethod
    def _managed_item_completions(
        entries: Iterable[_ManagedCompletionEntry],
        partial: str,
        *,
        managed_only: bool = False,
        include_indices: bool = True,
    ) -> Iterator[Completion]:
        include_numbers = include_indices and (not partial or partial.isdigit())
        for entry in entries:
            if managed_only and not entry.is_managed:
                continue

            if entry.name and (not partial or starts_with_casefold(entry.name, partial)):
                yield Completion(
                    entry.name,
                    start_position=-len(partial),
                    display=entry.name,
                    display_meta=entry.display_meta,
                )

            if include_numbers:
                index_text = str(entry.index)
                if not partial or index_text.startswith(partial):
                    yield Completion(
                        index_text,
                        start_position=-len(partial),
                        display=index_text,
                        display_meta=entry.name,
                    )

    def _complete_local_skill_names(
        self,
        partial: str,
        *,
        managed_only: bool = False,
        include_indices: bool = True,
    ):
        """Generate completions for local skill names and indices."""
        from fast_agent.skills.provenance import read_installed_skill_source
        from fast_agent.skills.registry import SkillRegistry
        from fast_agent.skills.scope import get_manager_directory

        managed_skills_dir = get_manager_directory()
        manifests = SkillRegistry.load_directory(managed_skills_dir)
        if not manifests:
            return

        entries = (
            _ManagedCompletionEntry(
                index=index,
                name=manifest.name,
                is_managed=(
                    read_installed_skill_source(Path(manifest.path).parent).source is not None
                ),
                local_meta="local skill",
                managed_meta="managed skill",
            )
            for index, manifest in enumerate(manifests, 1)
        )
        yield from self._managed_item_completions(
            entries,
            partial,
            managed_only=managed_only,
            include_indices=include_indices,
        )

    def _complete_skill_registries(self, partial: str):
        """Generate completions for configured and MCP-backed skills registries."""
        from fast_agent.skills.configuration import (
            format_marketplace_display_url,
            resolve_skill_registries,
        )

        configured_urls = resolve_skill_registries(get_settings())
        yield from self._complete_registry_urls(
            partial,
            configured_urls=configured_urls,
            display_formatter=format_marketplace_display_url,
        )
        yield from self._complete_mcp_skill_registries(partial, offset=len(configured_urls))

    def _complete_mcp_skill_registries(self, partial: str, *, offset: int = 0):
        """Generate completions for live MCP skill registries."""
        cache_key = (
            "mcp_skill_registries",
            self.current_agent,
            partial,
            offset,
        )
        cached = self._completion_cache_get(cache_key)
        if cached is not None:
            yield from cached
            return

        choices = self._list_mcp_skill_registry_choices()
        partial_lower = partial.lower()
        include_numbers = not partial or partial.isdigit()
        include_servers = bool(partial) and not partial.isdigit()
        completions: list[Completion] = []

        for index, (server_name, display, skill_count) in enumerate(choices, offset + 1):
            skill_word = "skill" if skill_count == 1 else "skills"
            meta = f"{display} ({skill_count:,} {skill_word})"
            index_text = str(index)
            if include_numbers and index_text.startswith(partial):
                completions.append(
                    Completion(
                        index_text,
                        start_position=-len(partial),
                        display=index_text,
                        display_meta=meta,
                    )
                )
            if include_servers:
                if partial_lower.startswith("mcp://"):
                    source = f"mcp://{server_name}"
                    if source.lower().startswith(partial_lower):
                        completions.append(
                            Completion(
                                source,
                                start_position=-len(partial),
                                display=server_name,
                                display_meta=meta,
                            )
                        )
                elif server_name.lower().startswith(partial_lower):
                    completions.append(
                        Completion(
                            server_name,
                            start_position=-len(partial),
                            display=server_name,
                            display_meta=meta,
                        )
                    )

        self._completion_cache_put(cache_key, completions)
        yield from completions

    def _complete_registry_urls(
        self,
        partial: str,
        *,
        configured_urls: "Sequence[str]",
        display_formatter: "Callable[[str], str]",
    ):
        """Generate index/url completions for a registry URL list."""
        include_numbers = not partial or partial.isdigit()
        include_urls = bool(partial) and not partial.isdigit()

        for index, url in enumerate(configured_urls, 1):
            display = display_formatter(url)
            index_text = str(index)
            if include_numbers and index_text.startswith(partial):
                yield Completion(
                    index_text,
                    start_position=-len(partial),
                    display=index_text,
                    display_meta=display,
                )
            if include_urls and starts_with_casefold(url, partial):
                yield Completion(
                    url,
                    start_position=-len(partial),
                    display=index_text,
                    display_meta=display,
                )

    def _complete_registry_paths(self, partial: str):
        """Generate filesystem path completions for registry arguments."""
        candidate = partial.strip()
        if not candidate or "://" in candidate:
            return

        yield from self._complete_shell_paths(candidate, len(candidate))

    def _complete_local_card_pack_names(
        self,
        partial: str,
        *,
        managed_only: bool = False,
        include_indices: bool = True,
    ):
        """Generate completions for installed card packs."""
        from fast_agent.cards.manager import list_local_card_packs
        from fast_agent.paths import resolve_environment_paths

        env_paths = resolve_environment_paths(get_settings())
        packs = list_local_card_packs(environment_paths=env_paths)
        if not packs:
            return

        entries = (
            _ManagedCompletionEntry(
                index=entry.index,
                name=entry.name,
                is_managed=entry.source is not None,
                local_meta="local card pack",
                managed_meta="managed card pack",
            )
            for entry in packs
        )
        yield from self._managed_item_completions(
            entries,
            partial,
            managed_only=managed_only,
            include_indices=include_indices,
        )

    def _complete_card_registries(self, partial: str):
        """Generate completions for configured card registries."""
        from fast_agent.cards.manager import format_marketplace_display_url, resolve_card_registries

        configured_urls = resolve_card_registries(get_settings())
        yield from self._complete_registry_urls(
            partial,
            configured_urls=configured_urls,
            display_formatter=format_marketplace_display_url,
        )

    def _complete_local_plugin_names(
        self,
        partial: str,
        *,
        managed_only: bool = False,
        include_indices: bool = True,
    ):
        """Generate completions for installed plugins."""
        from fast_agent.paths import resolve_environment_paths
        from fast_agent.plugins.operations import list_local_plugins

        env_paths = resolve_environment_paths(get_settings())
        plugins = list_local_plugins(destination_root=env_paths.plugins)
        if not plugins:
            return

        entries = (
            _ManagedCompletionEntry(
                index=entry.index,
                name=entry.name,
                is_managed=entry.source is not None,
                local_meta="local plugin",
                managed_meta="managed plugin",
            )
            for entry in plugins
        )
        yield from self._managed_item_completions(
            entries,
            partial,
            managed_only=managed_only,
            include_indices=include_indices,
        )

    def _complete_plugin_registries(self, partial: str):
        """Generate completions for configured plugin registries."""
        from fast_agent.plugins.configuration import resolve_registries

        configured_urls = resolve_registries(get_settings())
        yield from self._complete_registry_urls(
            partial,
            configured_urls=configured_urls,
            display_formatter=lambda value: value,
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

    def _is_shell_path_token(self, token: str) -> bool:
        if not token:
            return False
        if token.startswith((".", "~", os.sep)):
            return True
        if os.sep in token:
            return True
        return bool(os.altsep and os.altsep in token)

    @staticmethod
    def _shell_current_token(shell_text: str) -> str:
        quote_char: str | None = None
        escaped = False
        token_start = 0
        for index, char in enumerate(shell_text):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if quote_char is not None:
                if char == quote_char:
                    quote_char = None
                continue
            if char in {"'", '"'}:
                quote_char = char
                continue
            if char.isspace():
                token_start = index + 1
        return shell_text[token_start:]

    @staticmethod
    def _quote_shell_path_completion(completion: Completion) -> Completion:
        return Completion(
            join_commandline([completion.text]),
            start_position=completion.start_position,
            display=completion.display,
            display_meta=completion.display_meta,
        )

    def _quoted_shell_path_completions(self, partial: str, delete_len: int) -> list[Completion]:
        return [
            self._quote_shell_path_completion(completion)
            for completion in self._complete_shell_paths(partial, delete_len)
        ]

    def _complete_shell_paths(self, partial: str, delete_len: int, max_results: int = 100):
        """Complete file/directory paths for shell commands.

        Args:
            partial: The partial path to complete.
            delete_len: Number of characters to delete for the completion.
            max_results: Maximum number of completions to yield (default 100).
        """
        resolved = self._resolve_completion_search(partial)
        if not resolved:
            return

        search_dir = resolved.search_dir
        prefix = resolved.prefix
        completion_prefix = resolved.completion_prefix

        try:
            count = 0
            for entry in sorted(search_dir.iterdir()):
                if count >= max_results:
                    break
                name = entry.name
                if name.startswith(".") and not prefix.startswith("."):
                    continue
                if not starts_with_casefold(name, prefix):
                    continue

                completion_text = f"{completion_prefix}{name}" if completion_prefix else name

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
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            pass

    def _complete_local_attachment_paths(self, partial: str) -> list[Completion]:
        decoded_partial = unquote(partial)
        if starts_with_casefold(decoded_partial, "file://"):
            from fast_agent.ui.prompt.attachment_tokens import normalize_local_attachment_reference

            try:
                decoded_partial = str(
                    normalize_local_attachment_reference(decoded_partial, cwd=self.cwd)
                )
            except ValueError:
                return []

        resolved = self._resolve_completion_search(decoded_partial)
        if not resolved:
            return []

        search_dir = resolved.search_dir
        prefix = resolved.prefix
        completion_prefix = resolved.completion_prefix
        completions: list[Completion] = []
        try:
            for entry in sorted(search_dir.iterdir()):
                name = entry.name
                if name.startswith(".") and not prefix.startswith("."):
                    continue
                if not starts_with_casefold(name, prefix):
                    continue

                completion_text = f"{completion_prefix}{name}" if completion_prefix else name
                if entry.is_dir():
                    completion_text += "/"

                completions.append(
                    Completion(
                        encode_local_attachment_reference(completion_text),
                        start_position=-len(partial),
                        display=name + ("/" if entry.is_dir() else ""),
                        display_meta="directory" if entry.is_dir() else "file",
                    )
                )
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return []

        return completions

    def _complete_subcommands(
        self,
        parts: Sequence[str],
        remainder: str,
        subcommands: Mapping[str, str],
    ) -> Iterator[Completion]:
        """Yield completions for subcommand names from a dict.

        Args:
            parts: Split parts of the remainder text
            remainder: Full remainder text after the command prefix
            subcommands: Dict mapping subcommand names to descriptions
        """
        if not parts or (len(parts) == 1 and not remainder.endswith(" ")):
            partial = parts[0] if parts else ""
            for subcmd, description in subcommands.items():
                if starts_with_casefold(subcmd, partial):
                    yield Completion(
                        subcmd,
                        start_position=-len(partial),
                        display=subcmd,
                        display_meta=description,
                    )

    @staticmethod
    def _configured_mcp_server_target(server_config: Any) -> str | None:
        target_parts = _mcp_server_target_parts(server_config)

        if isinstance(target_parts.url, str):
            normalized = target_parts.url.strip()
            if normalized:
                if target_parts.management == "provider":
                    return provider_managed_base_url(normalized)
                return normalized

        if isinstance(target_parts.command, str):
            command = target_parts.command.strip()
            if command:
                args: list[str] = []
                if isinstance(target_parts.args, list):
                    args = [str(value) for value in target_parts.args]
                return join_commandline([command, *args], syntax="posix")

        return None

    @staticmethod
    def _format_mcp_server_meta(target: str | None) -> str:
        if not target:
            return ""
        if len(target) > 80:
            return f"{target[:79]}…"
        return target

    def _runtime_mcp_servers(self) -> _ConfiguredMcpServers:
        configured: set[str] = set()
        attached: set[str] = set()
        server_targets: dict[str, str] = {}

        if self.agent_provider is None or not self.current_agent:
            return _ConfiguredMcpServers(configured, attached, server_targets)

        try:
            agent = cast("CompleterMcpAgent", lookup_agent(self.agent_provider, self.current_agent))
            if agent is None:
                return _ConfiguredMcpServers(configured, attached, server_targets)
            aggregator = agent.aggregator
            attached.update(aggregator.list_attached_servers())
            configured.update(aggregator.list_configured_detached_servers())
            registry_data = aggregator.context.server_registry.registry
        except Exception:
            return _ConfiguredMcpServers(configured, attached, server_targets)

        self._add_registry_mcp_servers(registry_data, configured, server_targets)
        return _ConfiguredMcpServers(configured, attached, server_targets)

    def _settings_mcp_servers(self) -> _ConfiguredMcpServers:
        configured: set[str] = set()
        server_targets: dict[str, str] = {}
        try:
            settings = get_settings()
        except Exception:
            return _ConfiguredMcpServers(configured, set(), server_targets)
        if settings.mcp is None:
            return _ConfiguredMcpServers(configured, set(), server_targets)
        self._add_registry_mcp_servers(
            settings.mcp.servers,
            configured,
            server_targets,
        )
        return _ConfiguredMcpServers(configured, set(), server_targets)

    def _add_registry_mcp_servers(
        self,
        registry_data: object,
        configured: set[str],
        server_targets: dict[str, str],
    ) -> None:
        if not isinstance(registry_data, dict):
            return
        for name, server_config in registry_data.items():
            server_name = str(name)
            configured.add(server_name)
            target = self._configured_mcp_server_target(server_config)
            if target is not None:
                server_targets[server_name] = target

    def _list_configured_mcp_servers(self) -> list[tuple[str, str | None]]:
        runtime_servers = self._runtime_mcp_servers()
        settings_servers = self._settings_mcp_servers()

        configured = runtime_servers.configured | settings_servers.configured
        configured.difference_update(runtime_servers.attached)
        server_targets = {
            **settings_servers.server_targets,
            **runtime_servers.server_targets,
        }

        return [
            (server_name, server_targets.get(server_name)) for server_name in sorted(configured)
        ]

    def _complete_configured_mcp_servers(self, partial: str):
        for server_name, server_url in self._list_configured_mcp_servers():
            if partial and not starts_with_casefold(server_name, partial):
                continue
            yield Completion(
                server_name,
                start_position=-len(partial),
                display=server_name,
                display_meta=self._format_mcp_server_meta(server_url),
            )

    def _mcp_connect_target_hint(self, partial: str) -> Completion:
        return Completion(
            partial,
            start_position=-len(partial),
            display="[url|npx|uvx|stdio]",
            display_meta="enter url, npx/uvx, or stdio cmd",
        )

    @staticmethod
    def _mcp_connect_context(remainder: str) -> McpConnectCompletionState:
        """Classify completion context for `/mcp connect ...`.

        `target_count` is the number of fully-formed target tokens before
        `partial`, which is the token currently being edited.
        """

        raw_tokens = remainder.split()
        trailing_space = remainder.endswith(" ")
        partial = _current_completion_token(raw_tokens, trailing_space=trailing_space)
        complete_tokens = _completed_completion_tokens(
            raw_tokens,
            trailing_space=trailing_space,
        )

        target_count = 0
        waiting_for_flag_value = False
        for token in complete_tokens:
            if waiting_for_flag_value:
                waiting_for_flag_value = False
                continue
            flag_name = connect_flag_name(token)
            if connect_flag_requires_value_token(token):
                waiting_for_flag_value = True
                continue
            if flag_name is not None:
                continue
            target_count += 1

        if trailing_space:
            return McpConnectCompletionState(
                context="flag_value" if waiting_for_flag_value else "new_token",
                target_count=target_count,
                partial=partial,
            )

        if waiting_for_flag_value:
            return McpConnectCompletionState(
                context="flag_value",
                target_count=target_count,
                partial=partial,
            )
        if connect_flag_name(partial) is not None or partial.startswith("--"):
            return McpConnectCompletionState(
                context="flag",
                target_count=target_count,
                partial=partial,
            )
        return McpConnectCompletionState(
            context="target",
            target_count=target_count,
            partial=partial,
        )

    def _current_agent_object(self) -> object | None:
        if self.agent_provider is None or not self.current_agent:
            return None
        try:
            return self.agent_provider._agent(self.current_agent)
        except Exception:
            return None

    @staticmethod
    def _feature_enabled(value: object | None) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        try:
            return bool(value)
        except Exception:
            return True

    @staticmethod
    def _resource_capability_enabled(capabilities: object | None) -> bool:
        if not isinstance(capabilities, _ResourceServerCapabilities):
            return False
        return AgentCompleter._feature_enabled(capabilities.resources)

    @staticmethod
    def _resource_server_status_enabled(status: object) -> bool:
        return (
            isinstance(status, _ResourceServerStatus)
            and status.is_connected is True
            and AgentCompleter._resource_capability_enabled(status.server_capabilities)
        )

    @staticmethod
    def _resource_server_names_from_status(status_map: object) -> list[str] | None:
        if not isinstance(status_map, dict):
            return None
        names = {
            str(server_name)
            for server_name, status in status_map.items()
            if AgentCompleter._resource_server_status_enabled(status)
        }
        return sorted(names)

    async def _resource_server_names_from_status_api(
        self,
        aggregator: CompleterMcpAggregator,
    ) -> list[str] | None:
        try:
            status_map = await aggregator.collect_server_status()
        except Exception:
            return None
        return self._resource_server_names_from_status(status_map)

    async def _resource_server_names_from_capabilities_api(
        self,
        aggregator: CompleterMcpAggregator,
    ) -> list[str]:
        names: set[str] = set()
        try:
            attached_servers = aggregator.list_attached_servers()
        except Exception:
            return []

        for server_name in attached_servers:
            try:
                capabilities = await aggregator.get_capabilities(server_name)
            except Exception:
                continue
            if self._resource_capability_enabled(capabilities):
                names.add(str(server_name))
        return sorted(names)

    async def _list_connected_resource_servers(self) -> list[str]:
        agent = cast("CompleterMcpAgent | None", self._current_agent_object())
        if agent is None:
            return []

        try:
            aggregator = agent.aggregator
        except Exception:
            return []

        names = await self._resource_server_names_from_status_api(aggregator)
        if names is not None:
            return names
        return await self._resource_server_names_from_capabilities_api(aggregator)

    def _completion_cache_get(self, key: tuple[Any, ...]) -> tuple[Completion, ...] | None:
        cached = self._mention_cache.get(key)
        if cached is None:
            return None
        if (time.monotonic() - cached.created_at) > self._mention_cache_ttl_seconds:
            self._mention_cache.pop(key, None)
            return None
        return cached.completions

    def _completion_cache_put(self, key: tuple[Any, ...], completions: list[Completion]) -> None:
        self._mention_cache[key] = self._CacheEntry(
            created_at=time.monotonic(),
            completions=tuple(completions),
        )

    def _run_async_completion(
        self, create_awaitable: Callable[[], Coroutine[Any, Any, CompletionResultT]]
    ) -> CompletionResultT | None:
        owner_loop = self._owner_loop
        if owner_loop is not None and owner_loop.is_running():
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None

            if current_loop is owner_loop:
                return None

            if current_loop is not owner_loop:
                awaitable = create_awaitable()
                future = asyncio.run_coroutine_threadsafe(awaitable, owner_loop)
                try:
                    return future.result(timeout=self._completion_wait_timeout_seconds)
                except FuturesTimeoutError:
                    future.cancel()
                    return None
                except Exception:
                    return None

        try:
            awaitable = create_awaitable()
            return run_coroutine(awaitable)
        except Exception:
            return None

    def _run_plugin_command_completion(
        self,
        create_awaitable: Callable[
            [],
            Coroutine[Any, Any, list[PluginCommandCompletion | str]],
        ],
    ) -> list[PluginCommandCompletion | str] | None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is None or not current_loop.is_running():
            try:
                return run_coroutine(create_awaitable())
            except Exception:
                return None

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fast-agent-completer")
        future = executor.submit(lambda: run_coroutine(create_awaitable()))
        try:
            return future.result(timeout=self._completion_wait_timeout_seconds)
        except FuturesTimeoutError:
            future.cancel()
            return None
        except Exception:
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _list_server_resource_uris(self, server_name: str) -> list[str]:
        agent = cast("CompleterResourceAgent | None", self._current_agent_object())
        if agent is None:
            return []

        try:
            result = await agent.list_resources(namespace=server_name)
        except Exception:
            return []

        uris = self._server_result_list(result, server_name)
        return [str(uri) for uri in uris]

    def _list_mcp_skill_registry_choices(self) -> list[tuple[str, str, int]]:
        agent = self._current_agent_object()
        if agent is None:
            return []

        aggregator = getattr(agent, "aggregator", None)
        cached_registries = getattr(aggregator, "cached_mcp_skill_registries", None)
        if not callable(cached_registries):
            return []

        try:
            registries = cached_registries()
        except Exception:
            return []

        choices: list[tuple[str, str, int]] = []
        for registry in registries:
            server_name = getattr(registry, "server_name", None)
            if not isinstance(server_name, str) or not server_name:
                continue
            display_name = getattr(registry, "display_name", None)
            display = display_name if isinstance(display_name, str) else f"mcp-server {server_name}"
            skills = getattr(registry, "skills", ())
            try:
                skill_count = len(skills)
            except TypeError:
                skill_count = 0
            choices.append((server_name, display, skill_count))

        return sorted(choices, key=lambda choice: choice[0].lower())

    async def _list_server_resource_templates(self, server_name: str) -> list[ResourceTemplate]:
        agent = cast("CompleterResourceMcpAgent | None", self._current_agent_object())
        if agent is None:
            return []

        try:
            result = await agent.aggregator.list_resource_templates(server_name)
        except Exception:
            return []

        templates = self._server_result_list(result, server_name)
        return [template for template in templates if isinstance(template, ResourceTemplate)]

    @staticmethod
    def _server_result_list(result: object, server_name: str) -> list[object]:
        if not isinstance(result, dict):
            return []
        result_by_server = cast("dict[str, object]", result)
        values = result_by_server.get(server_name)
        if not isinstance(values, list):
            return []
        return list(values)

    async def _complete_server_template_argument(
        self,
        server_name: str,
        template_uri: str,
        argument_name: str,
        value: str,
        context_args: dict[str, str] | None = None,
    ) -> list[str]:
        agent = cast("CompleterResourceMcpAgent | None", self._current_agent_object())
        if agent is None:
            return []

        try:
            completion = cast(
                "CompleterResourceArgumentCompletion",
                await agent.aggregator.complete_resource_argument(
                    server_name=server_name,
                    template_uri=template_uri,
                    argument_name=argument_name,
                    value=value,
                    context_args=context_args,
                ),
            )
            values = completion.values
        except Exception:
            return []
        if not isinstance(values, list):
            return []
        return [str(item) for item in values]

    @staticmethod
    def _split_mention_argument_section(remainder: str) -> MentionArgumentSection | None:
        """Split a mention remainder into template URI and argument text.

        Returns ``None`` when no trailing argument section has been started.
        The argument section starts at the first unmatched ``{`` from the end,
        while URI-template placeholders remain balanced ``{...}`` segments.
        """

        open_index: int | None = None
        depth = 0

        for index, char in enumerate(remainder):
            if char == "{":
                if depth == 0:
                    open_index = index
                depth += 1
                continue
            if char == "}" and depth > 0:
                depth -= 1
                if depth == 0:
                    open_index = None

        if depth == 0 or open_index is None:
            return None
        if depth != 1:
            return None

        return MentionArgumentSection(
            template_uri=remainder[:open_index],
            argument_text=remainder[open_index + 1 :],
        )

    @classmethod
    def _mention_context(
        cls,
        *,
        token: str,
        kind: str,
        server_name: str | None = None,
        partial: str = "",
        template_uri: str | None = None,
        argument_name: str | None = None,
        argument_value: str = "",
        context_args: dict[str, str] | None = None,
    ) -> _MentionContext:
        return cls._MentionContext(
            token=token,
            kind=kind,
            server_name=server_name,
            partial=partial,
            template_uri=template_uri,
            argument_name=argument_name,
            argument_value=argument_value,
            context_args=context_args or {},
        )

    @staticmethod
    def _completed_mention_arguments(argument_text: str) -> tuple[dict[str, str], str]:
        context_args: dict[str, str] = {}
        raw_segments = [segment.strip() for segment in argument_text.strip().split(",")]
        if not raw_segments:
            return context_args, ""

        for segment in raw_segments[:-1]:
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            key = key.strip()
            if key:
                context_args[key] = value.strip()

        return context_args, raw_segments[-1]

    def _argument_mention_context(
        self,
        *,
        token: str,
        server_name: str,
        split_result: MentionArgumentSection,
    ) -> _MentionContext | None:
        argument_text = split_result.argument_text
        if "}" in argument_text:
            return None

        context_args, current_segment = self._completed_mention_arguments(argument_text)
        if not current_segment or "=" not in current_segment:
            return self._mention_context(
                token=token,
                kind="argument_name",
                server_name=server_name,
                partial=current_segment,
                template_uri=split_result.template_uri,
                context_args=context_args,
            )

        argument_name, value_partial = current_segment.split("=", 1)
        return self._mention_context(
            token=token,
            kind="argument_value",
            server_name=server_name,
            partial=value_partial,
            template_uri=split_result.template_uri,
            argument_name=argument_name.strip(),
            argument_value=value_partial,
            context_args=context_args,
        )

    def _mention_context_for_text(self, text: str) -> _MentionContext | None:
        match = self._MENTION_RE.search(text)
        if not match:
            return None

        token = match.group(1)
        payload = token[1:]
        if ":" not in payload:
            return self._mention_context(token=token, kind="server", partial=payload)

        server_name, remainder = payload.split(":", 1)
        if not server_name:
            return None

        split_result = self._split_mention_argument_section(remainder)
        if split_result is None:
            return self._mention_context(
                token=token,
                kind="resource",
                server_name=server_name,
                partial=remainder,
            )

        return self._argument_mention_context(
            token=token,
            server_name=server_name,
            split_result=split_result,
        )

    @staticmethod
    def _mention_server_meta(server_name: str) -> str:
        if server_name == FILE_MENTION_SERVER:
            return "local file attachment"
        if server_name == URL_MENTION_SERVER:
            return "remote URL attachment"
        return "connected mcp server (resources)"

    def _mention_server_completions(self, context: _MentionContext) -> list[Completion]:
        cache_key = (
            "resource_server",
            self.current_agent,
            context.partial,
        )
        cached = self._completion_cache_get(cache_key)
        if cached is not None:
            return list(cached)

        server_names = self._run_async_completion(self._list_connected_resource_servers) or []
        server_names = list(dict.fromkeys([*server_names, FILE_MENTION_SERVER, URL_MENTION_SERVER]))
        completions = [
            Completion(
                f"{server_name}:",
                start_position=-len(context.partial),
                display=server_name,
                display_meta=self._mention_server_meta(server_name),
            )
            for server_name in server_names
            if not context.partial or starts_with_casefold(server_name, context.partial)
        ]
        self._completion_cache_put(cache_key, completions)
        return completions

    def _mention_url_completions(self, partial: str) -> list[Completion]:
        return [
            Completion(
                scheme,
                start_position=-len(partial),
                display=scheme,
                display_meta="remote URL attachment",
            )
            for scheme in ("https://", "http://")
            if not partial or starts_with_casefold(scheme, partial)
        ]

    def _mention_remote_resource_completions(
        self,
        context: _MentionContext,
    ) -> list[Completion]:
        if context.server_name is None:
            return []

        cache_key = (
            "resource",
            self.current_agent,
            context.server_name,
            context.partial,
        )
        cached = self._completion_cache_get(cache_key)
        if cached is not None:
            return list(cached)

        server_name = context.server_name
        resources = (
            self._run_async_completion(lambda: self._list_server_resource_uris(server_name)) or []
        )
        templates = (
            self._run_async_completion(lambda: self._list_server_resource_templates(server_name))
            or []
        )

        completions = [
            Completion(
                uri,
                start_position=-len(context.partial),
                display=uri,
                display_meta="resource",
            )
            for uri in sorted(set(resources))
            if not context.partial or starts_with_casefold(uri, context.partial)
        ]
        completions.extend(
            Completion(
                f"{template.uriTemplate}{{",
                start_position=-len(context.partial),
                display=template.uriTemplate,
                display_meta="resource template",
            )
            for template in templates
            if not context.partial or starts_with_casefold(template.uriTemplate, context.partial)
        )

        self._completion_cache_put(cache_key, completions)
        return completions

    def _mention_resource_completions(self, context: _MentionContext) -> list[Completion]:
        if context.server_name == FILE_MENTION_SERVER:
            return self._complete_local_attachment_paths(context.partial)
        if context.server_name == URL_MENTION_SERVER:
            return self._mention_url_completions(context.partial)
        return self._mention_remote_resource_completions(context)

    def _mention_argument_name_completions(
        self,
        context: _MentionContext,
    ) -> list[Completion]:
        if not context.template_uri:
            return []
        return [
            Completion(
                f"{argument_name}=",
                start_position=-len(context.partial),
                display=argument_name,
                display_meta="template argument",
            )
            for argument_name in template_argument_names(context.template_uri)
            if argument_name not in context.context_args
            if not context.partial or starts_with_casefold(argument_name, context.partial)
        ]

    def _mention_argument_value_completions(
        self,
        context: _MentionContext,
    ) -> list[Completion]:
        if not context.template_uri or not context.argument_name or not context.server_name:
            return []

        cache_key = (
            "arg_value",
            self.current_agent,
            context.server_name,
            context.template_uri,
            context.argument_name,
            tuple(sorted(context.context_args.items())),
            context.argument_value,
        )
        cached = self._completion_cache_get(cache_key)
        if cached is not None:
            return list(cached)

        server_name = context.server_name
        template_uri = context.template_uri
        argument_name = context.argument_name
        values = (
            self._run_async_completion(
                lambda: self._complete_server_template_argument(
                    server_name=server_name,
                    template_uri=template_uri,
                    argument_name=argument_name,
                    value=context.argument_value,
                    context_args=context.context_args or None,
                )
            )
            or []
        )

        completions = [
            Completion(
                value,
                start_position=-len(context.argument_value),
                display=value,
                display_meta=f"{context.server_name} completion",
            )
            for value in values
        ]
        self._completion_cache_put(cache_key, completions)
        return completions

    def _server_scoped_mention_completions(self, context: _MentionContext) -> list[Completion]:
        if context.server_name is None:
            return []

        handlers: dict[
            str,
            Callable[[AgentCompleter._MentionContext], list[Completion]],
        ] = {
            "resource": self._mention_resource_completions,
            "argument_name": self._mention_argument_name_completions,
            "argument_value": self._mention_argument_value_completions,
        }
        handler = handlers.get(context.kind)
        if handler is None:
            return []
        return handler(context)

    def _mention_completions(self, text_before_cursor: str) -> list[Completion] | None:
        context = self._mention_context_for_text(text_before_cursor)
        if context is None:
            return None

        if context.kind == "server":
            return self._mention_server_completions(context)
        return self._server_scoped_mention_completions(context)

    def _shell_token_completions(
        self,
        shell_text: str,
        *,
        completion_requested: bool,
    ) -> list[Completion]:
        if not shell_text:
            if completion_requested:
                return list(self._complete_executables("", max_results=100))
            return []

        path_part = self._shell_current_token(shell_text)
        if path_part == shell_text:
            if self._is_shell_path_token(path_part):
                return self._quoted_shell_path_completions(path_part, len(path_part))
            return list(self._complete_executables(shell_text))

        return self._quoted_shell_path_completions(path_part, len(path_part))

    def _shell_command_completions(
        self,
        text: str,
        complete_event,
        *,
        completion_requested: bool,
    ) -> list[Completion] | None:
        if text.lstrip().startswith("!"):
            if complete_event and complete_event.text_inserted:
                return []
            # Text after "!" with leading/trailing whitespace stripped
            shell_text = text.lstrip()[1:].lstrip()
            return self._shell_token_completions(
                shell_text,
                completion_requested=completion_requested,
            )

        return None

    def _slash_command_completions(self, text_lower: str) -> list[Completion] | None:
        if not text_lower.startswith("/"):
            return None

        cmd = text_lower[1:]
        return [
            Completion(
                command,
                start_position=-len(cmd),
                display=command,
                display_meta=description,
            )
            for command, description in self.commands.items()
            if starts_with_casefold(command, cmd)
        ]

    def _agent_name_completions(self, text: str) -> list[Completion] | None:
        if not text.startswith("@"):
            return None

        agent_name = text[1:]
        return [
            Completion(
                agent,
                start_position=-len(agent_name),
                display=agent,
                display_meta=self.agent_types.get(agent, AgentType.BASIC).value,
            )
            for agent in self.agents
            if starts_with_casefold(agent, agent_name)
        ]

    def _requested_path_completions(self, text: str) -> list[Completion]:
        partial = text.split()[-1] if text and not text[-1].isspace() else ""
        return list(self._complete_shell_paths(partial, len(partial)))

    def _hash_agent_completions(self, text: str) -> list[Completion] | None:
        if not text.startswith(("##", "#")):
            return None

        prefix = "##" if text.startswith("##") else "#"
        rest = text[len(prefix) :]
        if rest and rest[0].isspace():
            return []
        if " " in rest or "\t" in rest:
            return []

        agent_name = rest
        return [
            Completion(
                agent + " ",
                start_position=-len(agent_name),
                display=agent,
                display_meta=f"{prefix} {self.agent_types.get(agent, AgentType.BASIC).value}",
            )
            for agent in self.agents
            if starts_with_casefold(agent, agent_name)
        ]

    def get_completions(self, document, complete_event):
        """Synchronous completions method - this is what prompt_toolkit expects by default"""
        text = document.text_before_cursor
        text_lower = casefold_text(text)
        completion_requested = bool(complete_event and complete_event.completion_requested)

        shell_completions = self._shell_command_completions(
            text,
            complete_event,
            completion_requested=completion_requested,
        )
        if shell_completions is not None:
            yield from shell_completions
            return

        mention_completions = self._mention_completions(text)
        if mention_completions is not None:
            yield from mention_completions
            return

        plugin_completions = self._plugin_command_argument_completions(text)
        if plugin_completions is not None:
            yield from plugin_completions
            return

        from fast_agent.ui.prompt.completion_sources import command_completions

        source_completions = command_completions(self, text, text_lower)
        if source_completions is not None:
            yield from source_completions
            return

        slash_completions = self._slash_command_completions(text_lower)
        if slash_completions is not None:
            yield from slash_completions
        else:
            agent_completions = self._agent_name_completions(text)
            if agent_completions is not None:
                yield from agent_completions

        if completion_requested:
            yield from self._requested_path_completions(text)
            return

        hash_completions = self._hash_agent_completions(text)
        if hash_completions is not None:
            yield from hash_completions
