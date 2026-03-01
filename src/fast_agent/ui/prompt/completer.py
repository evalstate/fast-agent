"""Interactive prompt command completer."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.types import ResourceTemplate
from prompt_toolkit.completion import Completer, Completion

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.config import get_settings
from fast_agent.llm.reasoning_effort import available_reasoning_values
from fast_agent.llm.text_verbosity import available_text_verbosity_values
from fast_agent.ui.prompt.resource_mentions import template_argument_names

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.types import PromptMessageExtended

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
    ) -> None:
        self.agents = agents
        self.current_agent = current_agent
        self.agent_provider = agent_provider
        self.noenv_mode = noenv_mode
        # Map commands to their descriptions for better completion hints
        self.commands = {
            "mcp": "Manage MCP runtime servers (/mcp list|connect|disconnect|reconnect|session)",
            "connect": "Alias for /mcp connect with target auto-detection",
            "history": "Show conversation history overview (or /history save|load|clear|rewind|review|fix)",
            "tools": "List tools",
            "model": (
                "Update model settings "
                "(/model reasoning|verbosity|web_search|web_fetch <value>)"
            ),
            "models": (
                "Inspect model onboarding "
                "(/models, /models doctor, /models aliases [list|set|unset], /models catalog <provider>)"
            ),
            "skills": (
                "Manage skills "
                "(/skills, /skills available, /skills search, /skills add, "
                "/skills remove, /skills update, /skills registry, /skills help)"
            ),
            "cards": (
                "Manage card packs "
                "(/cards, /cards add, /cards remove, /cards update, /cards publish, /cards registry)"
            ),
            "prompt": "Load a Prompt File or use MCP Prompt",
            "system": "Show the current system prompt",
            "usage": "Show current usage statistics",
            "markdown": "Show last assistant message without markdown formatting",
            "resume": "Resume the last session or specified session id",
            "session": "Manage sessions (/session list|new|resume|title|fork|delete|pin)",
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
        self._mention_cache: dict[tuple[Any, ...], AgentCompleter._CacheEntry] = {}
        self._mention_cache_ttl_seconds = 3.0
        self._completion_wait_timeout_seconds = 1.5
        try:
            self._owner_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._owner_loop = None

    def _current_agent_has_web_tools_enabled(self) -> bool:
        if self.agent_provider is None or not self.current_agent:
            return False
        try:
            agent_obj = self.agent_provider._agent(self.current_agent)
        except Exception:
            return False
        return history_handlers.web_tools_enabled_for_agent(agent_obj)

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
            if (
                partial.endswith("/")
                or partial.endswith(os.sep)
                or (os.altsep is not None and partial.endswith(os.altsep))
            ):
                raw_dir = partial
                prefix = ""
            else:
                raw_dir, prefix = os.path.split(partial)
                explicit_current_dir = partial.startswith(f".{os.sep}") or (
                    os.altsep is not None and partial.startswith(f".{os.altsep}")
                )

        raw_dir = raw_dir or "."
        expanded_dir = Path(os.path.expandvars(os.path.expanduser(raw_dir)))
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
                if not name.lower().startswith(prefix.lower()):
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

    def _current_agent_llm(self) -> object | None:
        if not self.agent_provider or not self.current_agent:
            return None
        try:
            agent_obj = self.agent_provider._agent(self.current_agent)
        except Exception:
            return None
        llm = getattr(agent_obj, "llm", None) or getattr(agent_obj, "_llm", None)
        return llm

    def _resolve_reasoning_values(self) -> list[str]:
        llm = self._current_agent_llm()
        if not llm:
            return []
        return available_reasoning_values(getattr(llm, "reasoning_effort_spec", None))

    def _resolve_verbosity_values(self) -> list[str]:
        llm = self._current_agent_llm()
        if not llm:
            return []
        return available_text_verbosity_values(getattr(llm, "text_verbosity_spec", None))

    def _supports_web_search_setting(self) -> bool:
        llm = self._current_agent_llm()
        if llm is None:
            return False
        return model_handlers.model_supports_web_search(llm)

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

    def _complete_session_ids(self, partial: str, *, start_position: int | None = None):
        """Generate completions for recent session ids."""
        if self.noenv_mode:
            return

        from fast_agent.session import (
            apply_session_window,
            display_session_name,
            get_session_manager,
        )

        manager = get_session_manager()
        sessions = apply_session_window(manager.list_sessions())
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
                start_position=-len(partial) if start_position is None else start_position,
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

    def _complete_local_skill_names(
        self,
        partial: str,
        *,
        managed_only: bool = False,
        include_indices: bool = True,
    ):
        """Generate completions for local skill names and indices."""
        from fast_agent.skills.manager import get_manager_directory, read_installed_skill_source
        from fast_agent.skills.registry import SkillRegistry

        manager_dir = get_manager_directory()
        manifests = SkillRegistry.load_directory(manager_dir)
        if not manifests:
            return

        partial_lower = partial.lower()
        include_numbers = include_indices and (not partial or partial.isdigit())
        for index, manifest in enumerate(manifests, 1):
            if managed_only:
                source, _ = read_installed_skill_source(Path(manifest.path).parent)
                if source is None:
                    continue

            name = manifest.name
            if name and (not partial or name.lower().startswith(partial_lower)):
                yield Completion(
                    name,
                    start_position=-len(partial),
                    display=name,
                    display_meta="managed skill" if managed_only else "local skill",
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
        from fast_agent.skills.manager import (
            format_marketplace_display_url,
            resolve_skill_registries,
        )

        configured_urls = resolve_skill_registries(get_settings())
        yield from self._complete_registry_urls(
            partial,
            configured_urls=configured_urls,
            display_formatter=format_marketplace_display_url,
        )

    def _complete_registry_urls(
        self,
        partial: str,
        *,
        configured_urls: "Sequence[str]",
        display_formatter: "Callable[[str], str]",
    ):
        """Generate index/url completions for a registry URL list."""
        partial_lower = partial.lower()
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
            if include_urls and url.lower().startswith(partial_lower):
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

        partial_lower = partial.lower()
        include_numbers = include_indices and (not partial or partial.isdigit())
        for entry in packs:
            if managed_only and entry.source is None:
                continue

            name = entry.name
            if name and (not partial or name.lower().startswith(partial_lower)):
                yield Completion(
                    name,
                    start_position=-len(partial),
                    display=name,
                    display_meta="managed card pack" if entry.source else "local card pack",
                )

            if include_numbers:
                index_text = str(entry.index)
                if not partial or index_text.startswith(partial):
                    yield Completion(
                        index_text,
                        start_position=-len(partial),
                        display=index_text,
                        display_meta=name,
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
        if os.altsep and os.altsep in token:
            return True
        return False

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
                if not name.lower().startswith(prefix.lower()):
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

    def _complete_subcommands(
        self,
        parts: Sequence[str],
        remainder: str,
        subcommands: dict[str, str],
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
                if subcmd.startswith(partial.lower()):
                    yield Completion(
                        subcmd,
                        start_position=-len(partial),
                        display=subcmd,
                        display_meta=description,
                    )

    @staticmethod
    def _configured_mcp_server_target(server_config: Any) -> str | None:
        url_value: Any
        if isinstance(server_config, dict):
            url_value = server_config.get("url")
        else:
            url_value = getattr(server_config, "url", None)

        if isinstance(url_value, str):
            normalized = url_value.strip()
            if normalized:
                return normalized

        command_value: Any
        args_value: Any
        if isinstance(server_config, dict):
            command_value = server_config.get("command")
            args_value = server_config.get("args")
        else:
            command_value = getattr(server_config, "command", None)
            args_value = getattr(server_config, "args", None)

        if isinstance(command_value, str):
            command = command_value.strip()
            if command:
                args: list[str] = []
                if isinstance(args_value, list):
                    args = [str(value) for value in args_value]
                return shlex.join([command, *args])

        return None

    @staticmethod
    def _format_mcp_server_meta(target: str | None) -> str:
        if not target:
            return ""
        if len(target) > 80:
            return f"{target[:79]}…"
        return target

    def _list_configured_mcp_servers(self) -> list[tuple[str, str | None]]:
        configured: set[str] = set()
        attached: set[str] = set()
        server_targets: dict[str, str] = {}

        # Prefer the runtime aggregator when available so completions include
        # both config-backed entries and runtime-attached server names.
        if self.agent_provider is not None and self.current_agent:
            try:
                agent = self.agent_provider._agent(self.current_agent)
                aggregator = getattr(agent, "aggregator", None)
                list_attached = getattr(aggregator, "list_attached_servers", None)
                if callable(list_attached):
                    attached.update(list_attached())

                list_detached = getattr(aggregator, "list_configured_detached_servers", None)
                if callable(list_detached):
                    configured.update(list_detached())

                context = getattr(aggregator, "context", None)
                server_registry = getattr(context, "server_registry", None)
                registry_data = getattr(server_registry, "registry", None)
                if isinstance(registry_data, dict):
                    for name, server_config in registry_data.items():
                        server_name = str(name)
                        configured.add(server_name)
                        target = self._configured_mcp_server_target(server_config)
                        if target is not None:
                            server_targets[server_name] = target
            except Exception:
                pass

        # Fall back to global settings so completion still works before agent
        # startup wiring is fully available.
        try:
            settings = get_settings()
            mcp_settings = getattr(settings, "mcp", None)
            server_map = getattr(mcp_settings, "servers", None)
            if isinstance(server_map, dict):
                for name, server_config in server_map.items():
                    server_name = str(name)
                    configured.add(server_name)
                    target = self._configured_mcp_server_target(server_config)
                    if target is not None:
                        server_targets[server_name] = target
        except Exception:
            pass

        if attached:
            configured.difference_update(attached)

        return [(server_name, server_targets.get(server_name)) for server_name in sorted(configured)]

    def _complete_configured_mcp_servers(self, partial: str):
        partial_lower = partial.lower()
        for server_name, server_url in self._list_configured_mcp_servers():
            if partial and not server_name.lower().startswith(partial_lower):
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
            display="[url|npx|uvx]",
            display_meta="enter url or npx/uvx cmd",
        )

    @staticmethod
    def _mcp_connect_context(remainder: str) -> tuple[str, int, str]:
        """Classify completion context for `/mcp connect ...`.

        Returns (context, target_count, partial) where:
          - context: one of "target", "flag", "flag_value", "new_token"
          - target_count: number of fully-formed target tokens before `partial`
          - partial: token currently being edited (or "" when cursor is after whitespace)
        """

        takes_value = {"--name", "-n", "--timeout", "--auth"}
        switch_only = {"--oauth", "--no-oauth", "--reconnect", "--no-reconnect"}

        raw_tokens = remainder.split()
        trailing_space = remainder.endswith(" ")
        partial = "" if trailing_space else (raw_tokens[-1] if raw_tokens else "")
        complete_tokens = raw_tokens if trailing_space else raw_tokens[:-1]

        target_count = 0
        waiting_for_flag_value = False
        for token in complete_tokens:
            if waiting_for_flag_value:
                waiting_for_flag_value = False
                continue
            if token in takes_value:
                waiting_for_flag_value = True
                continue
            if token.startswith("--auth="):
                continue
            if token in switch_only:
                continue
            target_count += 1

        if trailing_space:
            return (
                "flag_value" if waiting_for_flag_value else "new_token",
                target_count,
                partial,
            )

        if waiting_for_flag_value:
            return "flag_value", target_count, partial
        if partial in takes_value or partial in switch_only or partial.startswith("--"):
            return "flag", target_count, partial
        return "target", target_count, partial

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
        except Exception:  # noqa: BLE001 - defensive fallback for capability objects
            return True

    async def _list_connected_resource_servers(self) -> list[str]:
        agent = self._current_agent_object()
        if agent is None:
            return []

        aggregator = getattr(agent, "aggregator", None)
        if aggregator is None:
            return []

        collect_status = getattr(aggregator, "collect_server_status", None)
        if callable(collect_status):
            try:
                status_map = await collect_status()
            except Exception:
                status_map = None
            if isinstance(status_map, dict):
                names: set[str] = set()
                for server_name, status in status_map.items():
                    if getattr(status, "is_connected", None) is not True:
                        continue

                    capabilities = getattr(status, "server_capabilities", None)
                    resources_capability = (
                        getattr(capabilities, "resources", None) if capabilities is not None else None
                    )
                    if not self._feature_enabled(resources_capability):
                        continue

                    names.add(str(server_name))
                return sorted(names)

        list_attached = getattr(aggregator, "list_attached_servers", None)
        get_capabilities = getattr(aggregator, "get_capabilities", None)
        if not callable(list_attached) or not callable(get_capabilities):
            return []

        names: set[str] = set()
        try:
            attached_servers = list_attached()
        except Exception:
            attached_servers = []

        for server_name in attached_servers:
            try:
                capabilities = await get_capabilities(server_name)
            except Exception:
                continue
            resources_capability = (
                getattr(capabilities, "resources", None) if capabilities is not None else None
            )
            if self._feature_enabled(resources_capability):
                names.add(str(server_name))

        return sorted(names)

    async def _list_server_session_cookie_choices(
        self,
        server_identifier: str | None,
    ) -> list[tuple[str, str | None, bool]]:
        agent = self._current_agent_object()
        if agent is None:
            return []

        aggregator = getattr(agent, "aggregator", None)
        if aggregator is None:
            return []

        session_client = getattr(aggregator, "experimental_sessions", None)
        list_server_cookies = getattr(session_client, "list_server_cookies", None)
        if not callable(list_server_cookies):
            return []

        try:
            _server_name, _identity, active_id, cookies = await list_server_cookies(server_identifier)
        except Exception:
            return []

        choices: list[tuple[str, str | None, bool]] = []
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            cookie_id = cookie.get("id")
            if not isinstance(cookie_id, str) or not cookie_id:
                continue
            title = cookie.get("title")
            title_text = title if isinstance(title, str) and title.strip() else None
            is_active = cookie_id == active_id
            choices.append((cookie_id, title_text, is_active))

        return choices

    async def _list_connected_mcp_servers_with_identity(self) -> list[tuple[str, str | None]]:
        agent = self._current_agent_object()
        if agent is None:
            return []

        aggregator = getattr(agent, "aggregator", None)
        if aggregator is None:
            return []

        collect_status = getattr(aggregator, "collect_server_status", None)
        if callable(collect_status):
            try:
                status_map = await collect_status()
            except Exception:
                status_map = None
            if isinstance(status_map, dict):
                connected: list[tuple[str, str | None]] = []
                for server_name, status in sorted(status_map.items()):
                    if getattr(status, "is_connected", None) is not True:
                        continue
                    identity = getattr(status, "implementation_name", None)
                    identity_name = identity if isinstance(identity, str) and identity.strip() else None
                    connected.append((str(server_name), identity_name))
                if connected:
                    return connected

        list_attached = getattr(aggregator, "list_attached_servers", None)
        if not callable(list_attached):
            return []
        try:
            attached = list_attached()
        except Exception:
            return []

        return [(str(server_name), None) for server_name in attached]

    async def _list_attached_session_cookie_choices(
        self,
    ) -> list[tuple[str, str | None, str, str | None, bool]]:
        agent = self._current_agent_object()
        if agent is None:
            return []

        aggregator = getattr(agent, "aggregator", None)
        session_client = getattr(aggregator, "experimental_sessions", None) if aggregator else None
        list_server_cookies = getattr(session_client, "list_server_cookies", None)
        if not callable(list_server_cookies):
            return []

        choices: list[tuple[str, str | None, str, str | None, bool]] = []
        servers = await self._list_connected_mcp_servers_with_identity()
        for server_name, fallback_identity in servers:
            try:
                _resolved_name, identity, active_id, cookies = await list_server_cookies(server_name)
            except Exception:
                continue

            display_identity = (
                identity if isinstance(identity, str) and identity.strip() else fallback_identity
            )
            for cookie in cookies:
                if not isinstance(cookie, dict):
                    continue
                cookie_id = cookie.get("id")
                if not isinstance(cookie_id, str) or not cookie_id:
                    continue
                title = cookie.get("title")
                title_text = title if isinstance(title, str) and title.strip() else None
                choices.append(
                    (
                        server_name,
                        display_identity,
                        cookie_id,
                        title_text,
                        cookie_id == active_id,
                    )
                )

        return choices

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

    def _run_async_completion(self, awaitable) -> Any:
        owner_loop = self._owner_loop
        if owner_loop is not None and owner_loop.is_running():
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None

            if current_loop is not owner_loop:
                future = asyncio.run_coroutine_threadsafe(awaitable, owner_loop)
                try:
                    return future.result(timeout=self._completion_wait_timeout_seconds)
                except FuturesTimeoutError:
                    future.cancel()
                    return None
                except Exception:
                    return None

        try:
            return asyncio.run(awaitable)
        except Exception:
            return None

    async def _list_server_resource_uris(self, server_name: str) -> list[str]:
        agent = self._current_agent_object()
        if agent is None:
            return []

        list_resources = getattr(agent, "list_resources", None)
        if not callable(list_resources):
            return []

        result = await list_resources(namespace=server_name)
        uris = result.get(server_name, []) if isinstance(result, dict) else []
        return [str(uri) for uri in uris]

    async def _list_server_resource_templates(self, server_name: str) -> list[ResourceTemplate]:
        agent = self._current_agent_object()
        if agent is None:
            return []

        aggregator = getattr(agent, "aggregator", None)
        list_templates = getattr(aggregator, "list_resource_templates", None)
        if not callable(list_templates):
            return []

        result = await list_templates(server_name)
        templates = result.get(server_name, []) if isinstance(result, dict) else []
        return [template for template in templates if isinstance(template, ResourceTemplate)]

    async def _complete_server_template_argument(
        self,
        server_name: str,
        template_uri: str,
        argument_name: str,
        value: str,
        context_args: dict[str, str] | None = None,
    ) -> list[str]:
        agent = self._current_agent_object()
        if agent is None:
            return []

        aggregator = getattr(agent, "aggregator", None)
        complete_arg = getattr(aggregator, "complete_resource_argument", None)
        if not callable(complete_arg):
            return []

        completion = await complete_arg(
            server_name=server_name,
            template_uri=template_uri,
            argument_name=argument_name,
            value=value,
            context_args=context_args,
        )
        values = getattr(completion, "values", None)
        if not isinstance(values, list):
            return []
        return [str(item) for item in values]

    @staticmethod
    def _extract_template_argument_names(template_uri: str) -> list[str]:
        return template_argument_names(template_uri)

    @staticmethod
    def _split_mention_argument_section(remainder: str) -> tuple[str, str] | None:
        """Split a mention remainder into (template_uri, argument_text).

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
            if char == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        open_index = None

        if depth == 0 or open_index is None:
            return None
        if depth != 1:
            return None

        return remainder[:open_index], remainder[open_index + 1 :]

    def _mention_context_for_text(self, text: str) -> _MentionContext | None:
        match = self._MENTION_RE.search(text)
        if not match:
            return None

        token = match.group(1)
        payload = token[1:]

        if ":" not in payload:
            return self._MentionContext(
                token=token,
                kind="server",
                server_name=None,
                partial=payload,
                template_uri=None,
                argument_name=None,
                argument_value="",
                context_args={},
            )

        server_name, remainder = payload.split(":", 1)
        if not server_name:
            return None

        split_result = self._split_mention_argument_section(remainder)
        if split_result is None:
            return self._MentionContext(
                token=token,
                kind="resource",
                server_name=server_name,
                partial=remainder,
                template_uri=None,
                argument_name=None,
                argument_value="",
                context_args={},
            )

        template_uri, argument_text = split_result
        if "}" in argument_text:
            return None

        argument_text = argument_text.strip()
        context_args: dict[str, str] = {}
        raw_segments = [segment.strip() for segment in argument_text.split(",")]
        if not raw_segments:
            raw_segments = [""]

        for segment in raw_segments[:-1]:
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            key = key.strip()
            if not key:
                continue
            context_args[key] = value.strip()

        current_segment = raw_segments[-1]
        if not current_segment:
            return self._MentionContext(
                token=token,
                kind="argument_name",
                server_name=server_name,
                partial="",
                template_uri=template_uri,
                argument_name=None,
                argument_value="",
                context_args=context_args,
            )

        if "=" not in current_segment:
            return self._MentionContext(
                token=token,
                kind="argument_name",
                server_name=server_name,
                partial=current_segment,
                template_uri=template_uri,
                argument_name=None,
                argument_value="",
                context_args=context_args,
            )

        argument_name, value_partial = current_segment.split("=", 1)
        argument_name = argument_name.strip()
        return self._MentionContext(
            token=token,
            kind="argument_value",
            server_name=server_name,
            partial=value_partial,
            template_uri=template_uri,
            argument_name=argument_name,
            argument_value=value_partial,
            context_args=context_args,
        )

    def _mention_completions(self, text_before_cursor: str) -> list[Completion] | None:
        context = self._mention_context_for_text(text_before_cursor)
        if context is None:
            return None

        if context.kind == "server":
            cache_key = (
                "resource_server",
                self.current_agent,
                context.partial,
            )
            cached = self._completion_cache_get(cache_key)
            if cached is not None:
                return list(cached)

            server_names = self._run_async_completion(self._list_connected_resource_servers()) or []
            partial = context.partial.lower()
            completions = [
                Completion(
                    f"{server_name}:",
                    start_position=-len(context.partial),
                    display=server_name,
                    display_meta="connected mcp server (resources)",
                )
                for server_name in server_names
                if not partial or server_name.lower().startswith(partial)
            ]
            self._completion_cache_put(cache_key, completions)
            return completions

        if context.server_name is None:
            return []

        if context.kind == "resource":
            cache_key = (
                "resource",
                self.current_agent,
                context.server_name,
                context.partial,
            )
            cached = self._completion_cache_get(cache_key)
            if cached is not None:
                return list(cached)

            resources = self._run_async_completion(
                self._list_server_resource_uris(context.server_name)
            ) or []
            templates = self._run_async_completion(
                self._list_server_resource_templates(context.server_name)
            ) or []

            prefix = context.partial.lower()
            completions: list[Completion] = []
            for uri in sorted(set(resources)):
                if prefix and not uri.lower().startswith(prefix):
                    continue
                completions.append(
                    Completion(
                        uri,
                        start_position=-len(context.partial),
                        display=uri,
                        display_meta="resource",
                    )
                )

            for template in templates:
                template_uri = template.uriTemplate
                if prefix and not template_uri.lower().startswith(prefix):
                    continue
                completions.append(
                    Completion(
                        f"{template_uri}{{",
                        start_position=-len(context.partial),
                        display=template_uri,
                        display_meta="resource template",
                    )
                )

            self._completion_cache_put(cache_key, completions)
            return completions

        if context.kind == "argument_name":
            if not context.template_uri:
                return []
            argument_names = self._extract_template_argument_names(context.template_uri)
            prefix = context.partial.lower()
            return [
                Completion(
                    f"{argument_name}=",
                    start_position=-len(context.partial),
                    display=argument_name,
                    display_meta="template argument",
                )
                for argument_name in argument_names
                if argument_name not in context.context_args
                if not prefix or argument_name.lower().startswith(prefix)
            ]

        if context.kind == "argument_value":
            if (
                not context.template_uri
                or not context.argument_name
                or not context.server_name
            ):
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

            values = self._run_async_completion(
                self._complete_server_template_argument(
                    server_name=context.server_name,
                    template_uri=context.template_uri,
                    argument_name=context.argument_name,
                    value=context.argument_value,
                    context_args=context.context_args or None,
                )
            ) or []

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

        return []

    def get_completions(self, document, complete_event):
        """Synchronous completions method - this is what prompt_toolkit expects by default"""
        text = document.text_before_cursor
        text_lower = text.lower()
        completion_requested = bool(complete_event and complete_event.completion_requested)

        # Shell completion mode - detect ! prefix
        if text.lstrip().startswith("!"):
            if complete_event and complete_event.text_inserted:
                return
            # Text after "!" with leading/trailing whitespace stripped
            shell_text = text.lstrip()[1:].lstrip()
            if not shell_text:
                if completion_requested:
                    yield from self._complete_executables("", max_results=100)
                return

            if " " not in shell_text:
                # First token: complete executables or paths.
                if self._is_shell_path_token(shell_text):
                    yield from self._complete_shell_paths(shell_text, len(shell_text))
                else:
                    yield from self._complete_executables(shell_text)
            else:
                # After first token: complete paths
                _, path_part = shell_text.rsplit(" ", 1)
                yield from self._complete_shell_paths(path_part, len(path_part))
            return

        mention_completions = self._mention_completions(text)
        if mention_completions is not None:
            yield from mention_completions
            return

        from fast_agent.ui.prompt.completion_sources import command_completions

        source_completions = command_completions(self, text, text_lower)
        if source_completions is not None:
            yield from source_completions
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
