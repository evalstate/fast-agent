"""Interactive prompt command completer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.config import get_settings
from fast_agent.llm.reasoning_effort import available_reasoning_values
from fast_agent.llm.text_verbosity import available_text_verbosity_values

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.types import PromptMessageExtended

class AgentCompleter(Completer):
    """Provide completion for agent names and common commands."""

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
            "mcp": "Manage MCP runtime servers (/mcp list|connect|disconnect)",
            "connect": "Alias for /mcp connect with target auto-detection",
            "history": "Show conversation history overview (or /history save|load|clear|rewind|review|fix)",
            "tools": "List tools",
            "model": (
                "Update model settings "
                "(/model reasoning|verbosity|web_search|web_fetch <value>)"
            ),
            "skills": "Manage skills (/skills, /skills add, /skills remove, /skills update, /skills registry)",
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

    def _list_configured_mcp_servers(self) -> list[str]:
        configured: set[str] = set()
        attached: set[str] = set()

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
                    configured.update(str(name) for name in registry_data)
            except Exception:
                pass

        # Fall back to global settings so completion still works before agent
        # startup wiring is fully available.
        try:
            settings = get_settings()
            mcp_settings = getattr(settings, "mcp", None)
            server_map = getattr(mcp_settings, "servers", None)
            if isinstance(server_map, dict):
                configured.update(str(name) for name in server_map)
        except Exception:
            pass

        if attached:
            configured.difference_update(attached)

        return sorted(configured)

    def _complete_configured_mcp_servers(self, partial: str):
        partial_lower = partial.lower()
        for server_name in self._list_configured_mcp_servers():
            if partial and not server_name.lower().startswith(partial_lower):
                continue
            yield Completion(
                server_name,
                start_position=-len(partial),
                display=server_name,
                display_meta="configured mcp server",
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

