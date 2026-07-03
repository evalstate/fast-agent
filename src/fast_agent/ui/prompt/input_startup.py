"""Startup rendering helpers for prompt input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rich import print as rich_print
from rich.markup import escape as escape_markup
from rich.text import Text

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.protocols import HfDisplayInfoProvider
from fast_agent.ui.message_primitives import MessageType
from fast_agent.ui.model_shortcuts import build_model_shortcut_hints
from fast_agent.ui.prompt.input_toolbar import resolve_active_llm
from fast_agent.ui.shell_notice import format_shell_notice
from fast_agent.ui.streaming_preferences import resolve_streaming_preferences
from fast_agent.utils.count_display import format_count
from fast_agent.utils.path_display import format_home_relative_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from fast_agent.core.agent_app import AgentApp


@runtime_checkable
class ContextBackedStatusAgent(Protocol):
    @property
    def context(self) -> object | None: ...


@runtime_checkable
class LlmBackedStatusAgent(Protocol):
    @property
    def llm(self) -> object | None: ...


@runtime_checkable
class ConfigBackedContext(Protocol):
    @property
    def config(self) -> object | None: ...


@runtime_checkable
class ConfigBackedAgent(Protocol):
    @property
    def config(self) -> object | None: ...


@runtime_checkable
class LoggerBackedConfig(Protocol):
    @property
    def logger(self) -> object | None: ...


@runtime_checkable
class ModelSourceConfig(Protocol):
    @property
    def model_source(self) -> object | None: ...


@runtime_checkable
class FastAgentHomeConfig(Protocol):
    @property
    def _fast_agent_home(self) -> object | None: ...

    @property
    def _fast_agent_home_source(self) -> object | None: ...

    @property
    def model_references(self) -> object | None: ...


@runtime_checkable
class ToolHookConfig(Protocol):
    @property
    def tool_hooks(self) -> object | None: ...


@runtime_checkable
class LifecycleHookConfig(Protocol):
    @property
    def lifecycle_hooks(self) -> object | None: ...


@runtime_checkable
class CommandBackedConfig(Protocol):
    @property
    def commands(self) -> object | None: ...


@runtime_checkable
class PluginCommandProvider(Protocol):
    @property
    def plugin_commands(self) -> object | None: ...


@runtime_checkable
class ChatVisibilitySettings(Protocol):
    @property
    def show_chat(self) -> bool: ...


@runtime_checkable
class A2APromptStatusAgent(Protocol):
    def prompt_status_line(self) -> str | None: ...


class ShellInputContextLike(Protocol):
    enabled: bool
    access_modes: tuple[str, ...]
    runtime: Any | None


@dataclass(slots=True)
class StartupNotice:
    text: str
    render_markdown: bool = False
    title: str | None = None
    right_info: str | None = None
    agent_name: str | None = None


def queue_startup_notice(notices: list[object], notice: object) -> None:
    if notice:
        notices.append(notice)


def queue_startup_markdown_notice(
    notices: list[object],
    text: str,
    *,
    title: str | None = None,
    style: str | None = None,
    right_info: str | None = None,
    agent_name: str | None = None,
) -> None:
    """Queue a markdown notice for display at next interactive prompt render."""
    if not text:
        return

    if style is not None and right_info is None and agent_name is None:
        from rich.markdown import Markdown

        if title:
            queue_startup_notice(notices, title)
        queue_startup_notice(notices, Markdown(text, style=style or ""))
        return

    notices.append(
        StartupNotice(
            text=text,
            render_markdown=True,
            title=title,
            right_info=right_info,
            agent_name=agent_name,
        )
    )


def count_configured_hooks(agent_provider: "AgentApp") -> int:
    total = 0
    for agent in agent_provider.registered_agents().values():
        config = agent_config(agent)
        if isinstance(config, ToolHookConfig):
            total += dict_size(config.tool_hooks)
        if isinstance(config, LifecycleHookConfig):
            total += dict_size(config.lifecycle_hooks)
    return total


def count_configured_extensions(agent_provider: "AgentApp") -> int:
    total = (
        dict_size(agent_provider.plugin_commands)
        if isinstance(agent_provider, PluginCommandProvider)
        else 0
    )

    for agent in agent_provider.registered_agents().values():
        config = agent_config(agent)
        if isinstance(config, CommandBackedConfig):
            total += dict_size(config.commands)

    return total


def agent_config(agent: object) -> object | None:
    if isinstance(agent, ConfigBackedAgent):
        return agent.config
    if not isinstance(agent, ContextBackedStatusAgent):
        return None
    context = agent.context
    if not isinstance(context, ConfigBackedContext):
        return None
    return context.config


def dict_size(value: object | None) -> int:
    return len(value) if isinstance(value, dict) else 0


def show_fast_agent_home_summary(agent_provider: "AgentApp | None") -> None:
    if agent_provider is None:
        return
    try:
        first_agent = next(iter(agent_provider.registered_agents().values()))
    except StopIteration:
        return

    if not isinstance(first_agent, ContextBackedStatusAgent):
        return
    context = first_agent.context
    if not isinstance(context, ConfigBackedContext):
        return
    config = context.config
    if not isinstance(config, FastAgentHomeConfig):
        return

    home = config._fast_agent_home
    if not home:
        return

    model_refs = config.model_references
    model_ref_count = (
        sum(len(namespace_refs) for namespace_refs in model_refs.values())
        if isinstance(model_refs, dict)
        else 0
    )
    parts = [
        format_count(len(agent_provider.registered_agent_names()), "agent"),
        format_count(count_configured_hooks(agent_provider), "hook"),
        format_count(count_configured_extensions(agent_provider), "extension"),
        format_count(model_ref_count, "modelref"),
    ]
    source = config._fast_agent_home_source
    source_suffix = f" [dim]via {escape_markup(str(source))}[/dim]" if source else ""
    rich_print(
        f"[dim]fast-agent home[/dim] [blue]{format_home_relative_path(str(home))}[/blue]"
        f"[dim] ({', '.join(parts)}){source_suffix}[/dim]"
    )


def show_stop_hint_message(*, default: str, show_stop_hint: bool) -> None:
    if not show_stop_hint:
        return
    if default == "STOP":
        rich_print("Enter a prompt, [red]STOP[/red] or [red]Ctrl+D[/red] to finish")
        if default:
            rich_print(f"Press <ENTER> to use the default prompt:\n[cyan]{default}[/cyan]")


def show_input_help_banner(
    *,
    is_human_input: bool,
    supports_clipboard_image_paste: bool,
) -> None:
    if is_human_input:
        rich_print("[dim]Type /help for commands. Ctrl+T toggles multiline mode.[/dim]")
        return

    attachment_hint = (
        "Use /attach, `^file:`, `^url:`, or [yellow]Ctrl+Alt+V[/yellow] "
        "for attachments [dim](experimental)[/dim]."
        if supports_clipboard_image_paste
        else "Use /attach, `^file:`, or `^url:` for attachments."
    )
    rich_print(
        """[dim]Use '/' for commands, '!' for shell. '#' to query, '@' to switch agents\n"""
        """CTRL+T multiline, CTRL+Y copy last message, CTRL+E external editor.\n"""
        f"""CTRL+Space or Tab for path completion. {attachment_hint} F10 to clear.[/dim]"""
    )


def show_model_shortcut_hints(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> None:
    startup_llm = resolve_active_llm(agent_provider, agent_name)
    shortcut_hints = build_model_shortcut_hints(startup_llm)
    if not shortcut_hints:
        return
    rich_print("[dim]Model shortcuts:[/dim]")
    for hint in shortcut_hints:
        rich_print(f"[dim]  {hint.key} = {hint.label} ({hint.values_text})[/dim]")
    rich_print()


async def show_shell_startup(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
    shell_context: ShellInputContextLike,
    shell_agent: object | None,
    is_human_input: bool,
    available_agents: Iterable[str],
    display_all_agents_with_hierarchy: Callable[
        [Iterable[str], "AgentApp | None"], Awaitable[None]
    ],
) -> None:
    if not shell_context.enabled:
        return

    rich_print(format_shell_notice(shell_context.access_modes, shell_context.runtime))
    if agent_provider and not is_human_input:
        await display_all_agents_with_hierarchy(available_agents, agent_provider)
    await show_streaming_status(
        agent_name=agent_name,
        agent_provider=agent_provider,
        shell_agent=shell_agent,
    )


async def show_streaming_status(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
    shell_agent: object | None,
) -> None:
    if agent_provider is None:
        return

    active_agent = active_status_agent(
        agent_name=agent_name,
        agent_provider=agent_provider,
        shell_agent=shell_agent,
    )
    if active_agent is None:
        return

    config = status_agent_config(active_agent)
    logger_settings = logger_settings_for_config(config)
    if not show_chat_enabled(logger_settings):
        return

    show_streaming_mode_notice(agent_provider, logger_settings)
    model_source = model_source_for_config(config)
    if model_source:
        rich_print(f"[dim]Model selected via {escape_markup(str(model_source))}[/dim]")

    show_hf_display_info(active_agent)


def active_status_agent(
    *,
    agent_name: str,
    agent_provider: "AgentApp",
    shell_agent: object | None,
) -> object | None:
    if shell_agent is not None:
        return shell_agent
    try:
        return agent_provider._agent(agent_name)
    except Exception:
        return None


def status_agent_config(active_agent: object) -> object | None:
    if not isinstance(active_agent, ContextBackedStatusAgent):
        return None
    context = active_agent.context
    if not isinstance(context, ConfigBackedContext):
        return None
    return context.config


def logger_settings_for_config(config: object | None) -> object | None:
    if not isinstance(config, LoggerBackedConfig):
        return None
    return config.logger


def show_chat_enabled(logger_settings: object | None) -> bool:
    if logger_settings is None:
        return False
    if not isinstance(logger_settings, ChatVisibilitySettings):
        return True
    return logger_settings.show_chat


def model_source_for_config(config: object | None) -> object | None:
    if not isinstance(config, ModelSourceConfig):
        return None
    return config.model_source


def show_hf_display_info(active_agent: object) -> None:
    if not isinstance(active_agent, LlmBackedStatusAgent):
        return
    llm = active_agent.llm
    if not isinstance(llm, HfDisplayInfoProvider):
        return
    display_info = llm.get_hf_display_info
    if not callable(display_info):
        return
    hf_info = display_info()
    model = hf_info.get("model", "unknown")
    provider = hf_info.get("provider", "auto-routing")
    rich_print(
        f"[dim]HuggingFace: {escape_markup(str(model))} via {escape_markup(str(provider))}[/dim]"
    )


def show_streaming_mode_notice(agent_provider: "AgentApp", logger_settings: object) -> None:
    agent_types = agent_provider.registered_agent_types().values()
    has_parallel = any(agent_type == AgentType.PARALLEL for agent_type in agent_types)
    if has_parallel:
        rich_print("[dim]Markdown Streaming disabled (Parallel Agents configured)[/dim]")
        return

    preferences = resolve_streaming_preferences(logger_settings)
    if preferences.enabled:
        rich_print(f"[dim]Streaming Enabled - {preferences.mode} mode[/dim]")


def render_startup_notices(
    notices: list[object],
    *,
    agent_name: str,
    agent_provider: "AgentApp",
) -> None:
    for notice in notices:
        if isinstance(notice, StartupNotice) and notice.render_markdown:
            target_agent_name = notice.agent_name or agent_name
            target_display = None
            try:
                target_agent = agent_provider._agent(target_agent_name)
                target_display = getattr(target_agent, "display", None)
            except Exception:
                target_display = None

            if target_display is not None:
                if notice.title:
                    target_display.show_status_message(Text(notice.title, style="bold"))
                target_display.display_message(
                    content=notice.text,
                    message_type=MessageType.ASSISTANT,
                    name=target_agent_name,
                    right_info=notice.right_info or "",
                    truncate_content=False,
                    render_markdown=True,
                )
            else:
                rich_print(notice.text)
            continue

        rich_print(notice)
    notices.clear()


def show_a2a_prompt_status(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> None:
    if agent_provider is None:
        return
    try:
        agent = agent_provider._agent(agent_name)
    except Exception:
        return
    if not isinstance(agent, A2APromptStatusAgent):
        return
    status_line = agent.prompt_status_line()
    if status_line:
        rich_print(f"[dim]{escape_markup(status_line)}[/dim]")


__all__ = [
    "StartupNotice",
    "queue_startup_markdown_notice",
    "queue_startup_notice",
    "render_startup_notices",
    "show_a2a_prompt_status",
    "show_fast_agent_home_summary",
    "show_input_help_banner",
    "show_model_shortcut_hints",
    "show_shell_startup",
    "show_stop_hint_message",
]
