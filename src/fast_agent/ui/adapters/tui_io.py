"""TUI adapter implementation for shared command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rich.text import Text

from fast_agent.commands.context import CommandIO
from fast_agent.commands.results import CommandMessage
from fast_agent.config import Settings, get_settings
from fast_agent.llm.model_reference_config import resolve_model_reference_start_path
from fast_agent.ui.enhanced_prompt import get_argument_input, get_selection_input
from fast_agent.ui.history_actions import display_history_turn
from fast_agent.ui.message_primitives import MessageType
from fast_agent.ui.model_picker_common import normalize_generic_model_spec
from fast_agent.ui.model_reference_picker import (
    ModelReferencePickerItem,
    run_model_reference_picker_async,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import AgentProvider
    from fast_agent.config import Settings
    from fast_agent.llm.model_reference_diagnostics import ModelReferenceSetupItem
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


@runtime_checkable
class TuiStatusDisplay(Protocol):
    @property
    def markup_enabled(self) -> bool: ...

    def show_status_message(self, content: Text) -> None: ...


@runtime_checkable
class TuiMarkdownDisplay(TuiStatusDisplay, Protocol):
    def display_message(
        self,
        *,
        content: str,
        message_type: MessageType,
        name: str,
        right_info: str,
        truncate_content: bool,
        render_markdown: bool,
    ) -> None: ...


@runtime_checkable
class TuiSystemDisplay(TuiStatusDisplay, Protocol):
    def show_system_message(
        self,
        system_prompt: str,
        *,
        agent_name: str,
        server_count: int = 0,
    ) -> None: ...


@runtime_checkable
class TuiContextCarrier(Protocol):
    @property
    def config(self) -> "Settings | None": ...


@runtime_checkable
class TuiDisplayAgent(Protocol):
    @property
    def display(self) -> TuiStatusDisplay: ...


@runtime_checkable
class TuiContextAgent(Protocol):
    @property
    def context(self) -> TuiContextCarrier | None: ...


@dataclass(slots=True)
class TuiCommandIO(CommandIO):
    """Command IO implementation backed by the interactive TUI."""

    prompt_provider: "AgentProvider"
    agent_name: str
    settings: Settings | None = None
    config_payload: dict[str, Any] | None = None

    @staticmethod
    def _normalize_reference_token(token: str | None) -> str | None:
        if token is None:
            return None
        stripped = token.strip()
        if not stripped:
            return stripped
        if stripped.startswith("$"):
            return stripped
        return f"${stripped}"

    def _resolve_display(self, agent_name: str | None):
        from fast_agent.ui.console_display import ConsoleDisplay

        target_agent = None
        if agent_name:
            try:
                target_agent = self.prompt_provider._agent(agent_name)
            except Exception:
                target_agent = None

        if isinstance(target_agent, TuiDisplayAgent):
            return target_agent.display

        config = None
        if isinstance(target_agent, TuiContextAgent) and target_agent.context is not None:
            config = target_agent.context.config

        if config is None:
            config = self.settings or get_settings()

        return ConsoleDisplay(config=config)

    @staticmethod
    def _apply_channel_style(content: Text, channel: str) -> None:
        if channel == "error":
            content.stylize("red")
        elif channel == "warning":
            content.stylize("yellow")
        elif channel == "info":
            content.stylize("cyan")

    async def _emit_markdown_message(
        self,
        display: TuiStatusDisplay,
        message: CommandMessage,
    ) -> None:
        content = message.text
        markdown_text = content.plain if isinstance(content, Text) else str(content)

        if message.title:
            title = Text(message.title, style="bold")
            self._apply_channel_style(title, message.channel)
            display.show_status_message(title)

        if isinstance(display, TuiMarkdownDisplay):
            display.display_message(
                content=markdown_text,
                message_type=MessageType.ASSISTANT,
                name=message.agent_name or self.agent_name,
                right_info=message.right_info or "",
                truncate_content=False,
                render_markdown=True,
            )
            return

        fallback = Text(markdown_text)
        self._apply_channel_style(fallback, message.channel)
        display.show_status_message(fallback)

    async def emit(self, message: CommandMessage) -> None:
        display = self._resolve_display(message.agent_name or self.agent_name)
        if message.render_markdown:
            await self._emit_markdown_message(display, message)
            return

        content = message.text

        if not isinstance(content, Text):
            if display.markup_enabled:
                content = Text.from_markup(str(content))
            else:
                content = Text(str(content))

        if message.title:
            header = Text(message.title, style="bold")
            if content.plain:
                header.append("\n")
                header.append_text(content)
            content = header

        self._apply_channel_style(content, message.channel)

        display.show_status_message(content)

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        arg_name = prompt.rstrip(":")
        value = await get_argument_input(
            arg_name=arg_name,
            description=None,
            required=not allow_empty,
            default=default,
        )
        if value is None or value == "":
            return default if default is not None else value
        return value

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        return await get_selection_input(
            prompt,
            options=list(options),
            allow_cancel=allow_cancel,
            default=default,
        )

    async def prompt_model_selection(
        self,
        *,
        initial_provider: str | None = None,
        default_model: str | None = None,
    ) -> str | None:
        from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error
        from fast_agent.llm.provider.openai.codex_oauth import login_codex_oauth
        from fast_agent.llm.provider_types import Provider
        from fast_agent.ui import console
        from fast_agent.ui.model_picker import run_model_picker_async

        provider_name = initial_provider

        while True:
            picker_result = await run_model_picker_async(
                config_payload=(
                    self.config_payload
                    if self.config_payload is not None
                    else self.settings.model_dump() if self.settings is not None else None
                ),
                start_path=(
                    resolve_model_reference_start_path(settings=self.settings)
                    if self.settings is not None
                    else None
                ),
                initial_provider=provider_name,
                initial_model_spec=default_model,
            )
            if picker_result is None:
                return None

            provider_name = picker_result.provider

            if picker_result.activation_action is not None:
                if picker_result.activation_action != "codex-login":
                    await self.emit(
                        CommandMessage(
                            text=(
                                "Selected provider requires an activation flow that is not "
                                "supported in this prompt yet."
                            ),
                            channel="warning",
                            agent_name=self.agent_name,
                        )
                    )
                    return None

                await self.emit(
                    CommandMessage(
                        text="Starting Codex OAuth login…",
                        channel="info",
                        agent_name=self.agent_name,
                    )
                )
                try:
                    console.ensure_blocking_console()
                    login_codex_oauth()
                except ProviderKeyError as exc:
                    await self.emit(
                        CommandMessage(
                            text=format_fast_agent_error(exc),
                            channel="error",
                            agent_name=self.agent_name,
                        )
                    )
                except (EOFError, KeyboardInterrupt):
                    await self.emit(
                        CommandMessage(
                            text="Codex OAuth login cancelled.",
                            channel="warning",
                            agent_name=self.agent_name,
                        )
                    )
                    return None
                else:
                    await self.emit(
                        CommandMessage(
                            text="Codex OAuth login complete. Choose a Codex model to continue.",
                            channel="info",
                            agent_name=self.agent_name,
                        )
                    )
                continue

            if (
                picker_result.provider == Provider.GENERIC.config_name
                and picker_result.resolved_model is None
            ):
                prompt_default = (default_model or "").strip() or "llama3.2"
                while True:
                    entered = await self.prompt_text(
                        "Local model (e.g. llama3.2):",
                        default=prompt_default,
                        allow_empty=False,
                    )
                    if entered is None:
                        return None
                    normalized = normalize_generic_model_spec(entered)
                    if normalized:
                        return normalized
                    await self.emit(
                        CommandMessage(
                            text="Please enter a non-empty model string.",
                            channel="warning",
                            agent_name=self.agent_name,
                        )
                    )

            if picker_result.refer_to_docs or not picker_result.resolved_model:
                await self.emit(
                    CommandMessage(
                        text=(
                            "Selected provider requires a concrete model ID. "
                            "Choose a listed model or cancel."
                        ),
                        channel="warning",
                        agent_name=self.agent_name,
                    )
                )
                continue

            return picker_result.resolved_model or picker_result.selected_model

    async def pick_model_reference_token(
        self,
        *,
        items: tuple["ModelReferenceSetupItem", ...],
    ) -> str | None:
        picker_items = tuple(
            ModelReferencePickerItem(
                token=item.token,
                priority=item.priority,
                status=f"{item.priority}/{item.status}",
                summary=item.summary,
                current_value=item.current_value,
                references=item.references,
                removable=False,
            )
            for item in items
        )
        result = await run_model_reference_picker_async(picker_items)
        if result is None:
            return None
        if result.action == "custom":
            return self._normalize_reference_token(
                await self.prompt_text(
                    "Reference token ($namespace.key):",
                    allow_empty=False,
                )
            )
        return result.token

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        return await get_argument_input(
            arg_name=arg_name,
            description=description,
            required=required,
        )

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        await display_history_turn(
            agent_name,
            turn,
            config=self.settings or get_settings(),
            turn_index=turn_index,
            total_turns=total_turns,
        )

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        from fast_agent.ui.history_display import display_history_overview

        display_history_overview(agent_name, history, usage)

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        from fast_agent.ui.usage_display import display_usage_report

        display_usage_report(agents, show_if_progress_disabled=True)

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        display = self._resolve_display(agent_name)
        if isinstance(display, TuiSystemDisplay):
            display.show_system_message(
                system_prompt,
                agent_name=agent_name,
                server_count=server_count,
            )
            return

        from fast_agent.ui.console_display import ConsoleDisplay

        ConsoleDisplay(config=self.settings or get_settings()).show_system_message(
            system_prompt,
            agent_name=agent_name,
            server_count=server_count,
        )
