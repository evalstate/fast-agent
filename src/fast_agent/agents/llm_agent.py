"""
LLM Agent class that adds interaction behaviors to LlmDecorator.

This class extends LlmDecorator with LLM-specific interaction behaviors including:
- UI display methods for messages, tools, and prompts
- Stop reason handling
- Tool call tracking
- Chat display integration
"""

import json
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from a2a.types import AgentCapabilities
from mcp import Tool
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_decorator import LlmDecorator, ModelT
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL, FAST_AGENT_URL_ELICITATION_CHANNEL
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.message_display_helpers import (
    build_tool_use_additional_message,
    build_user_message_display,
)
from fast_agent.workflow_telemetry import (
    NoOpWorkflowTelemetryProvider,
    WorkflowTelemetryProvider,
)

if TYPE_CHECKING:
    from fast_agent.agents.tool_runner import ToolRunnerHooks
    from fast_agent.ui.streaming import StreamingHandle
# TODO -- decide what to do with type safety for model/chat_turn()

logger = get_logger(__name__)

DEFAULT_CAPABILITIES = AgentCapabilities(
    streaming=False, push_notifications=False, state_transition_history=False
)


class LlmAgent(LlmDecorator):
    """
    An LLM agent that adds interaction behaviors to the base LlmDecorator.

    This class provides LLM-specific functionality including UI display methods,
    tool call tracking, and chat interaction patterns while delegating core
    LLM operations to the attached FastAgentLLMProtocol.
    """

    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
    ) -> None:
        super().__init__(config=config, context=context)

        # Initialize display component
        self._display = ConsoleDisplay(config=self._context.config if self._context else None)
        self._force_non_streaming_once = False
        self._force_non_streaming_reason: str | None = None
        self._active_stream_handle: "StreamingHandle | None" = None
        self.tool_runner_hooks: "ToolRunnerHooks | None" = None
        self._workflow_telemetry_provider: WorkflowTelemetryProvider = (
            NoOpWorkflowTelemetryProvider()
        )

    @property
    def display(self) -> ConsoleDisplay:
        """UI display helper for presenting messages and tool activity."""
        return self._display

    @display.setter
    def display(self, value: ConsoleDisplay) -> None:
        self._display = value

    @property
    def workflow_telemetry(self) -> WorkflowTelemetryProvider:
        """Telemetry provider for emitting workflow delegation steps."""
        return self._workflow_telemetry_provider

    @workflow_telemetry.setter
    def workflow_telemetry(self, provider: WorkflowTelemetryProvider | None) -> None:
        if provider is None:
            provider = NoOpWorkflowTelemetryProvider()
        self._workflow_telemetry_provider = provider

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Optional[Text] = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool | None = None,
    ) -> None:
        """Display an assistant message with appropriate styling based on stop reason.

        Args:
            message: The message to display
            bottom_items: Optional items for bottom bar (e.g., servers, destinations)
            highlight_items: Items to highlight in bottom bar
            max_item_length: Max length for bottom items
            name: Optional agent name to display
            model: Optional model name to display
            additional_message: Optional additional message to display
            render_markdown: Force markdown rendering (True) or plain rendering (False)
        """

        # Determine display content based on stop reason if not provided
        additional_segments: List[Text] = []

        # Generate additional message based on stop reason
        match message.stop_reason:
            case LlmStopReason.END_TURN:
                pass

            case LlmStopReason.MAX_TOKENS:
                additional_segments.append(
                    Text(
                        "\n\nMaximum output tokens reached - generation stopped.",
                        style="dim red italic",
                    )
                )

            case LlmStopReason.SAFETY:
                additional_segments.append(
                    Text(
                        "\n\nContent filter activated - generation stopped.",
                        style="dim red italic",
                    )
                )

            case LlmStopReason.PAUSE:
                additional_segments.append(
                    Text("\n\nLLM has requested a pause.", style="dim green italic")
                )

            case LlmStopReason.STOP_SEQUENCE:
                additional_segments.append(
                    Text(
                        "\n\nStop Sequence activated - generation stopped.",
                        style="dim red italic",
                    )
                )

            case LlmStopReason.TOOL_USE:
                tool_use_message = build_tool_use_additional_message(message)
                if tool_use_message is not None:
                    additional_segments.append(tool_use_message)

            case LlmStopReason.ERROR:
                # Check if there's detailed error information in the error channel
                if message.channels and FAST_AGENT_ERROR_CHANNEL in message.channels:
                    error_blocks = message.channels[FAST_AGENT_ERROR_CHANNEL]
                    if error_blocks:
                        # Extract text from the error block using the helper function
                        error_text = get_text(error_blocks[0])
                        if error_text:
                            additional_segments.append(
                                Text(f"\n\nError details: {error_text}", style="dim red italic")
                            )
                        else:
                            # Fallback if we couldn't extract text
                            additional_segments.append(
                                Text(
                                    f"\n\nError details: {str(error_blocks[0])}",
                                    style="dim red italic",
                                )
                            )
                else:
                    # Fallback if no detailed error is available
                    additional_segments.append(
                        Text("\n\nAn error occurred during generation.", style="dim red italic")
                    )

            case LlmStopReason.CANCELLED:
                additional_segments.append(
                    Text("\n\nGeneration cancelled by user.", style="dim yellow italic")
                )

            case _:
                if message.stop_reason:
                    additional_segments.append(
                        Text(
                            f"\n\nGeneration stopped for an unhandled reason ({message.stop_reason})",
                            style="dim red italic",
                        )
                    )

        if additional_message is not None:
            additional_segments.append(
                additional_message
                if isinstance(additional_message, Text)
                else Text(str(additional_message))
            )

        additional_message_text = None
        if additional_segments:
            combined = Text()
            for segment in additional_segments:
                combined += segment
            additional_message_text = combined

        message_text = message

        # Use provided name/model or fall back to defaults
        display_name = name if name is not None else self.name
        display_model = model if model is not None else (self.llm.model_name if self.llm else None)

        if message.tool_calls and display_model is not None:
            usage_accumulator = self.usage_accumulator
            context_percentage = (
                usage_accumulator.context_usage_percentage if usage_accumulator else None
            )
            if context_percentage is not None:
                display_model = f"{display_model} ({context_percentage:.1f}%)"

        # Convert highlight_items to highlight_index
        highlight_index = None
        if highlight_items and bottom_items:
            if isinstance(highlight_items, str):
                try:
                    highlight_index = bottom_items.index(highlight_items)
                except ValueError:
                    pass
            elif isinstance(highlight_items, list) and len(highlight_items) > 0:
                try:
                    highlight_index = bottom_items.index(highlight_items[0])
                except ValueError:
                    pass

        # Use explicit show_hook_indicator if provided, otherwise check for after_llm_call hook
        hook_indicator = (
            show_hook_indicator
            if show_hook_indicator is not None
            else getattr(self, "has_after_llm_call_hook", False)
        )
        await self.display.show_assistant_message(
            message_text,
            bottom_items=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
            name=display_name,
            model=display_model,
            additional_message=additional_message_text,
            render_markdown=render_markdown,
            show_hook_indicator=hook_indicator,
        )
        self._display_url_elicitations_from_history(display_name)

    def _display_url_elicitations_from_history(self, agent_name: str | None) -> None:
        """Display deferred URL elicitations from the previous tool-result turn."""
        channels = self._get_previous_user_channels()
        if not channels:
            return

        payload_blocks = channels.get(FAST_AGENT_URL_ELICITATION_CHANNEL, [])
        if not payload_blocks:
            return

        payload_entries: list[dict[str, object]] = []
        for block in payload_blocks:
            raw_text = get_text(block)
            if not raw_text:
                continue
            try:
                decoded = json.loads(raw_text)
            except Exception:
                continue

            if isinstance(decoded, list):
                payload_entries.extend(item for item in decoded if isinstance(item, dict))
            elif isinstance(decoded, dict):
                payload_entries.append(decoded)

        for payload in payload_entries:
            self._display_single_url_elicitation_payload(payload, agent_name)

    def _get_previous_user_channels(self) -> dict[str, list]:
        try:
            history = self.message_history
            if history and len(history) >= 2:
                prev = history[-2]
                if prev and prev.role == "user":
                    channels = prev.channels or {}
                    if isinstance(channels, dict):
                        return channels
        except Exception:
            pass
        return {}

    def _display_single_url_elicitation_payload(
        self,
        payload: dict[str, object],
        agent_name: str | None,
    ) -> None:
        from fast_agent.ui import console

        server_name = str(payload.get("server_name", "unknown"))
        raw_elicitations = payload.get("elicitations")
        raw_issues = payload.get("issues")

        elicitations = (
            [item for item in raw_elicitations if isinstance(item, dict)]
            if isinstance(raw_elicitations, list)
            else []
        )
        issues = [str(item) for item in raw_issues] if isinstance(raw_issues, list) else []

        if elicitations:
            count = len(elicitations)
            for index, elicitation in enumerate(elicitations, start=1):
                message = str(elicitation.get("message", "Authorization required."))
                if count > 1:
                    message = f"[{index}/{count}] {message}"
                url = str(elicitation.get("url", ""))
                elicitation_id = str(elicitation.get("elicitation_id", ""))

                self.display.show_url_elicitation(
                    message=message,
                    url=url,
                    server_name=server_name,
                    agent_name=agent_name,
                    elicitation_id=elicitation_id,
                )

        if issues:
            if elicitations:
                console.console.print(
                    "[yellow]"
                    f"MCP server {server_name} returned non-compliant URL elicitation payload:"
                    "[/yellow]"
                )
            else:
                console.console.print(
                    "[yellow]"
                    f"MCP server {server_name} returned malformed URL elicitation error.data:"
                    "[/yellow]"
                )

            for issue in issues[:3]:
                console.console.print(f"[dim yellow]  - {issue}[/dim yellow]")

            hidden_issue_count = len(issues) - 3
            if hidden_issue_count > 0:
                console.console.print(
                    f"[dim yellow]  - ... and {hidden_issue_count} more issue(s)[/dim yellow]"
                )

            if not elicitations:
                console.console.print(
                    "[dim yellow]"
                    "No valid URL elicitations could be extracted from error.data."
                    "[/dim yellow]"
                )

    def _display_user_messages(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
    ) -> None:
        if not messages:
            return

        display_messages = [
            message for message in messages if self._should_display_user_message(message)
        ]
        if not display_messages:
            return

        _ = request_params
        part_count = len(display_messages)

        message_text, attachments = build_user_message_display(display_messages)

        self.display.show_user_message(
            message_text,
            chat_turn=0,
            name=self.name,
            attachments=attachments if attachments else None,
            part_count=part_count if part_count > 1 else None,
            show_hook_indicator=getattr(self, "has_before_llm_call_hook", False),
        )

    def show_user_message(self, message: PromptMessageExtended) -> None:
        """Display a user message in a formatted panel."""
        self._display_user_messages([message])

    def _should_display_user_message(self, message: PromptMessageExtended) -> bool:
        return True

    def _should_stream(self) -> bool:
        """Determine whether streaming display should be used."""
        if self._force_non_streaming_once:
            self._force_non_streaming_once = False
            reason = self._force_non_streaming_reason
            self._force_non_streaming_reason = None
            if reason:
                logger.info(
                    "Streaming disabled for next turn",
                    agent_name=self.name,
                    reason=reason,
                )
            return False
        if getattr(self, "display", None):
            enabled, _ = self.display.resolve_streaming_preferences()
            return enabled
        return True

    def force_non_streaming_next_turn(self, *, reason: str | None = None) -> bool:
        """Disable streaming for the next assistant turn."""
        if self._force_non_streaming_once:
            if reason and not self._force_non_streaming_reason:
                self._force_non_streaming_reason = reason
            return False
        self._force_non_streaming_once = True
        self._force_non_streaming_reason = reason
        return True

    async def generate_impl(
        self,
        messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Enhanced generate implementation that resets tool call tracking.
        Messages are already normalized to List[PromptMessageExtended].
        """

        if "user" == messages[-1].role:
            trailing_users: list[PromptMessageExtended] = []
            for message in reversed(messages):
                if message.role != "user":
                    break
                trailing_users.append(message)
            self._display_user_messages(
                list(reversed(trailing_users)), request_params=request_params
            )

        # TODO - manage error catch, recovery, pause
        summary_text: Text | None = None

        if self._should_stream():
            llm = self._require_llm()
            display_name = self.name
            display_model = llm.model_name
            _, streaming_mode = self.display.resolve_streaming_preferences()
            render_markdown = True if streaming_mode == "markdown" else False

            remove_listener: Callable[[], None] | None = None
            remove_tool_listener: Callable[[], None] | None = None

            with self.display.streaming_assistant_message(
                name=display_name,
                model=display_model,
            ) as stream_handle:
                self._active_stream_handle = stream_handle
                try:
                    remove_listener = llm.add_stream_listener(stream_handle.update_chunk)
                    remove_tool_listener = llm.add_tool_stream_listener(
                        stream_handle.handle_tool_event
                    )
                except Exception:
                    remove_listener = None
                    remove_tool_listener = None

                try:
                    result, summary = await self._generate_with_summary(
                        messages, request_params, tools
                    )
                finally:
                    if remove_listener:
                        remove_listener()
                    if remove_tool_listener:
                        remove_tool_listener()

                if summary:
                    summary_text = Text(f"\n\n{summary.message}", style="dim red italic")

                self._maybe_close_streaming_for_tool_calls(result)
                stream_handle.finalize(result)
                self._active_stream_handle = None

            await self.show_assistant_message(
                result,
                additional_message=summary_text,
                render_markdown=render_markdown,
            )
        else:
            result, summary = await self._generate_with_summary(messages, request_params, tools)

            summary_text = (
                Text(f"\n\n{summary.message}", style="dim red italic") if summary else None
            )
            await self.show_assistant_message(result, additional_message=summary_text)

        return result

    def close_active_streaming_display(self, *, reason: str | None = None) -> bool:
        """Close the current streaming display if active."""
        handle = self._active_stream_handle
        if handle is None:
            return False
        if reason:
            logger.info(
                "Closing active streaming display",
                agent_name=self.name,
                reason=reason,
            )
        try:
            handle.close()
        finally:
            self._active_stream_handle = None
        return True

    def _maybe_close_streaming_for_tool_calls(
        self, message: PromptMessageExtended
    ) -> None:
        tool_calls = message.tool_calls
        if not tool_calls or len(tool_calls) <= 1:
            logger.debug(
                "Streaming tool-call guard: no parallel tool calls found",
                agent_name=self.name,
                tool_call_count=len(tool_calls or {}),
            )
            return
        tool_call_items = list(tool_calls.items())
        subagent_calls = 0
        counter = getattr(self, "_count_agent_tool_calls", None)
        if callable(counter):
            try:
                subagent_calls = counter(tool_call_items)
            except Exception:
                subagent_calls = 0
        logger.debug(
            "Streaming tool-call guard: evaluated tool calls",
            agent_name=self.name,
            tool_call_count=len(tool_call_items),
            subagent_call_count=subagent_calls,
        )
        if subagent_calls > 1:
            self.close_active_streaming_display(
                reason="parallel subagent tool calls"
            )

    async def structured_impl(
        self,
        messages: List[PromptMessageExtended],
        model: type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        if "user" == messages[-1].role:
            trailing_users: list[PromptMessageExtended] = []
            for message in reversed(messages):
                if message.role != "user":
                    break
                trailing_users.append(message)
            self._display_user_messages(
                list(reversed(trailing_users)), request_params=request_params
            )

        (result, message), summary = await self._structured_with_summary(
            messages, model, request_params
        )
        summary_text = Text(f"\n\n{summary.message}", style="dim red italic") if summary else None
        await self.show_assistant_message(message=message, additional_message=summary_text)
        return result, message
