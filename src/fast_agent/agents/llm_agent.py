"""
LLM Agent class that adds interaction behaviors to LlmDecorator.

This class extends LlmDecorator with LLM-specific interaction behaviors including:
- UI display methods for messages, tools, and prompts
- Stop reason handling
- Tool call tracking
- Chat display integration
"""

import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from a2a.types import AgentCapabilities
from mcp import Tool
from mcp.types import ContentBlock
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_decorator import LlmDecorator, ModelT
from fast_agent.constants import (
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_URL_ELICITATION_CHANNEL,
)
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.model_display_name import resolve_llm_display_name
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.citation_display import (
    collect_citation_sources,
    render_sources_pre_content,
    web_tool_badges,
)
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.context_usage_display import format_compact_context_usage_percent
from fast_agent.ui.interactive_diagnostics import write_interactive_trace
from fast_agent.ui.message_display_helpers import (
    build_tool_use_additional_message,
    build_user_message_display,
    resolve_highlight_index,
    tool_use_requests_file_read_access,
    tool_use_requests_shell_access,
)
from fast_agent.workflow_telemetry import (
    NoOpWorkflowTelemetryProvider,
    WorkflowTelemetryProvider,
)

if TYPE_CHECKING:
    from fast_agent.agents.llm_decorator import RemovedContentSummary
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
        render_message: bool = True,
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
            render_message: When False, skip reprinting body text and only show post-turn output
        """
        shell_only_tool_use = tool_use_requests_shell_access(
            message,
            shell_tool_name=self._shell_tool_name_for_display(),
        )
        read_only_tool_use = tool_use_requests_file_read_access(message)
        additional_segments = self._build_stop_reason_additional_segments(
            message,
            shell_only_tool_use=shell_only_tool_use,
            read_only_tool_use=read_only_tool_use,
        )
        caller_additional_segment = self._coerce_additional_message_segment(additional_message)
        if caller_additional_segment is not None:
            additional_segments.append(caller_additional_segment)

        pre_segments, metadata_segments, bottom_items, highlight_items = (
            self._collect_post_turn_metadata(
                message,
                bottom_items=bottom_items,
                highlight_items=highlight_items,
            )
        )
        additional_segments.extend(metadata_segments)

        additional_message_text = self._combine_additional_segments(additional_segments)
        pre_content = self._combine_additional_segments(pre_segments)
        status_message_text = self._combine_additional_segments(
            [*pre_segments, *additional_segments]
        )
        display_name = self._resolve_assistant_display_name(name)
        display_model = self._resolve_assistant_display_model(message=message, model=model)
        bottom_items, highlight_items = self._filter_bottom_metadata_for_tool_use(
            bottom_items=bottom_items,
            highlight_items=highlight_items,
            shell_only_tool_use=shell_only_tool_use,
            read_only_tool_use=read_only_tool_use,
        )
        highlight_index = resolve_highlight_index(bottom_items, highlight_items)
        hook_indicator = self._resolve_assistant_hook_indicator(show_hook_indicator)
        message_text = message
        if render_message:
            await self.display.show_assistant_message(
                message_text,
                bottom_items=bottom_items,
                highlight_index=highlight_index,
                max_item_length=max_item_length,
                name=display_name,
                model=display_model,
                additional_message=additional_message_text,
                pre_content=pre_content,
                render_markdown=render_markdown,
                show_hook_indicator=hook_indicator,
            )
        else:
            if status_message_text is not None:
                self.display.show_status_message(status_message_text)
            self.display.show_mermaid_diagrams_from_message_text(message_text)
        self._display_url_elicitations_from_history(display_name)

    def _build_stop_reason_additional_segments(
        self,
        message: PromptMessageExtended,
        *,
        shell_only_tool_use: bool,
        read_only_tool_use: bool,
    ) -> list[Text]:
        segments: list[Text] = []
        match message.stop_reason:
            case LlmStopReason.END_TURN:
                pass
            case LlmStopReason.MAX_TOKENS:
                segments.append(
                    Text(
                        "\n\nMaximum output tokens reached - generation stopped.",
                        style="dim red italic",
                    )
                )
            case LlmStopReason.SAFETY:
                segments.append(
                    Text(
                        "\n\nContent filter activated - generation stopped.",
                        style="dim red italic",
                    )
                )
            case LlmStopReason.PAUSE:
                segments.append(Text("\n\nLLM has requested a pause.", style="dim green italic"))
            case LlmStopReason.STOP_SEQUENCE:
                segments.append(
                    Text(
                        "\n\nStop Sequence activated - generation stopped.",
                        style="dim red italic",
                    )
                )
            case LlmStopReason.TOOL_USE:
                tool_use_message = build_tool_use_additional_message(
                    message,
                    shell_access=shell_only_tool_use,
                    file_read=read_only_tool_use,
                )
                if tool_use_message is not None:
                    segments.append(tool_use_message)
            case LlmStopReason.ERROR:
                error_segment = self._build_error_additional_segment(message)
                if error_segment is not None:
                    segments.append(error_segment)
            case LlmStopReason.CANCELLED:
                segments.append(Text("\n\nGeneration cancelled by user.", style="dim yellow italic"))
            case _:
                if message.stop_reason:
                    segments.append(
                        Text(
                            f"\n\nGeneration stopped for an unhandled reason ({message.stop_reason})",
                            style="dim red italic",
                        )
                    )
        return segments

    def _build_error_additional_segment(self, message: PromptMessageExtended) -> Text | None:
        """Build error detail text while preserving current channel fallback behavior."""
        channels = message.channels
        if channels and FAST_AGENT_ERROR_CHANNEL in channels:
            error_blocks = channels[FAST_AGENT_ERROR_CHANNEL]
            if not error_blocks:
                return None
            error_text = get_text(error_blocks[0])
            if error_text:
                return Text(f"\n\nError details: {error_text}", style="dim red italic")
            return Text(f"\n\nError details: {str(error_blocks[0])}", style="dim red italic")
        return Text("\n\nAn error occurred during generation.", style="dim red italic")

    def _coerce_additional_message_segment(self, additional_message: Text | str | None) -> Text | None:
        if additional_message is None:
            return None
        if isinstance(additional_message, Text):
            return additional_message
        return Text(str(additional_message))

    def _should_show_post_turn_metadata(self, message: PromptMessageExtended) -> bool:
        return message.stop_reason != LlmStopReason.TOOL_USE

    def _collect_post_turn_metadata(
        self,
        message: PromptMessageExtended,
        *,
        bottom_items: list[str] | None,
        highlight_items: str | list[str] | None,
    ) -> tuple[list[Text], list[Text], list[str] | None, str | list[str] | None]:
        pre_segments: list[Text] = []
        additional_segments: list[Text] = []
        show_post_turn_metadata = self._should_show_post_turn_metadata(message)
        sources_text = render_sources_pre_content(message) if show_post_turn_metadata else None
        badge_items = web_tool_badges(message) if show_post_turn_metadata else []
        self._log_web_metadata_debug(
            message,
            show_post_turn_metadata=show_post_turn_metadata,
            sources_text=sources_text,
            badge_items=badge_items,
        )

        if sources_text is not None:
            pre_segments.append(sources_text)

        if badge_items:
            additional_segments.append(
                Text(f"\n\nWeb activity: {', '.join(badge_items)}", style="bright_cyan")
            )
            merged_bottom = list(bottom_items or [])
            for badge in badge_items:
                if badge not in merged_bottom:
                    merged_bottom.append(badge)
            bottom_items = merged_bottom
            if highlight_items is None:
                highlight_items = badge_items[0]

        return pre_segments, additional_segments, bottom_items, highlight_items

    def _log_web_metadata_debug(
        self,
        message: PromptMessageExtended,
        *,
        show_post_turn_metadata: bool,
        sources_text: Text | None,
        badge_items: Sequence[str],
    ) -> None:
        if not os.environ.get("FAST_AGENT_WEBDEBUG"):
            return

        channels = message.channels or {}
        channel_names = sorted(channels.keys()) if isinstance(channels, dict) else []
        source_count = len(collect_citation_sources(message))
        print(
            "[webdebug]"
            f" agent={self.name}"
            f" channels={channel_names}"
            f" server_tool_blocks={len(channels.get(ANTHROPIC_SERVER_TOOLS_CHANNEL, [])) if isinstance(channels, dict) else 0}"
            f" citation_blocks={len(channels.get(ANTHROPIC_CITATIONS_CHANNEL, [])) if isinstance(channels, dict) else 0}"
            f" source_count={source_count}"
            f" show_post_turn_metadata={show_post_turn_metadata}"
            f" sources_rendered={bool(sources_text)}"
        )
        print(f"[webdebug] agent={self.name} badges={list(badge_items)}")

    def _combine_additional_segments(self, segments: Sequence[Text]) -> Text | None:
        if not segments:
            return None
        combined = Text()
        for segment in segments:
            combined += segment
        return combined

    def _resolve_assistant_display_name(self, name: str | None) -> str | None:
        return name if name is not None else self.name

    def _resolve_assistant_display_model(
        self,
        *,
        message: PromptMessageExtended,
        model: str | None,
    ) -> str | None:
        display_model = model
        if display_model is None:
            display_model = resolve_llm_display_name(self.llm)
        if display_model is None:
            return None

        if self.llm is not None:
            websocket_indicator = getattr(self.llm, "websocket_turn_indicator", None)
            if isinstance(websocket_indicator, str) and websocket_indicator:
                display_model = f"{display_model} {websocket_indicator}"

        if message.tool_calls:
            usage_accumulator = self.usage_accumulator
            context_percentage = (
                usage_accumulator.context_usage_percentage if usage_accumulator else None
            )
            context_label = format_compact_context_usage_percent(context_percentage)
            if context_label is not None:
                display_model = f"{display_model} ({context_label})"

        return display_model

    def _filter_bottom_metadata_for_tool_use(
        self,
        *,
        bottom_items: list[str] | None,
        highlight_items: str | list[str] | None,
        shell_only_tool_use: bool,
        read_only_tool_use: bool,
    ) -> tuple[list[str] | None, str | list[str] | None]:
        if shell_only_tool_use or read_only_tool_use:
            return None, None
        return bottom_items, highlight_items

    def _resolve_assistant_hook_indicator(self, show_hook_indicator: bool | None) -> bool:
        if show_hook_indicator is not None:
            return show_hook_indicator
        return getattr(self, "has_after_llm_call_hook", False)

    def _shell_tool_name_for_display(self) -> str | None:
        """Return the tool name used for local shell execution, if any."""
        return None

    def _summary_text_for_result(
        self,
        message: PromptMessageExtended,
        summary: "RemovedContentSummary | None",
    ) -> Text | None:
        if summary is None or message.stop_reason == LlmStopReason.TOOL_USE:
            return None
        return Text(f"\n\n{summary.message}", style="dim red italic")

    def _can_preserve_streamed_final_frame(
        self,
        *,
        message: PromptMessageExtended,
        summary_text: Text | None,
        streaming_mode: str,
        stream_handle: "StreamingHandle",
    ) -> bool:
        """Return True when the streamed frame can replace final reprint safely."""
        if streaming_mode != "markdown":
            return False
        if summary_text is not None:
            return False
        if message.stop_reason != LlmStopReason.END_TURN:
            return False
        if stream_handle.has_scrolled():
            return False
        if collect_citation_sources(message):
            return False

        display_text = message.all_text() or message.last_text() or ""
        if not display_text.strip():
            return False
        return True

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

    def _get_previous_user_channels(self) -> dict[str, Sequence[ContentBlock]]:
        try:
            history = self.message_history
            if history and len(history) >= 2:
                prev = history[-2]
                if prev and prev.role == "user":
                    channels = prev.channels or {}
                    if isinstance(channels, dict):
                        return dict(channels)
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
            display_model = resolve_llm_display_name(llm)
            _, streaming_mode = self.display.resolve_streaming_preferences()
            render_markdown = True if streaming_mode == "markdown" else False

            remove_listener: Callable[[], None] | None = None
            remove_tool_listener: Callable[[], None] | None = None

            with self.display.streaming_assistant_message(
                name=display_name,
                model=display_model,
            ) as stream_handle:
                self._active_stream_handle = stream_handle
                write_interactive_trace(
                    "llm_agent.stream_handle.attach",
                    agent=self.name,
                    handle_id=id(stream_handle),
                )
                try:
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

                    summary_text = self._summary_text_for_result(result, summary)

                    await stream_handle.wait_for_drain()
                    self._maybe_close_streaming_for_tool_calls(result)
                    preserve_streamed_frame = self._can_preserve_streamed_final_frame(
                        message=result,
                        summary_text=summary_text,
                        streaming_mode=streaming_mode,
                        stream_handle=stream_handle,
                    ) and stream_handle.preserve_final_frame()
                    stream_handle.finalize(result)
                finally:
                    write_interactive_trace(
                        "llm_agent.stream_handle.clear",
                        agent=self.name,
                        handle_id=id(stream_handle),
                    )
                    self._active_stream_handle = None

            if preserve_streamed_frame:
                await self.show_assistant_message(
                    result,
                    additional_message=summary_text,
                    render_markdown=render_markdown,
                    render_message=False,
                )
            else:
                await self.show_assistant_message(
                    result,
                    additional_message=summary_text,
                    render_markdown=render_markdown,
                )
        else:
            result, summary = await self._generate_with_summary(messages, request_params, tools)
            summary_text = self._summary_text_for_result(result, summary)
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
        write_interactive_trace(
            "llm_agent.stream_handle.close",
            agent=self.name,
            handle_id=id(handle),
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
        summary_text = self._summary_text_for_result(message, summary)
        await self.show_assistant_message(message=message, additional_message=summary_text)
        return result, message
