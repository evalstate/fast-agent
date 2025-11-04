"""
LLM Agent class that adds interaction behaviors to LlmDecorator.

This class extends LlmDecorator with LLM-specific interaction behaviors including:
- UI display methods for messages, tools, and prompts
- Stop reason handling
- Tool call tracking
- Chat display integration
"""

from typing import Callable, List, Optional, Tuple

from a2a.types import AgentCapabilities
from mcp import Tool
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_decorator import LlmDecorator, ModelT
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.context import Context
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.console_display import ConsoleDisplay

# TODO -- decide what to do with type safety for model/chat_turn()

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

    @property
    def display(self) -> ConsoleDisplay:
        """UI display helper for presenting messages and tool activity."""
        return self._display

    @display.setter
    def display(self, value: ConsoleDisplay) -> None:
        self._display = value

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Optional[Text] = None,
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
                if None is message.last_text():
                    additional_segments.append(
                        Text("The assistant requested tool calls", style="dim green italic")
                    )

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
        display_model = model if model is not None else (self.llm.model_name if self._llm else None)

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

        await self.display.show_assistant_message(
            message_text,
            bottom_items=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
            name=display_name,
            model=display_model,
            additional_message=additional_message_text,
        )

    def show_user_message(self, message: PromptMessageExtended) -> None:
        """Display a user message in a formatted panel."""
        model = self.llm.model_name
        chat_turn = self._llm.chat_turn()
        self.display.show_user_message(message.last_text() or "", model, chat_turn, name=self.name)

    def _should_stream(self) -> bool:
        """Determine whether streaming display should be used."""
        if getattr(self, "display", None):
            enabled, _ = self.display.resolve_streaming_preferences()
            return enabled
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
            self.show_user_message(message=messages[-1])

        # TODO - manage error catch, recovery, pause
        summary_text: Text | None = None

        if self._should_stream():
            display_name = self.name
            display_model = self.llm.model_name if self._llm else None

            remove_listener: Callable[[], None] | None = None
            remove_tool_listener: Callable[[], None] | None = None

            with self.display.streaming_assistant_message(
                name=display_name,
                model=display_model,
            ) as stream_handle:
                try:
                    remove_listener = self.llm.add_stream_listener(stream_handle.update)
                    remove_tool_listener = self.llm.add_tool_stream_listener(
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

                stream_handle.finalize(result)

            await self.show_assistant_message(result, additional_message=summary_text)
        else:
            result, summary = await self._generate_with_summary(messages, request_params, tools)

            summary_text = (
                Text(f"\n\n{summary.message}", style="dim red italic") if summary else None
            )
            await self.show_assistant_message(result, additional_message=summary_text)

        return result

    async def structured_impl(
        self,
        messages: List[PromptMessageExtended],
        model: type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        if "user" == messages[-1].role:
            self.show_user_message(message=messages[-1])

        (result, message), summary = await self._structured_with_summary(
            messages, model, request_params
        )
        summary_text = Text(f"\n\n{summary.message}", style="dim red italic") if summary else None
        await self.show_assistant_message(message=message, additional_message=summary_text)
        return result, message
