"""
LLM Agent class that adds interaction behaviors to LlmDecorator.

This class extends LlmDecorator with LLM-specific interaction behaviors including:
- UI display methods for messages, tools, and prompts
- Stop reason handling
- Tool call tracking
- Chat display integration
"""

from typing import List

from a2a.types import AgentCapabilities
from mcp import Tool
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.context import Context
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
    LLM operations to the attached AugmentedLLMProtocol.
    """

    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
    ) -> None:
        super().__init__(config=config, context=context)

        # Initialize display component
        self.display = ConsoleDisplay(config=self._context.config if self._context else None)

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
    ) -> None:
        """Display an assistant message with appropriate styling based on stop reason.

        Args:
            message: The message to display
            bottom_items: Optional items for bottom bar (e.g., servers, destinations)
            highlight_items: Items to highlight in bottom bar
            max_item_length: Max length for bottom items
        """

        # Determine display content based on stop reason
        additional_message: Text | None = None

        match message.stop_reason:
            case LlmStopReason.END_TURN:
                # No additional message needed for normal end turn
                pass

            case LlmStopReason.MAX_TOKENS:
                additional_message = Text(
                    "\n\nMaximum output tokens reached - generation stopped.",
                    style="dim red italic",
                )

            case LlmStopReason.SAFETY:
                additional_message = Text(
                    "\n\nContent filter activated - generation stopped.", style="dim red italic"
                )

            case LlmStopReason.PAUSE:
                additional_message = Text(
                    "\n\nLLM has requested a pause.", style="dim green italic"
                )

            case LlmStopReason.STOP_SEQUENCE:
                additional_message = Text(
                    "\n\nStop Sequence activated - generation stopped.", style="dim red italic"
                )

            case LlmStopReason.TOOL_USE:
                if None is message.last_text():
                    additional_message = Text(
                        "The assistant requested tool calls", style="dim green italic"
                    )

            case _:
                if message.stop_reason:
                    additional_message = Text(
                        f"\n\nGeneration stopped for an unhandled reason ({message.stop_reason})",
                        style="dim red italic",
                    )

        message_text = message.last_text() or ""

        await self.display.show_assistant_message(
            message_text,
            bottom_items=bottom_items,
            highlight_items=highlight_items,
            max_item_length=max_item_length,
            name=self.name,
            model=self._llm.default_request_params.model
            if self._llm and hasattr(self._llm, "default_request_params")
            else None,
            additional_message=additional_message,
        )

    def show_user_message(self, message: PromptMessageExtended) -> None:
        """Display a user message in a formatted panel."""
        model = self._llm.default_request_params.model
        chat_turn = self._llm.chat_turn()
        self.display.show_user_message(message.last_text(), model, chat_turn, name=self.name)

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

        # TODO -- we should merge the request parameters here with the LLM defaults?
        # TODO - manage error catch, recovery, pause
        result = await super().generate_impl(messages, request_params, tools)

        await self.show_assistant_message(result)
        return result

    # async def show_prompt_loaded(
    #     self,
    #     prompt_name: str,
    #     description: Optional[str] = None,
    #     message_count: int = 0,
    #     arguments: Optional[dict[str, str]] = None,
    # ) -> None:
    #     """
    #     Display information about a loaded prompt template.

    #     Args:
    #         prompt_name: The name of the prompt
    #         description: Optional description of the prompt
    #         message_count: Number of messages in the prompt
    #         arguments: Optional dictionary of arguments passed to the prompt
    #     """
    #     # Get aggregator from attached LLM if available
    #     aggregator = None
    #     if self._llm and hasattr(self._llm, "aggregator"):
    #         aggregator = self._llm.aggregator

    #     await self.display.show_prompt_loaded(
    #         prompt_name=prompt_name,
    #         description=description,
    #         message_count=message_count,
    #         agent_name=self.name,
    #         aggregator=aggregator,
    #         arguments=arguments,
    #     )
