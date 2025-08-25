"""
LLM Agent class that adds interaction behaviors to LlmDecorator.

This class extends LlmDecorator with LLM-specific interaction behaviors including:
- UI display methods for messages, tools, and prompts
- Stop reason handling
- Tool call tracking
- Chat display integration
"""

from typing import List

from mcp import Tool
from rich.text import Text

from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.context import Context
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.ui.console_display import ConsoleDisplay

# TODO -- move tool counting logic to relevant place
# TODO -- decide what to do with type safety for model/chat_turn()


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

        # Tool call tracking for current turn
        self._current_turn_tool_calls = 0

    def _reset_turn_tool_calls(self) -> None:
        """Reset tool call counter for new turn."""
        self._current_turn_tool_calls = 0

    async def show_assistant_message(
        self,
        message: PromptMessageMultipart,
        highlight_namespaced_tool: str = "",
        title: str = "ASSISTANT",
    ) -> None:
        """Display an assistant message with appropriate styling based on stop reason."""

        # Determine display content based on stop reason
        additional_message: Text = Text()

        match message.stop_reason:
            case LlmStopReason.END_TURN:
                display_content = message.last_text() or "No content to display"

            case LlmStopReason.MAX_TOKENS:
                additional_message.append(
                    "\n\nMaximum output tokens reached - generation stopped.",
                    style="dim red italic",
                )

            case LlmStopReason.SAFETY:
                additional_message.append(
                    "\n\nContent filter activated - generation stopped.", style="dim red italic"
                )

            case LlmStopReason.PAUSE:
                additional_message.append(
                    "\n\nLLM has requested a pause.", style="dim green italic"
                )

            case LlmStopReason.STOP_SEQUENCE:
                # Create a rich Text object with the actual content plus warning
                additional_message.append(
                    "\n\nStop Sequence activated - generation stopped.", style="dim red italic"
                )

            case LlmStopReason.TOOL_USE:
                if None is message.last_text():
                    additional_message.append(
                        "The assistant requested tool calls", "dim green italic"
                    )

            case _:
                additional_message.append(
                    f"Generation stopped for an unhandled reason ({message.stop_reason or 'unknown'})",
                    style="dim red italic",
                )

        message_text = message.last_text() or ""
        display_content = Text(message_text).append(additional_message)

        await self.display.show_assistant_message(
            display_content,
            aggregator=None,
            highlight_namespaced_tool=highlight_namespaced_tool,
            title=title,
            name=self.name,
        )

    def show_user_message(self, message: PromptMessageMultipart) -> None:
        """Display a user message in a formatted panel."""
        model = self._llm.default_request_params.model
        chat_turn = self._llm.chat_turn()
        self.display.show_user_message(message.last_text(), model, chat_turn, name=self.name)

    async def generate_impl(
        self,
        messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageMultipart:
        """
        Enhanced generate implementation that resets tool call tracking.
        Messages are already normalized to List[PromptMessageMultipart].
        """
        self._reset_turn_tool_calls()
        if "user" == messages[-1].role:
            self.show_user_message(message=messages[-1])

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
