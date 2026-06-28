from typing import Any

from mcp import Tool

from fast_agent.core.exceptions import ModelConfigError
from fast_agent.interfaces import ModelT
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import create_turn_usage_from_messages
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompts.prompt_helpers import MessageContent
from fast_agent.types import PromptMessageExtended, RequestParams

# TODO -- support tool usage/replay


class PlaybackLLM(PassthroughLLM):
    """
    A specialized LLM implementation that plays back assistant messages when loaded with prompts.

    Unlike the PassthroughLLM which simply passes through messages without modification,
    PlaybackLLM is designed to simulate a conversation by playing back prompt messages
    in sequence when loaded with prompts through apply_prompt_template.

    After apply_prompts has been called, each call to generate_str returns the next
    "ASSISTANT" message in the loaded messages. If no messages are set or all messages have
    been played back, it returns a message indicating that messages are exhausted.
    """

    def __init__(self, name: str = "Playback", **kwargs: dict[str, Any]) -> None:
        super().__init__(name=name, provider=Provider.FAST_AGENT, **kwargs)
        self._messages: list[PromptMessageExtended] = []
        self._current_index = -1
        self._overage = -1

    def _get_next_assistant_message(self) -> PromptMessageExtended:
        """
        Get the next assistant message from the loaded messages.
        Increments the current message index and skips user messages.
        """
        # Find next assistant message
        while self._current_index < len(self._messages):
            message = self._messages[self._current_index]
            self._current_index += 1
            if message.role != "assistant":
                continue

            return message

        self._overage += 1
        return Prompt.assistant(
            f"MESSAGES EXHAUSTED (list size {len(self._messages)}) ({self._overage} overage)"
        )

    async def generate(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Handle playback of messages in two modes:
        1. First call: store messages for playback and return "HISTORY LOADED"
        2. Subsequent calls: return the next assistant message
        """
        del request_params, tools

        # If this is the first call (initialization) or we're loading a prompt template
        # with multiple messages (comes from apply_prompt)
        if self._current_index == -1:
            if len(messages) > 1:
                self._messages = list(messages)
            else:
                self._messages.extend(messages)

            # Reset the index to the beginning for proper playback
            self._current_index = 0

            # In PlaybackLLM, we always return "HISTORY LOADED" on initialization,
            # regardless of the prompt content. The next call will return messages.
            return Prompt.assistant(f"HISTORY LOADED ({len(self._messages)}) messages")

        response = self._get_next_assistant_message()

        # Track usage for this playback "turn"
        try:
            input_content = str(messages) if messages else ""
            output_content = MessageContent.get_first_text(response) or ""

            turn_usage = create_turn_usage_from_messages(
                input_content=input_content,
                output_content=output_content,
                model="playback",
                model_type="playback",
                tool_calls=0,
                delay_seconds=0.0,
            )
            self.usage_accumulator.add_turn(turn_usage)

        except Exception as e:
            self.logger.warning(f"Failed to track usage: {e}")

        return response

    async def structured(
        self,
        messages: list[PromptMessageExtended],
        model: type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """
        Handle structured requests by returning the next assistant message.
        """
        del messages, request_params

        if self._current_index == -1:
            raise ModelConfigError("Use generate() to load playback history")

        return self._structured_from_multipart(
            self._get_next_assistant_message(),
            model,
        )
