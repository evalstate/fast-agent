# from openai.types.beta.chat import
from typing import List

from mcp import Tool
from mcp.types import ContentBlock, TextContent
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.responses import (
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from fast_agent.constants import REASONING
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

_logger = get_logger(__name__)

DEFAULT_RESPONSES_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "medium"


# model selection
# system prompt
# usage info
# reasoning/thinking display and summary
# encrypted tokens


class ResponsesLLM(FastAgentLLM[ChatCompletionMessageParam, ChatCompletionMessage]):
    """LLM implementation for OpenAI's Responses models."""

    # OpenAI-specific parameter exclusions

    def __init__(self, provider=Provider.RESPONSES, *args, **kwargs):
        super().__init__(*args, provider=provider, **kwargs)

    async def _responses_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self._api_key())

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        responses_client = await self._responses_client()

        async with responses_client.responses.stream(
            model="gpt-5-mini",
            instructions="You are a helpful assistant.",
            input=multipart_messages[-1].all_text(),
            reasoning={"summary": "auto", "effort": DEFAULT_REASONING_EFFORT},
        ) as stream:
            reasoning_chars: int = 0
            text_chars: int = 0

            async for event in stream:
                if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
                    reasoning_chars += len(event.delta)
                    await self._emit_streaming_progress(
                        model="gpt-5-mini (thinking)",
                        new_total=reasoning_chars,
                        type=ProgressAction.THINKING,
                    )
                if isinstance(event, ResponseTextDeltaEvent):
                    # Notify stream listeners with the delta text
                    self._notify_stream_listeners(event.delta)
                    text_chars += len(event.delta)
                    await self._emit_streaming_progress(
                        model="gpt-5-mini",
                        new_total=text_chars,
                    )

            final_response = await stream.get_final_response()
            reasoning_content: List[ContentBlock] = []
            for output_item in final_response.output:
                if isinstance(output_item, ResponseReasoningItem):
                    summary_text = "\n".join(part.text for part in output_item.summary if part.text)
                    # reasoning text is not supplied by openai - leaving for future use with other providers
                    reasoning_text = "".join(
                        chunk.text
                        for chunk in (output_item.content or [])
                        if chunk.type == "reasoning_text"
                    )
                    if summary_text.strip():
                        reasoning_content.append(TextContent(type="text", text=summary_text.strip()))
                    if reasoning_text.strip():
                        reasoning_content.append(
                            TextContent(type="text", text=reasoning_text.strip())
                        )
        channels = {REASONING: reasoning_content} if reasoning_content else None

        return PromptMessageExtended(
            role="assistant",
            channels=channels,
            content=[TextContent(type="text", text=final_response.output_text)],
            stop_reason=LlmStopReason.END_TURN,
        )

    async def _emit_streaming_progress(
        self,
        model: str,
        new_total: int,
        type: ProgressAction = ProgressAction.STREAMING,
    ) -> None:
        """Emit a streaming progress event.

        Args:
            model: The model being used.
            new_total: The new total token count.
        """
        token_str = str(new_total).rjust(5)

        # Emit progress event
        data = {
            "progress_action": type,
            "model": model,
            "agent_name": self.name,
            "chat_turn": self.chat_turn(),
            "details": token_str.strip(),  # Token count goes in details for STREAMING action
        }
        self.logger.info("Streaming progress", data=data)
