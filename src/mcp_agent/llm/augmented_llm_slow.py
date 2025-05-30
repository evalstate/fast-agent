import asyncio
from typing import Any, List, Optional, Union

from mcp_agent.llm.augmented_llm import (
    MessageParamT,
    RequestParams,
)
from mcp_agent.llm.augmented_llm_passthrough import PassthroughLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class SlowLLM(PassthroughLLM):
    """
    A specialized LLM implementation that sleeps for 3 seconds before responding like PassthroughLLM.

    This is useful for testing scenarios where you want to simulate slow responses
    or for debugging timing-related issues in parallel workflows.
    """

    def __init__(
        self, provider=Provider.FAST_AGENT, name: str = "Slow", **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(name=name, provider=provider, **kwargs)

    async def generate_str(
        self,
        message: Union[str, MessageParamT, List[MessageParamT]],
        request_params: Optional[RequestParams] = None,
    ) -> str:
        """Sleep for 3 seconds then return the input message as a string."""
        await asyncio.sleep(3)
        return await super().generate_str(message, request_params)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """Sleep for 3 seconds then apply prompt like PassthroughLLM."""
        await asyncio.sleep(3)
        return await super()._apply_prompt_provider_specific(multipart_messages, request_params)
