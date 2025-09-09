import asyncio
from typing import Any, List

from mcp import Tool

from fast_agent.llm.fastagent_llm import (
    RequestParams,
)
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import PromptMessageExtended


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

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageExtended"],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """Sleep for 3 seconds then apply prompt like PassthroughLLM."""
        await asyncio.sleep(3)
        return await super()._apply_prompt_provider_specific(
            multipart_messages, request_params, tools, is_template
        )
