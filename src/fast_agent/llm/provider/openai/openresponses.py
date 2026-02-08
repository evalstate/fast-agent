from __future__ import annotations

from typing import Any

from fast_agent.llm.provider.openai.openresponses_streaming import OpenResponsesStreamingMixin
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider


class OpenResponsesLLM(OpenResponsesStreamingMixin, ResponsesLLM):
    """LLM implementation for Open Responses-compatible APIs."""

    config_section: str | None = "openresponses"

    def __init__(self, provider: Provider = Provider.OPENRESPONSES, **kwargs: Any) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=provider, **kwargs)
