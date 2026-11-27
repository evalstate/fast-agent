"""OpenAI client for gateways that explicitly omit authorization."""

from typing import Any

from openai import AsyncOpenAI


class NoAuthAsyncOpenAI(AsyncOpenAI):
    """AsyncOpenAI subclass that permits requests with Authorization explicitly omitted."""

    def _validate_headers(
        self,
        headers: Any,
        custom_headers: Any,
    ) -> None:
        return None
