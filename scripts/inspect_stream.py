"""Utility to print raw streaming chunks from HuggingFace (OpenAI-compatible) endpoints.

Defaults mirror `HuggingFaceLLM` in fast-agent: base URL is the HF router
(`https://router.huggingface.co/v1`), and the default model is
`moonshotai/Kimi-K2-Thinking` with the provider suffix `:nebius`.

Environment overrides:
- MODEL: full model string (e.g., moonshotai/Kimi-K2-Thinking:novita). Leading
  "hf." is stripped for convenience.
- HF_DEFAULT_PROVIDER: provider suffix when MODEL is not set (default: nebius).
- OPENAI_BASE_URL: override base URL (default: router.huggingface.co/v1).
- OPENAI_API_KEY: HF token (required).
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI

# DEFAULT_BASE_MODEL = "moonshotai/Kimi-K2-Thinking"
# DEFAULT_PROVIDER = "together"

# DEFAULT_BASE_MODEL = "MiniMaxAI/MiniMax-M2"
# DEFAULT_PROVIDER = "novita"

# DEFAULT_BASE_MODEL = "zai-org/GLM-4.6"
# DEFAULT_PROVIDER = "zai-org"

DEFAULT_BASE_MODEL = "openai/gpt-oss-120b"
DEFAULT_PROVIDER = "groq"


DEFAULT_BASE_URL = "https://router.huggingface.co/v1"


def _resolve_model() -> str:
    env_model = os.environ.get("MODEL")
    if env_model:
        model = env_model
    else:
        provider = os.environ.get("HF_DEFAULT_PROVIDER") or DEFAULT_PROVIDER
        model = f"{DEFAULT_BASE_MODEL}:{provider}" if provider else DEFAULT_BASE_MODEL

    if model.startswith("hf."):
        model = model[len("hf.") :]
    return model


def _client() -> AsyncOpenAI:
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    return AsyncOpenAI(base_url=base_url)


async def main() -> None:
    client = _client()
    model = _resolve_model()

    tool = {
        "type": "function",
        "function": {
            "name": "whoami",
            "description": "Return who you are",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    async with await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Who are you? Call a tool if needed."}],
        stream=True,
        stream_options={"include_usage": True},
        tools=[tool],
    ) as stream:
        async for chunk in stream:
            try:
                payload: Any = chunk.model_dump()
            except Exception:
                payload = str(chunk)
            print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
