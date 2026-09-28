"""Live, credentialed structured-output smoke tests (not simulated endpoints)."""

from secrets import randbelow

import pytest
from pydantic import BaseModel

from fast_agent import FastAgent
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt import Prompt


class ArithmeticResult(BaseModel):
    answer: int


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [
        "responses.gpt-6-astra",
        "responses.gpt-6-sol",
        "responses.gpt-6-luna",
        "opus55",
        "sonnet55",
        "copilot.gpt-6-astra",
        "copilot.gpt-6-sol",
        "copilot.gpt-6-luna",
        "copilot.opus55",
        "copilot.sonnet55",
        "copilot.sonnet55?reasoning=off",
    ],
)
@pytest.mark.parametrize("with_tools", [False, True], ids=["schema-only", "with-tools"])
async def test_current_model_structured_output(
    fast_agent: FastAgent, model_name: str, with_tools: bool
) -> None:
    tool_calls: list[str] = []
    expected = randbelow(900_000) + 100_000 if with_tools else 4
    if with_tools:

        @fast_agent.tool
        def lookup_answer() -> int:
            """Read the current answer from the external source."""
            tool_calls.append("lookup_answer")
            return expected

    prompt = (
        "Call lookup_answer to obtain the current answer. Return that exact value; do not guess."
        if with_tools
        else "What is 2 + 2?"
    )

    @fast_agent.agent("chat", model=model_name, instruction="Answer simple arithmetic questions.")
    async def run() -> None:
        async with fast_agent.run() as agent:
            if with_tools:
                data, _ = await agent.chat.structured_schema(
                    [Prompt.user(prompt)],
                    schema=ArithmeticResult.model_json_schema(),
                    request_params=RequestParams(max_tokens=4096),
                )
                parsed = ArithmeticResult.model_validate(data)
            else:
                parsed, _ = await agent.chat.structured(
                    [Prompt.user(prompt)],
                    model=ArithmeticResult,
                    request_params=RequestParams(max_tokens=4096),
                )
            assert isinstance(parsed, ArithmeticResult)
            assert parsed.answer == expected
            if with_tools:
                assert tool_calls

    await run()
