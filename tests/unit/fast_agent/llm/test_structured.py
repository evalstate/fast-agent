from typing import List, Literal

import pytest
from pydantic import BaseModel

from fast_agent.llm.internal.passthrough import PassthroughLLM
from mcp_agent.core.prompt import Prompt


# Example model similar to what's used in the Router workflow
class StructuredResponseCategory(BaseModel):
    category: str
    confidence: Literal["high", "medium", "low"]
    reasoning: str | None


class StructuredResponse(BaseModel):
    categories: List[StructuredResponseCategory]


@pytest.mark.asyncio
async def test_direct_pydantic_conversion():
    # JSON string that would typically come from an LLM
    json_str = """
    {
        "categories": [
            {
                "category": "tech_support",
                "confidence": "high",
                "reasoning": "Query relates to system troubleshooting"
            },
            {
                "category": "documentation",
                "confidence": "medium",
                "reasoning": null
            }
        ]
    }
    """

    # Create PassthroughLLM instance and use it to process the JSON
    llm = PassthroughLLM(name="structured")
    result, _ = await llm.structured([Prompt.user(json_str)], model=StructuredResponse)

    # Verify conversion worked correctly
    assert isinstance(result, StructuredResponse)
    assert len(result.categories) == 2
    assert result.categories[0].category == "tech_support"
    assert result.categories[0].confidence == "high"
    assert result.categories[1].category == "documentation"
    assert result.categories[1].confidence == "medium"
    assert result.categories[1].reasoning is None


@pytest.mark.asyncio
async def test_structured_with_bad_json():
    # JSON string that would typically come from an LLM
    json_str = """
    {
        "categories": [
            {
                "category": "tech_support",
            },
            {
                "category": "documentation",
                "confidence": "medium",
                "reaso: null
            }
        ]
    }
    """

    # Create PassthroughLLM instance and use it to process the JSON
    llm = PassthroughLLM(name="structured")
    result, _ = await llm.structured([Prompt.user(json_str)], model=StructuredResponse)

    assert None is result


@pytest.mark.asyncio
async def test_chat_turn_counting():
    # Create PassthroughLLM instance and use it to process the JSON
    llm = PassthroughLLM()
    # no messages yet, so chat turn should be 1
    assert 1 == llm.chat_turn()
    await llm.generate([Prompt.user("test")])
    assert 2 == llm.chat_turn()

    # just increment for each assistant message
    await llm.generate([Prompt.user("foo")])
    await llm.generate([Prompt.user("bar")])

    assert 4 == llm.chat_turn()
