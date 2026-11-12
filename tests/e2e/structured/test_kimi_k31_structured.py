"""Live Kimi K3.1 structured output using normal Moonshot/HF credentials."""

import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from fast_agent import FastAgent
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import split_thinking_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.ptc import PTC_TOOL_NAME


class InventoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category: Literal["fruit", "vegetable"]
    quantity: int
    names: list[str]


class InventoryReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    location: str
    items: list[InventoryItem]


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["moonshot.kimi-k3.1-pro", "hf.moonshotai/Kimi-K3.1-Pro:together"],
)
@pytest.mark.parametrize("api", ["typed", "schema-with-tools"])
@pytest.mark.parametrize("ptc", [False, True])
async def test_kimi_k31_structured_output(
    fast_agent: FastAgent, model_name: str, api: str, ptc: bool
) -> None:
    tool_calls: list[str] = []

    @fast_agent.tool
    def lookup_inventory() -> str:
        """Look up the inventory for the orchard."""
        tool_calls.append("lookup_inventory")
        return "The orchard has 3 apples and 2 pears, both fruit."

    @fast_agent.agent(
        "chat",
        model=f"{model_name}?reasoning=low" + ("" if ptc else "&ptc=off"),
        instruction="Report the supplied inventory accurately using the requested JSON schema.",
        use_history=False,
    )
    async def run() -> None:
        async with fast_agent.run() as agent:
            chat = agent.chat
            tools = await chat.list_tools()
            expected_tool = PTC_TOOL_NAME if ptc else "lookup_inventory"
            assert any(tool.name == expected_tool for tool in tools.tools)
            assert chat.llm is not None
            params = RequestParams(max_tokens=4096, max_iterations=1, streaming_timeout=90)
            assert chat.llm.resolve_structured_tool_policy(params) == "no_tools"
            messages = [
                Prompt.user(
                    "If lookup_inventory is available, call it before answering. Otherwise use "
                    "this supplied inventory: location orchard; 3 apples and 2 pears, both fruit. "
                    "Return one item per fruit, with its name in the names array and its count "
                    "in quantity. List apples first, then pears."
                )
            ]
            if api == "typed":
                parsed, response = await chat.structured(
                    messages, model=InventoryReport, request_params=params
                )
            else:
                # Unlike typed structured(), this path normally exposes regular tools;
                # the model's default no_tools policy must suppress them for this turn.
                data, response = await chat.structured_schema(
                    messages, schema=InventoryReport.model_json_schema(), request_params=params
                )
                parsed = InventoryReport.model_validate(data)

            assert isinstance(parsed, InventoryReport)
            _, json_text = split_thinking_content(response.first_text())
            assert InventoryReport.model_validate_json(json_text) == parsed
            assert parsed.model_dump() == {
                "location": "orchard",
                "items": [
                    {"category": "fruit", "quantity": 3, "names": ["apples"]},
                    {"category": "fruit", "quantity": 2, "names": ["pears"]},
                ],
            }
            assert not tool_calls
            assert not response.tool_calls

    async with asyncio.timeout(180):
        await run()
