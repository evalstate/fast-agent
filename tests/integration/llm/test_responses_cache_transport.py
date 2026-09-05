"""Real SDK requests to a local Responses simulator. No provider credentials."""

import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from aiohttp import WSMsgType, web
from mcp_types import CallToolResult, ImageContent, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunner
from fast_agent.config import CodexResponsesSettings, OpenAISettings, Settings
from fast_agent.constants import FAST_AGENT_PENDING_MEDIA_ATTACHMENTS
from fast_agent.context import Context
from fast_agent.interfaces import AgentProtocol
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import from_json, to_json

pytestmark = [pytest.mark.integration, pytest.mark.simulated_endpoints, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def simulator(unused_tcp_port: int) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    def events(payload: dict[str, Any]) -> list[dict[str, Any]]:
        requests.append(payload)
        number = len(requests)
        response = {
            "id": f"resp_{number}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": payload["model"],
            # Deliberately report the baseline, NOT the effective effort.
            "reasoning": payload.get("reasoning"),
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{number}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        if number == 1 and any(
            block.get("text") == "Call tool"
            for item in payload["input"]
            for block in item.get("content", [])
        ):
            tool_call = {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "test_tool",
                "arguments": "{}",
                "status": "completed",
            }
            response["output"] = [tool_call]
            return [
                {"type": "response.created", "response": {**response, "output": []}},
                {"type": "response.output_item.added", "output_index": 0, "item": tool_call},
                {"type": "response.output_item.done", "output_index": 0, "item": tool_call},
                {"type": "response.completed", "response": response},
            ]
        return [
            {"type": "response.created", "response": {**response, "output": []}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**response["output"][0], "content": []},
            },
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "item_id": f"msg_{number}",
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "delta": "OK",
                "output_index": 0,
                "content_index": 0,
                "item_id": f"msg_{number}",
            },
            {"type": "response.output_item.done", "output_index": 0, "item": response["output"][0]},
            {"type": "response.completed", "response": response},
        ]

    async def sse(request: web.Request) -> web.Response:
        payload = await request.json()
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events(payload))
        return web.Response(text=body, content_type="text/event-stream")

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            if message.type == WSMsgType.TEXT:
                for event in events(json.loads(message.data)):
                    await socket.send_json(event)
        return socket

    app = web.Application()
    app.router.add_post("/v1/responses", sse)
    app.router.add_get("/v1/responses", websocket)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}/v1", requests
    finally:
        await runner.cleanup()


async def make_agent(
    url: str, *, codex: bool, transport: str, model: str = "gpt-6-astra"
) -> tuple[LlmAgent, ResponsesLLM]:
    claim = {"https://api.openai.com/auth": {"chatgpt_account_id": "local-test"}}
    token = "test." + base64.urlsafe_b64encode(json.dumps(claim).encode()).decode() + ".test"
    context = Context(
        config=Settings(
            responses=OpenAISettings(api_key="local-test", base_url=url),
            codexresponses=CodexResponsesSettings(api_key=token, base_url=url),
        )
    )
    agent = LlmAgent(AgentConfig(name="cache-test", model=model), context=context)

    def factory(agent: AgentProtocol, **kwargs: Any) -> ResponsesLLM:
        return (CodexResponsesLLM if codex else ResponsesLLM)(agent=agent, **kwargs)

    llm = await agent.attach_llm(
        factory,
        model=model,
        transport=transport,
        reasoning_effort="low",
    )
    assert isinstance(llm, ResponsesLLM)
    return agent, llm


def updates(payload: dict[str, Any]) -> list[str]:
    return [
        item["reasoning"]["effort"]
        for item in payload["input"]
        if item["type"] == "configuration_update"
    ]


@pytest.mark.parametrize("codex", [False, True])
@pytest.mark.parametrize("transport", ["sse", "websocket"])
async def test_effort_history_resume_and_clear(simulator, codex: bool, transport: str) -> None:
    url, requests = simulator
    agent, llm = await make_agent(url, codex=codex, transport=transport)
    try:
        for effort in ("low", "high", "high", "low"):
            llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value=effort))
            await agent.generate([Prompt.user(f"Work at {effort}")])
        key = requests[0]["prompt_cache_key"]
        assert all(request["prompt_cache_key"] == key for request in requests)
        assert all(request["reasoning"]["effort"] == "low" for request in requests)
        if transport == "sse":
            assert [updates(request) for request in requests] == [
                [],
                ["high"],
                ["high"],
                ["high", "low"],
            ]
        else:
            assert [updates(request) for request in requests] == [[], ["high"], [], ["low"]]
            assert all("previous_response_id" in request for request in requests[1:])
        if codex:
            assert all(request["reasoning"]["context"] == "all_turns" for request in requests)
        for request in requests:
            for index, item in enumerate(request["input"]):
                if item["type"] == "configuration_update":
                    assert request["input"][index + 1]["role"] == "user"
        saved = to_json(agent.message_history)
    finally:
        await llm.close()

    resumed, llm = await make_agent(url, codex=codex, transport=transport)
    try:
        resumed.load_message_history(from_json(saved))
        await resumed.generate([Prompt.user("Resume")])
        replay = requests[-1]
        assert replay["prompt_cache_key"] == key
        assert replay["reasoning"]["effort"] == "low"
        assert updates(replay) == ["high", "low"]
        # Full replay must retain the original positions, not move updates to the end.
        user_efforts = []
        effective = replay["reasoning"]["effort"]
        for item in replay["input"]:
            if item["type"] == "configuration_update":
                effective = item["reasoning"]["effort"]
            elif item.get("role") == "user":
                user_efforts.append(effective)
        assert user_efforts == ["low", "high", "high", "low", "low"]

        # Summary compaction replaces the prefix and retains a tail. A surviving
        # high-effort turn establishes a fresh baseline, without stale indices.
        history = from_json(saved)
        resumed.load_message_history([Prompt.user("Summary"), *history[4:]])
        llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="high"))
        await resumed.generate([Prompt.user("After compaction")])
        assert requests[-1]["reasoning"]["effort"] == "high"
        assert updates(requests[-1]) == ["low", "high"]

        resumed.clear(clear_prompts=True)
        await resumed.generate([Prompt.user("Fresh")])
        assert requests[-1]["reasoning"]["effort"] == "high"
        assert updates(requests[-1]) == []
        assert requests[-1]["prompt_cache_key"] != key
    finally:
        await llm.close()


@pytest.mark.parametrize("codex", [False, True])
async def test_unsupported_model_and_explicit_overrides(simulator, codex: bool) -> None:
    url, requests = simulator
    agent, llm = await make_agent(url, codex=codex, transport="sse", model="gpt-5.4")
    try:
        for effort in ("low", "high"):
            await agent.generate(
                [Prompt.user("Work")],
                RequestParams(
                    metadata={
                        "prompt_cache_key": "explicit-key",
                        "reasoning": {"effort": effort},
                    }
                ),
            )
        assert [request["reasoning"]["effort"] for request in requests] == ["low", "high"]
        assert all(not updates(request) for request in requests)
        assert all(request["prompt_cache_key"] == "explicit-key" for request in requests)
    finally:
        await llm.close()


async def test_astra_explicit_effort_override(simulator) -> None:
    url, requests = simulator
    agent, llm = await make_agent(url, codex=False, transport="sse")
    try:
        for effort in ("low", "high"):
            await agent.generate(
                [Prompt.user("Work")],
                RequestParams(
                    metadata={
                        "prompt_cache_key": "explicit-key",
                        "reasoning": {"effort": effort},
                    }
                ),
            )
        assert requests[-1]["reasoning"]["effort"] == "low"
        assert requests[-1]["prompt_cache_key"] == "explicit-key"
        assert updates(requests[-1]) == ["high"]
    finally:
        await llm.close()


@pytest.mark.parametrize("codex", [False, True])
@pytest.mark.parametrize("transport", ["sse", "websocket"])
async def test_latest_explicit_key_survives_next_turn_and_resume(
    simulator, codex: bool, transport: str
) -> None:
    url, requests = simulator
    agent, llm = await make_agent(url, codex=codex, transport=transport)
    try:
        for key, effort in (("key-A", "low"), ("key-B", "high"), (None, "high")):
            llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value=effort))
            await agent.generate(
                [Prompt.user("Work")],
                RequestParams(metadata={"prompt_cache_key": key} if key else {}),
            )
        assert [request["prompt_cache_key"] for request in requests] == ["key-A", "key-B", "key-B"]
        saved = to_json(agent.message_history)
    finally:
        await llm.close()

    resumed, llm = await make_agent(url, codex=codex, transport=transport)
    try:
        resumed.load_message_history(from_json(saved))
        llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="high"))
        await resumed.generate([Prompt.user("Resume")])
        assert requests[-1]["prompt_cache_key"] == "key-B"
        assert all(request["reasoning"]["effort"] == "low" for request in requests)
        assert updates(requests[-1]) == ["high"]
    finally:
        await llm.close()


@pytest.mark.parametrize("codex", [False, True])
@pytest.mark.parametrize("transport", ["sse", "websocket"])
@pytest.mark.parametrize(
    ("model", "override"),
    [("gpt-5.4", "gpt-6-astra"), ("gpt-6-astra", "gpt-5.4")],
)
async def test_extra_body_cannot_cross_astra_boundary(
    simulator, codex: bool, transport: str, model: str, override: str
) -> None:
    url, requests = simulator
    agent, llm = await make_agent(url, codex=codex, transport=transport, model=model)
    try:
        with pytest.raises(ValueError, match="directly in metadata"):
            await agent.generate(
                [Prompt.user("Work")],
                RequestParams(metadata={"extra_body": {"model": override, "truncation": "auto"}}),
            )
        assert requests == []
    finally:
        await llm.close()


async def test_non_astra_extra_body_model_override_is_unchanged(simulator) -> None:
    url, requests = simulator
    agent, llm = await make_agent(url, codex=False, transport="sse", model="gpt-5.4")
    try:
        await agent.generate(
            [Prompt.user("Work")],
            RequestParams(metadata={"extra_body": {"model": "gpt-5.2"}}),
        )
        assert requests[-1]["model"] == "gpt-5.2"
        assert not updates(requests[-1])
    finally:
        await llm.close()


@pytest.mark.parametrize("codex", [False, True])
@pytest.mark.parametrize("transport", ["sse", "websocket"])
@pytest.mark.parametrize("pending_media", [False, True])
async def test_tool_error_and_staged_media_wait_for_genuine_turn(
    simulator, codex: bool, transport: str, pending_media: bool
) -> None:
    url, requests = simulator
    _, llm = await make_agent(url, codex=codex, transport=transport)
    history = [Prompt.user("Call tool")]
    try:
        response = await llm._apply_prompt_provider_specific(history)
        assert response.tool_calls and "call_test" in response.tool_calls
        history.append(response)
        runner = ToolRunner(agent=ToolAgent(AgentConfig("staging"), []), messages=[])
        runner._stage_tool_response(
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Tool execution failed")],
                tool_results={
                    "call_test": CallToolResult(
                        content=[TextContent(type="text", text="Error")], is_error=True
                    )
                },
                channels={
                    FAST_AGENT_PENDING_MEDIA_ATTACHMENTS: [
                        ImageContent(type="image", data="abcd", mime_type="image/png")
                    ]
                }
                if pending_media
                else None,
            )
        )
        history.extend(runner.delta_messages)
        history = from_json(to_json(history))
        llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="high"))
        history.append(await llm._apply_prompt_provider_specific(history))
        assert not updates(requests[-1])
        history.append(Prompt.user("Next task"))
        await llm._apply_prompt_provider_specific(history)
        assert updates(requests[-1]) == ["high"]
        assert all(request["reasoning"]["effort"] == "low" for request in requests)
        items = requests[-1]["input"]
        index = next(i for i, item in enumerate(items) if item["type"] == "configuration_update")
        assert items[index + 1]["content"] == [{"type": "input_text", "text": "Next task"}]
    finally:
        await llm.close()
