"""Contract tests over real loopback HTTP, with no provider or harness integration."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import pytest
from mcp.types import Tool
from pydantic import JsonValue, ValidationError

from fast_agent.tools.web_search import (
    WEB_SEARCH_DESCRIPTION,
    SearchCommands,
    SearchQuery,
    SearchRequest,
    SearchSettings,
    WebSearchClient,
    WebSearchError,
    commands_schema,
)


@dataclass
class RecordedRequest:
    target: str
    headers: dict[str, str]
    body: dict[str, JsonValue]


@dataclass
class Simulator:
    status: int = 200
    body: bytes = b'{"output":"ok"}'
    requests: list[RecordedRequest] = field(default_factory=list)
    disconnect: bool = False

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            target, *lines = head.split("\r\n")
            headers = dict(line.lower().split(": ", 1) for line in lines if line)
            body = json.loads(await reader.readexactly(int(headers["content-length"])))
            assert isinstance(body, dict)
            self.requests.append(RecordedRequest(target, headers, body))
            if not self.disconnect:
                writer.write(
                    f"HTTP/1.1 {self.status} Simulator\r\n"
                    f"Content-Length: {len(self.body)}\r\n"
                    "Content-Type: application/json\r\n"
                    "Location: /redirected\r\n"
                    "Connection: close\r\n\r\n".encode()
                    + self.body
                )
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


@asynccontextmanager
async def serve(simulator: Simulator) -> AsyncIterator[str]:
    async with await asyncio.start_server(simulator.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/backend-api/codex"


@pytest.mark.asyncio
async def test_round_trip_and_borrowed_client() -> None:
    output = "\n[Source](https://example.com)\n\u2603\n"
    results: list[JsonValue] = [
        {"type": "future_result", "nested": {"keep": None}, "ref_id": "turn0search0"},
        None,
        [1, True],
    ]
    simulator = Simulator(
        body=json.dumps(
            {
                "output": output,
                "encrypted_output": "ciphertext",
                "results": results,
                "future": {"keep": True},
            }
        ).encode()
    )
    request = SearchRequest(
        id="stable-session",
        model="caller-model",
        commands=SearchCommands(search_query=[SearchQuery(q="hello", recency=0)]),
        input=[
            {"type": "message", "content": [{"type": "input_text", "text": "hi"}], "future": None}
        ],
        settings=SearchSettings(external_web_access=False),
        max_output_tokens=0,
    )
    async with serve(simulator) as url, httpx.AsyncClient() as http:
        async with WebSearchClient(
            base_url=url + "/", headers={"Authorization": "Bearer supplied"}, http_client=http
        ) as client:
            for _ in range(2):
                response = await client.search(request)
                assert response.output == output
                assert response.results == results
                assert response.encrypted_output == "ciphertext"
                assert response.model_extra == {"future": {"keep": True}}
        assert not http.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            await client.search(request)
        # The same HTTP client can be reused by another wrapper.
        async with WebSearchClient(base_url=url, http_client=http) as other:
            await other.search(request)
    assert len(simulator.requests) == 3
    recorded = simulator.requests[0]
    assert recorded.target == "POST /backend-api/codex/alpha/search HTTP/1.1"
    assert recorded.headers["authorization"] == "bearer supplied"
    assert recorded.body == {
        "id": "stable-session",
        "model": "caller-model",
        "commands": {"search_query": [{"q": "hello", "recency": 0}]},
        "input": [
            {"type": "message", "content": [{"type": "input_text", "text": "hi"}], "future": None}
        ],
        "settings": {"external_web_access": False},
        "max_output_tokens": 0,
    }
    assert all(item.body["id"] == "stable-session" for item in simulator.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("results", [None, [], [{"new": "variant"}]])
async def test_optional_results(results: list[JsonValue] | None) -> None:
    payload: dict[str, JsonValue] = {"output": ""}
    if results is not None:
        payload["results"] = results
    simulator = Simulator(body=json.dumps(payload).encode())
    async with serve(simulator) as url:
        async with WebSearchClient(base_url=url) as client:
            response = await client.search(SearchRequest(id="s", model="m", input="query"))
            assert response.results == results
            assert response.encrypted_output is None
        await client.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await client.search(SearchRequest(id="s", model="m"))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 429, 503])
async def test_bounded_http_errors_without_retry_or_redirect(status: int) -> None:
    simulator = Simulator(status=status, body=b"private server details" * 100_000)
    async with serve(simulator) as url, WebSearchClient(base_url=url) as client:
        with pytest.raises(WebSearchError) as caught:
            await client.search(SearchRequest(id="s", model="m"))
    assert caught.value.kind == "http"
    assert caught.value.status_code == status
    assert len(str(caught.value)) < 100
    assert "private" not in str(caught.value)
    assert len(simulator.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not json", b'{"output":42}', b'{"results":[]}'])
async def test_invalid_response(body: bytes) -> None:
    simulator = Simulator(body=body)
    async with serve(simulator) as url, WebSearchClient(base_url=url) as client:
        with pytest.raises(WebSearchError, match="response error"):
            await client.search(SearchRequest(id="s", model="m"))
    assert len(simulator.requests) == 1


@pytest.mark.asyncio
async def test_disconnect_is_not_retried() -> None:
    simulator = Simulator(disconnect=True)
    async with serve(simulator) as url, WebSearchClient(base_url=url) as client:
        with pytest.raises(WebSearchError, match="transport error"):
            await client.search(SearchRequest(id="s", model="m"))
    assert len(simulator.requests) == 1


@pytest.mark.asyncio
async def test_operations_and_schema() -> None:
    # A single representative batch exercises every operation at the public boundary.
    batch = {
        "search_query": [{"q": "news", "domains": ["example.com"]}],
        "image_query": [{"q": "birds"}],
        "open": [{"ref_id": "https://example.com", "lineno": 0}],
        "click": [{"ref_id": "page", "id": 0}],
        "find": [{"ref_id": "page", "pattern": "text"}],
        "screenshot": [{"ref_id": "pdf", "pageno": 0}],
        "finance": [{"ticker": "BTC", "type": "crypto", "market": ""}],
        "weather": [{"location": "US, CA, SF", "start": "2026-09-05", "duration": 7}],
        "sports": [{"tool": "sports", "fn": "schedule", "league": "nba", "num_games": 2}],
        "time": [{"utc_offset": "+03:00"}],
        "response_length": "short",
    }
    commands = SearchCommands.model_validate(batch)
    simulator = Simulator()
    async with serve(simulator) as url, WebSearchClient(base_url=url) as client:
        await client.search(SearchRequest(id="s", model="m", commands=commands))
    assert simulator.requests[0].body["commands"] == batch
    tool = Tool(
        name="web_search", description=WEB_SEARCH_DESCRIPTION, input_schema=commands_schema()
    )
    assert tool.input_schema["type"] == "object"
    assert "id" not in tool.input_schema["properties"]
    assert "settings" not in tool.input_schema["properties"]
    assert "Markdown" in WEB_SEARCH_DESCRIPTION
    tool.input_schema.clear()
    assert commands_schema()["type"] == "object"


@pytest.mark.parametrize("value", [-1, 2**64, True, 1.5, "1"])
def test_unsigned_integer_validation(value: object) -> None:
    with pytest.raises(ValidationError):
        SearchCommands.model_validate({"screenshot": [{"ref_id": "pdf", "pageno": value}]})


@pytest.mark.asyncio
async def test_settings_and_reasoning() -> None:
    payload = {
        "id": "s",
        "model": "m",
        "reasoning": {"effort": "model-defined", "summary": "none", "context": "current_turn"},
        "settings": {
            "user_location": {"type": "approximate", "country": "US", "city": "SF"},
            "search_context_size": "low",
            "filters": {"allowed_domains": [], "blocked_domains": ["example.com"]},
            "image_settings": {"max_results": 0, "caption": False},
            "allowed_callers": ["direct", "shell", "code_interpreter"],
            "external_web_access": "cached",
        },
    }
    request = SearchRequest.model_validate(payload)
    simulator = Simulator()
    async with serve(simulator) as url, WebSearchClient(base_url=url) as client:
        await client.search(request)
    assert simulator.requests[0].body == payload
    with pytest.raises(ValidationError):
        SearchCommands.model_validate({"typo": []})


def test_sports_discriminator_is_sent_when_omitted() -> None:
    commands = SearchCommands.model_validate({"sports": [{"fn": "standings", "league": "mlb"}]})
    assert commands.model_dump(exclude_none=True)["sports"][0]["tool"] == "sports"
