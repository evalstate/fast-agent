"""Check the fast-agent → google-genai serializer boundary without contacting Google."""

import json

import httpx2
import pytest
from google import genai
from mcp import Tool

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM


@pytest.mark.asyncio
async def test_generate_content_serializes_gemini4_request(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_API_VERSION", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_HTTP_CLIENT", raising=False)
    llm = GoogleNativeLLM(
        context=Context(config=Settings()),
        model="gemini-4-hydrogen",
        reasoning_effort="xhigh",
    )
    schema = {
        "$defs": {"answer": {"type": "string", "enum": ["ok"]}},
        "type": "object",
        "properties": {"answer": {"$ref": "#/$defs/answer"}},
        "additionalProperties": False,
    }
    tools = [Tool(name="answer", input_schema=schema)]
    config = llm._google_generate_content_config(
        llm.get_request_params(),
        tools=tools,
        available_tools=llm._google_available_tools(tools, suppress_tools=False),
        response_mime_type="application/json",
        response_schema=schema,
        suppress_tools=False,
    )

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1beta/models/gemini-4-hydrogen:generateContent"
        body = json.loads(request.content)
        generation = body["generationConfig"]
        assert generation["thinkingConfig"]["thinking_level"] == "XHIGH"
        assert "thinking_budget" not in generation["thinkingConfig"]
        assert not {"temperature", "topP", "topK", "candidateCount"}.intersection(generation)
        assert generation["responseJsonSchema"] == schema
        assert body["tools"][0]["functionDeclarations"][0]["parameters_json_schema"] == schema
        return httpx2.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    options = llm._google_http_options()
    assert options is not None
    options.async_client_args = {"transport": httpx2.MockTransport(respond)}
    async with genai.Client(api_key="test-key", http_options=options).aio as client:
        result = await client.models.generate_content(
            model="gemini-4-hydrogen", contents="test", config=config
        )
    assert result.text == "ok"
