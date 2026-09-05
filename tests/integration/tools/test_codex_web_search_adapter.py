"""Standalone search through the harness and a real local HTTP endpoint."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import CodexResponsesSettings, OpenAIWebSearchSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.tool_permission_handler import ToolPermissionResult
from fast_agent.tools.codex_web_search import SEARCH_SESSION_META, CodexWebSearchAdapter
from fast_agent.types import LlmStopReason

OUTPUT = "[Example](https://example.com) — unique search text"
SECRET = "private-token-and-query"


@asynccontextmanager
async def endpoint(
    status: int = 200, *, malformed: bool = False
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode()
            headers = dict(line.split(": ", 1) for line in head.split("\r\n")[1:] if line)
            length = next(
                value for key, value in headers.items() if key.lower() == "content-length"
            )
            body = json.loads(await reader.readexactly(int(length)))
            requests.append(body)
            code = 400 if body.get("settings", {}).get("filters") == {} else status
            payload = json.dumps(
                {"output": OUTPUT, "results": [{"title": "Example"}]}
                if code == 200
                else {"error": SECRET}
            ).encode()
            if malformed:
                payload = SECRET.encode()
            writer.write(
                f"HTTP/1.1 {code} Simulator\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    async with server:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/codex", requests


class GenerateSimulator(CodexResponsesLLM):
    """Simulate model turns, retaining the real standalone search provider path."""

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        assert [tool.name for tool in tools or []].count("web_run") == 1
        args = self._build_response_args([], self.get_request_params(request_params), tools)
        assert not any(tool["type"] == "web_search" for tool in args.get("tools", []))
        if not multipart_messages[-1].tool_results:
            return Prompt.assistant(
                "",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "search-1": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="web_run", arguments={"search_query": [{"q": "example"}]}
                        ),
                    )
                },
            )
        results = multipart_messages[-1].tool_results
        assert len(results) == 1
        result = results["search-1"]
        assert not result.is_error
        assert result.structured_content is None
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == OUTPUT
        converted = self._convert_to_provider_format(multipart_messages)
        outputs = [item for item in converted if item["type"] == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["output"] == OUTPUT
        return Prompt.assistant(outputs[0]["output"], stop_reason=LlmStopReason.END_TURN)


def agent_for(url: str, domains: list[str] | None = None) -> McpAgent:
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"https://api.openai.com/auth": {"chatgpt_account_id": "simulator"}}
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    context = Context(
        config=Settings(
            codexresponses=CodexResponsesSettings(
                api_key=f"header.{payload}.signature",
                base_url=url,
                web_search=OpenAIWebSearchSettings(enabled=True, allowed_domains=domains),
            )
        )
    )
    agent = McpAgent(AgentConfig(name="search", servers=[]), context=context)
    agent._llm = GenerateSimulator(model="gpt-6-astra", context=context)
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("domains", [None, ["example.com"]])
async def test_generate_search_defaults_and_text(domains: list[str] | None) -> None:
    async with endpoint() as (url, requests):
        agent = agent_for(url, domains)
        response = await agent.generate("search", RequestParams(max_iterations=3))
        assert response.first_text() == OUTPUT
        assert agent._shell_runtime is None
    assert len(requests) == 1
    settings = requests[0]["settings"]
    if domains is None:
        assert "filters" not in settings
    else:
        assert settings["filters"] == {"allowed_domains": domains}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403, 429, 503])
async def test_safe_http_failure(status: int) -> None:
    async with endpoint(status) as (url, requests):
        result = await agent_for(url).call_tool("web_run", {"search_query": [{"q": SECRET}]})
    assert len(requests) == 1
    assert result.is_error
    assert result.content[0].type == "text"
    text = result.content[0].text
    assert f"HTTP {status}" in text and "http" in text
    assert SECRET not in text and url not in text
    assert len(text) < 200
    assert result.meta and result.meta[SEARCH_SESSION_META]


@pytest.mark.asyncio
async def test_validation_feedback_is_bounded_and_repairable() -> None:
    async with endpoint() as (url, requests):
        adapter = CodexWebSearchAdapter(agent_for(url))
        result = await adapter.run(
            search_query=[{"q": SECRET, "recency": SECRET, SECRET: SECRET}] * 20
        )
        assert result.is_error
        assert result.content[0].type == "text"
        text = result.content[0].text
        assert "search_query.0.recency: int_type" in text
        assert "extra_forbidden" in text
        assert SECRET not in text
        assert len(text) <= 1200
        assert requests == []
        repaired = await adapter.run(search_query=[{"q": "example", "recency": 7}])
        assert not repaired.is_error
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_invalid_response_does_not_expose_body() -> None:
    async with endpoint(malformed=True) as (url, requests):
        result = await agent_for(url).call_tool("web_run", {"open": [{"ref_id": SECRET}]})
    assert len(requests) == 1
    assert result.is_error
    assert result.content[0].type == "text"
    assert "(response)" in result.content[0].text
    assert SECRET not in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("loader", ["transcript", "card"])
async def test_transcript_loader_restores_search_identity(
    tmp_path, compacted: bool, loader: str
) -> None:
    from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL
    from fast_agent.mcp.prompts.prompt_load import load_transcript_into_agent
    from fast_agent.session.session_manager import SessionManager

    manager = SessionManager(
        cwd=tmp_path, home_override=tmp_path / ".fast-agent", respect_env_override=False
    )
    session = manager.create_session()
    async with endpoint() as (url, requests):
        original = agent_for(url)
        assert original.context is not None
        original.context.session_manager = manager
        await original.generate("search", RequestParams(max_iterations=3))
        identity = requests[-1]["id"]
        history = original.message_history
        if compacted:
            history = [
                PromptMessageExtended(role="user", channels={FAST_AGENT_COMPACTION_CHANNEL: []})
            ]
        persisted = manager.get_session(session.info.name)
        assert persisted is not None
        manager.set_current_session(persisted)
        resumed = agent_for(url)
        assert resumed.context is not None
        resumed.context.session_manager = manager
        from fast_agent.mcp.prompt_serialization import save_messages

        history_path = tmp_path / "history.json"
        save_messages(history, str(history_path))
        if loader == "transcript":
            load_transcript_into_agent(resumed, history_path)
        else:
            from fast_agent.core.fastagent import FastAgent

            harness = FastAgent("restore-test", parse_cli_args=False, quiet=True)
            harness._agent_card_histories = {resumed.name: [history_path]}
            harness._apply_agent_card_histories({resumed.name: resumed})
        await resumed.call_tool("web_run", {"open": [{"ref_id": "turn0"}]})
        assert requests[-1]["id"] == identity
        resumed.clear()
        await resumed.call_tool("web_run", {"search_query": [{"q": "new"}]})
        assert requests[-1]["id"] != identity


@pytest.mark.asyncio
async def test_same_name_detached_clones_keep_invocation_identity(tmp_path) -> None:
    from fast_agent.session.session_manager import SessionManager

    manager = SessionManager(
        cwd=tmp_path, home_override=tmp_path / ".fast-agent", respect_env_override=False
    )
    session = manager.create_session()
    async with endpoint() as (url, requests):
        parent = agent_for(url)
        assert parent.context is not None
        parent.context.session_manager = manager
        await parent.call_tool("web_run", {"search_query": [{"q": "parent"}]})
        from fast_agent.mcp_server_registry import ServerRegistry

        parent.context.server_registry = ServerRegistry()
        metadata = dict(session.info.metadata)
        first = await parent.spawn_detached_instance(name="child[tool]")
        second = await parent.spawn_detached_instance(name="child[tool]")
        try:
            for clone in (first, second):
                clone._llm = GenerateSimulator(model="gpt-6-astra", context=parent.context)
                clone.clear()
                await clone.call_tool("web_run", {"search_query": [{"q": "child"}]})
            first_id, second_id = [request["id"] for request in requests[-2:]]
            assert first_id != second_id
            await first.call_tool("web_run", {"open": [{"ref_id": "turn0"}]})
            assert requests[-1]["id"] == first_id
            assert session.info.metadata == metadata
            first.clear()
            await first.call_tool("web_run", {"search_query": [{"q": "fresh"}]})
            assert requests[-1]["id"] not in (first_id, second_id)
        finally:
            await first.shutdown()
            await second.shutdown()


class PermissionSimulator:
    def __init__(self, decision: str) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str, str | None]] = []

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        self.calls.append((tool_name, server_name, tool_use_id))
        if self.decision == "failure":
            raise RuntimeError(SECRET)
        return ToolPermissionResult(
            allowed=self.decision == "allow", is_cancelled=self.decision == "cancel"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["allow", "deny", "cancel", "failure"])
async def test_web_permission_gate(decision: str) -> None:
    async with endpoint() as (url, requests):
        agent = agent_for(url)
        permissions = PermissionSimulator(decision)
        agent.aggregator.set_permission_handler(permissions)
        result = await agent.call_tool(
            "web_run", {"search_query": [{"q": "example"}]}, tool_use_id="permission-1"
        )
    assert permissions.calls == [("web_run", "web", "permission-1")]
    assert len(requests) == (1 if decision == "allow" else 0)
    assert bool(result.is_error) == (decision != "allow")
    assert SECRET not in str(result.content)
