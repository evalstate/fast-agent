from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import Tool

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.tools.codex_web_search import SEARCH_SESSION_META, search_session_id
from fast_agent.tools.web_search import SearchCommands, SearchResponse, WebSearchError


class SearchSimulator(CodexResponsesLLM):
    def __init__(self, model: str = "gpt-6-astra") -> None:
        super().__init__(model=model, web_search=True)
        self.search_models: list[str | None] = []
        self.search_ids: list[str] = []
        self.fail = False

    async def run_standalone_web_search(
        self, session_id: str, arguments: dict[str, Any], *, model: str | None = None
    ) -> SearchResponse:
        SearchCommands.model_validate(arguments)
        self.search_ids.append(session_id)
        self.search_models.append(model)
        if self.fail:
            raise WebSearchError("http", 503)
        return SearchResponse(output="[Example](https://example.com)", results=[])


def agent_with(llm: CodexResponsesLLM | ResponsesLLM) -> McpAgent:
    agent = McpAgent(AgentConfig(name="search-test", servers=[]), context=Context())
    agent._llm = llm
    return agent


@pytest.mark.asyncio
async def test_dynamic_availability_without_shell() -> None:
    llm = SearchSimulator()
    agent = agent_with(llm)
    assert "web_run" in {tool.name for tool in (await agent.list_tools()).tools}
    assert agent._shell_runtime is None
    # A clone must bind search to its own provider and history, not the parent's.
    assert not agent._clone_constructor_kwargs().get("tools")
    llm.set_web_search_enabled(False)
    assert "web_run" not in {tool.name for tool in (await agent.list_tools()).tools}
    llm.set_web_search_enabled(True)
    assert "web_run" in {tool.name for tool in (await agent.list_tools()).tools}
    agent._llm = ResponsesLLM(model="gpt-6-astra", web_search=True)
    assert "web_run" not in {tool.name for tool in (await agent.list_tools()).tools}
    agent._llm = CodexResponsesLLM(model="gpt-5.3-codex", web_search=True)
    assert "web_run" not in {tool.name for tool in (await agent.list_tools()).tools}


@pytest.mark.asyncio
async def test_execution_and_history_resume() -> None:
    llm = SearchSimulator()
    agent = agent_with(llm)
    await agent.list_tools()
    result = await agent.call_tool("web_run", {"search_query": [{"q": "example"}]})
    assert not result.is_error
    assert result.content[0].type == "text"
    assert result.content[0].text == "[Example](https://example.com)"
    history = PromptMessageExtended(role="user", tool_results={"call-1": result})
    restored = PromptMessageExtended.model_validate_json(history.model_dump_json())
    resumed = agent_with(llm)
    resumed.load_message_history([restored])
    await resumed.list_tools()
    next_result = await resumed.call_tool("web_run", {"open": [{"ref_id": "turn0search0"}]})
    assert not next_result.is_error
    assert llm.search_ids[0] == llm.search_ids[1]
    assert search_session_id([restored], "new-session") == llm.search_ids[0]
    assert result.meta and result.meta[SEARCH_SESSION_META] == llm.search_ids[0]


@pytest.mark.asyncio
async def test_failures_and_stale_calls_are_recoverable() -> None:
    llm = SearchSimulator()
    agent = agent_with(llm)
    await agent.list_tools()
    assert (await agent.call_tool("web_run", {"bogus": True})).is_error
    llm.fail = True
    assert (await agent.call_tool("web_run", {"search_query": [{"q": "example"}]})).is_error
    llm.set_web_search_enabled(False)
    assert (await agent.call_tool("web_run", {})).is_error


def test_lite_suppresses_hosted_search_only() -> None:
    llm = SearchSimulator()
    args = llm._build_response_args([], RequestParams(model="gpt-6-astra"), None)
    assert "web_search_call.action.sources" not in args["include"]
    assert all(tool["type"] != "web_search" for tool in args["input"][0]["tools"])
    args = llm._build_response_args([], RequestParams(model="gpt-5.3-codex"), None)
    assert any(tool["type"] == "web_search" for tool in args["tools"])


@pytest.mark.asyncio
async def test_provider_adapter_against_http_simulator() -> None:
    import asyncio
    import base64
    import json

    from fast_agent.config import CodexResponsesSettings, OpenAIWebSearchSettings, Settings

    received: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        header = (await reader.readuntil(b"\r\n\r\n")).decode()
        lines = header.split("\r\n")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value
        body = await reader.readexactly(int(headers["content-length"]))
        received.append((lines[0], headers, json.loads(body)))
        payload = b'{"output":"[Example](https://example.com)","results":[]}'
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
            + payload
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    token_payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"https://api.openai.com/auth": {"chatgpt_account_id": "test-account"}}
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    token = f"header.{token_payload}.signature"
    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        llm = CodexResponsesLLM(
            model="gpt-5.3-codex",
            context=Context(
                config=Settings(
                    codexresponses=CodexResponsesSettings(
                        api_key=token,
                        base_url=f"http://127.0.0.1:{port}/codex",
                        web_search=OpenAIWebSearchSettings(
                            enabled=True,
                            allowed_domains=["example.com"],
                            tool_type="web_search_preview",
                            external_web_access=False,
                        ),
                    )
                )
            ),
        )
        result = await llm.run_standalone_web_search(
            "persisted-search", {"search_query": [{"q": "example"}]}, model="gpt-6-astra"
        )
    assert result.output == "[Example](https://example.com)"
    path, headers, body = received[0]
    assert path == "POST /codex/alpha/search HTTP/1.1"
    assert headers["authorization"] == f"Bearer {token}"
    assert headers["chatgpt-account-id"] == "test-account"
    assert body["id"] == "persisted-search"
    assert body["model"] == "gpt-6-astra"
    assert body["settings"]["filters"]["allowed_domains"] == ["example.com"]
    assert body["settings"]["external_web_access"] is False
    assert "Authorization" not in body


@pytest.mark.asyncio
async def test_open_text_survives_provider_conversion_and_direct_calls() -> None:
    from fast_agent.mcp.helpers.content_helpers import canonicalize_tool_result_content_for_llm

    llm = SearchSimulator()
    agent = agent_with(llm)
    result = await agent.call_tool("web_run", {"open": [{"ref_id": "https://example.com"}]})
    assert not result.is_error
    assert result.structured_content is None
    assert result.meta and result.meta["results"] == []
    assert canonicalize_tool_result_content_for_llm(result) == result.content
    converted = llm._convert_to_provider_format(
        [PromptMessageExtended(role="user", tool_results={"open-1": result})]
    )
    assert converted[0]["output"] == "[Example](https://example.com)"
    await agent.call_tool("web_run", {"find": [{"ref_id": "turn0", "pattern": "Example"}]})
    assert llm.search_ids[0] == llm.search_ids[1]
    from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL

    agent.load_message_history(
        [PromptMessageExtended(role="user", channels={FAST_AGENT_COMPACTION_CHANNEL: []})]
    )
    await agent.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    assert llm.search_ids[2] == llm.search_ids[0]
    agent.load_message_history([])
    await agent.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    assert llm.search_ids[3] != llm.search_ids[0]


class TurnSearchSimulator(SearchSimulator):
    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list["Tool"] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        from mcp.types import CallToolRequest, CallToolRequestParams

        from fast_agent.mcp.prompt import Prompt
        from fast_agent.types import LlmStopReason

        params = self.get_request_params(request_params)
        names = [tool.name for tool in tools or []]
        lite = self.standalone_web_search_enabled(params.model)
        assert names.count("web_run") == int(lite)
        args = self._build_response_args([], params, tools)
        hosted = [tool for tool in args.get("tools", []) if tool["type"] == "web_search"]
        assert bool(hosted) is not lite
        if lite and not multipart_messages[-1].tool_results:
            return Prompt.assistant(
                "search",
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
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


@pytest.mark.asyncio
@pytest.mark.parametrize("default", ["gpt-5.3-codex", "gpt-6-astra"])
async def test_turn_model_overrides_registration_and_dispatch(default: str) -> None:
    llm = TurnSearchSimulator(default)
    agent = agent_with(llm)
    for model in ["gpt-6-astra", "gpt-5.3-codex", "gpt-6-astra"]:
        response = await agent.generate(
            "search", RequestParams(model=model, use_history=False, max_iterations=3)
        )
        assert response.first_text() == "done"
    assert llm.search_models == ["gpt-6-astra", "gpt-6-astra"]
    assert ("web_run" in {tool.name for tool in (await agent.list_tools()).tools}) == (
        default == "gpt-6-astra"
    )


@pytest.mark.asyncio
async def test_search_identity_survives_compaction_and_session_reload(tmp_path: "Path") -> None:
    from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL
    from fast_agent.session.session_manager import SessionManager

    manager = SessionManager(
        cwd=tmp_path, home_override=tmp_path / ".fast-agent", respect_env_override=False
    )
    session = manager.create_session()
    llm = SearchSimulator()
    agent = agent_with(llm)
    assert agent.context is not None
    agent.context.session_manager = manager
    await agent.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    identity = llm.search_ids[-1]
    compacted = [
        PromptMessageExtended(role="user", content=[], channels={FAST_AGENT_COMPACTION_CHANNEL: []})
    ]
    agent.load_message_history(compacted)
    persisted = manager.get_session(session.info.name)
    assert persisted is not None
    assert identity in persisted.info.metadata.values()
    manager.set_current_session(persisted)
    resumed = agent_with(llm)
    assert resumed.context is not None
    resumed.context.session_manager = manager
    resumed.load_message_history(compacted)
    await resumed.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    assert llm.search_ids[-1] == identity
    manager.create_session()
    await resumed.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    assert llm.search_ids[-1] != identity
    manager.set_current_session(session)
    await resumed.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    assert llm.search_ids[-1] == identity
    resumed.clear()
    await resumed.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    assert llm.search_ids[-1] != identity


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", ["transcript", "agent_card", "default"])
async def test_history_loader_resets_state_without_rotating_session_identity(
    tmp_path: "Path", loader: str
) -> None:
    from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL
    from fast_agent.core.fastagent import FastAgent
    from fast_agent.llm.provider_types import Provider
    from fast_agent.llm.usage_tracking import (
        CompletionTokenUsage,
        PromptTokenUsage,
        TurnUsage,
        UsageSchema,
    )
    from fast_agent.mcp.prompt_serialization import save_messages
    from fast_agent.mcp.prompts.prompt_load import load_transcript_into_agent
    from fast_agent.session.session_manager import SessionManager

    manager = SessionManager(
        cwd=tmp_path, home_override=tmp_path / ".fast-agent", respect_env_override=False
    )
    session = manager.create_session()
    llm = SearchSimulator()
    agent = agent_with(llm)
    assert agent.context is not None
    agent.context.session_manager = manager
    await agent.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    identity = llm.search_ids[-1]
    assert llm.model_name is not None
    turn = TurnUsage(
        provider=Provider.CODEX_RESPONSES,
        model=llm.model_name,
        usage_schema=UsageSchema.OPENAI_RESPONSES,
        prompt=PromptTokenUsage(total=10),
        completion=CompletionTokenUsage(total=5),
    )
    llm.usage_accumulator.add_turn(turn)
    agent.subagent_usage_accumulator.add_turn(turn)
    llm._seen_tool_call_ids.add("old-call")
    messages = [PromptMessageExtended(role="user", channels={FAST_AGENT_COMPACTION_CHANNEL: []})]

    if loader == "transcript":
        load_transcript_into_agent(agent, messages)
    elif loader == "agent_card":
        history_path = tmp_path / "history.json"
        save_messages(messages, str(history_path))
        fast = FastAgent("loader-test", parse_cli_args=False, quiet=True)
        fast._agent_card_histories[agent.name] = [history_path]
        fast._apply_agent_card_histories({agent.name: agent})
    else:
        agent.load_message_history(messages)

    assert bool(llm._seen_tool_call_ids) == (loader == "default")
    assert bool(agent.subagent_usage_accumulator.turns) == (loader == "default")
    assert bool(llm.usage_accumulator.turns) == (loader != "agent_card")
    assert agent.message_history == messages
    persisted = manager.get_session(session.info.name)
    assert persisted is not None
    assert identity in persisted.info.metadata.values()
    await agent.call_tool("web_run", {"time": [{"utc_offset": "+00:00"}]})
    assert llm.search_ids[-1] == identity
