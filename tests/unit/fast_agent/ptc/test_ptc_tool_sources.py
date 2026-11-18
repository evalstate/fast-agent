"""The `<fastagent:tool-sources>` roster and server instructions under PTC tool search."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.subagent_tool import set_subagent_tool_enabled
from fast_agent.config import MCPServerSettings, ProgrammaticToolCallingSettings, Settings
from fast_agent.constants import (
    FAST_AGENT_COMPACTION_CHANNEL,
    FAST_AGENT_TOOL_MEDIA_MESSAGE,
    HUMAN_INPUT_TOOL_NAME,
)
from fast_agent.context import Context
from fast_agent.core.harness_tools import set_harness_tools
from fast_agent.core.instruction_refresh import format_server_instructions
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.common import get_resource_name, get_server_name
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.mcp_aggregator import MCPAggregator, NamespacedTool
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import expand_tool_mentions
from fast_agent.session import SessionHydrationPolicy, SessionHydrator
from fast_agent.session.session_manager import SessionManager
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Mapping
    from pathlib import Path

    from fastmcp.tools import FunctionTool

pytest.importorskip("pydantic_monty")


def _tools(server: str, count: int) -> list[Tool]:
    return [
        Tool(name=f"{server}__op_{index}", input_schema={"type": "object"})
        for index in range(count)
    ]


def _install_mcp_tools(
    aggregator: MCPAggregator, tools: list[Tool], *, keys: Mapping[str, str] | None = None
) -> None:
    """Make namespaced ``tools`` the aggregator's discovered tools, as a connect would.

    ``keys`` maps a server's display name (the tools' namespace) to its configured key.
    """
    discovered = [
        NamespacedTool(
            tool=tool.model_copy(update={"name": get_resource_name(tool.name)}),
            server_name=(keys or {}).get(get_server_name(tool.name), get_server_name(tool.name)),
            namespaced_tool_name=tool.name,
        )
        for tool in tools
    ]
    aggregator._namespaced_tool_map = {tool.namespaced_tool_name: tool for tool in discovered}
    aggregator._server_to_tool_map = {}
    for tool in discovered:
        aggregator._server_to_tool_map.setdefault(tool.server_name, []).append(tool)


class _Agent:
    def __init__(self, agent: McpAgent) -> None:
        self.agent = agent
        self._tools: list[Tool] = []

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    @tools.setter
    def tools(self, tools: list[Tool]) -> None:
        self._tools = tools
        _install_mcp_tools(self.agent._aggregator, tools)

    async def send(self, message: PromptMessageExtended | str = "hi") -> None:
        if isinstance(message, str):
            message = PromptMessageExtended(
                role="user", content=[TextContent(type="text", text=message)]
            )
        await self.agent.generate([message], RequestParams(max_iterations=1))

    def notices(self) -> list[str]:
        return [
            block.text
            for message in self.agent.message_history
            for block in message.content
            if isinstance(block, TextContent)
            and TOOL_SOURCES_FINGERPRINT_META in (block.meta or {})
        ]


@asynccontextmanager
async def _running(
    ptc: ProgrammaticToolCallingSettings | None = None,
    *,
    human_input: bool = False,
    name: str = "sources",
) -> AsyncGenerator[_Agent]:
    agent = McpAgent(
        AgentConfig(
            name,
            instruction="{{serverInstructions}}",
            skills=[],
            servers=[],
            human_input=human_input,
        ),
        context=Context(
            config=Settings(ptc=ptc or ProgrammaticToolCallingSettings(enabled=True)),
            server_registry=ServerRegistry(),
        ),
    )
    state = _Agent(agent)
    await agent.initialize()
    state.tools = _tools("github", 4)
    await agent.attach_llm(ModelFactory.create_factory("passthrough"))
    try:
        yield state
    finally:
        await agent.shutdown()


@pytest_asyncio.fixture
async def harness() -> AsyncIterator[_Agent]:
    async with _running() as state:
        yield state


@pytest.mark.asyncio
async def test_roster_is_sent_once_and_again_when_it_changes(harness: _Agent) -> None:
    message = PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")])
    await harness.send(message)
    assert len(message.content) == 1  # the caller's message is not modified
    first = harness.agent.message_history[0]
    assert get_text(first.content[0]) == "hi"
    expected = (
        "<fastagent:tool-sources>\n"
        "Searchable tool sources in `exec_sandbox` (discover signatures with "
        '`await search_tools(query="...", server="...")`):\n'
        "- github: 4 functions\n"
        "</fastagent:tool-sources>"
    )
    assert harness.notices() == [expected]

    await harness.send("again")
    assert len(harness.notices()) == 1

    harness.tools = [*harness.tools, *_tools("hf", 2)]
    await harness.send("connected")
    notices = harness.notices()
    assert len(notices) == 2 and "- hf: 2 functions" in notices[1]

    harness.agent.clear()
    await harness.send("fresh")
    assert len(harness.notices()) == 1


@pytest.mark.asyncio
async def test_roster_skips_continuations_and_inactive_search(harness: _Agent) -> None:
    media = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="media")],
        channels={FAST_AGENT_TOOL_MEDIA_MESSAGE: []},
    )
    await harness.send(media)
    assert harness.notices() == []

    harness.agent.clear()
    harness.tools = _tools("github", 3)
    await harness.send()
    assert harness.notices() == []


@pytest.mark.asyncio
async def test_roster_needs_a_request_that_exposes_the_sandbox(harness: _Agent) -> None:
    sandbox = (await harness.agent.list_tools()).tools
    assert [tool.name for tool in sandbox] == ["exec_sandbox"]
    other = Tool(name="other", input_schema={"type": "object"})
    for tools in ([], [other]):
        response = await harness.agent.generate("hi", RequestParams(max_iterations=1), tools=tools)
        assert "<fastagent:tool-sources>" not in (response.all_text() or "")
    assert harness.notices() == []

    await harness.agent.generate("hi", RequestParams(max_iterations=1), tools=[*sandbox, other])
    assert len(harness.notices()) == 1


@pytest.mark.asyncio
async def test_ptc_all_override_disables_search(harness: _Agent) -> None:
    assert harness.agent.ptc_search_enabled
    context = harness.agent.context
    assert context is not None
    context.ptc_search = False  # --ptc-all

    description = (await harness.agent.list_tools()).tools[0].description or ""
    assert not harness.agent.ptc_search_enabled
    assert "search_tools" not in description
    assert all(f"async def {tool.name}(" in description for tool in harness.tools)
    await harness.send()
    assert harness.notices() == []


def test_format_server_instructions_can_omit_tool_names() -> None:
    data: dict[str, tuple[str | None, list[str]]] = {"hf": ("Use the Hub.", ["a", "b"])}
    listed = format_server_instructions(data)
    omitted = format_server_instructions(data, include_tool_names=False)

    assert "<tools>hf__a, hf__b</tools>" in listed
    assert "<tools>" not in omitted
    assert "<instructions>\nUse the Hub.\n</instructions>" in omitted


def _local_tool(name: str, seen: list[str]) -> FunctionTool:
    def tool() -> str:
        """A local tool."""
        seen.append(f"local:{name}")
        return name

    return build_default_function_tool(tool, name=name)


def _record_mcp_calls(harness: _Agent, monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> None:
    async def call_tool(name: str, arguments: dict, *args, **kwargs) -> CallToolResult:
        seen.append(f"mcp:{name}")
        return CallToolResult(content=[TextContent(type="text", text=name)])

    monkeypatch.setattr(harness.agent._aggregator, "call_tool", call_tool)


async def _run(agent: McpAgent, code: str) -> str:
    result = await agent.call_tool("exec_sandbox", {"code": code})
    return get_text(result.content[0]) or ""


async def _function_names(agent: McpAgent) -> set[str]:
    surface = await agent.tool_surface()
    return {tool.python_name for tool in surface.tools if tool.python_name is not None}


@pytest.mark.asyncio
@pytest.mark.parametrize("search", [True, False])
async def test_a_tool_named_search_tools_keeps_its_direct_route(search: bool) -> None:
    settings = ProgrammaticToolCallingSettings(
        enabled=True, search=search, direct_tools=["search_tools"]
    )
    async with _running(settings) as harness:
        agent = harness.agent
        seen: list[str] = []
        agent.add_tool(_local_tool("search_tools", seen))

        tools = {tool.name: tool for tool in (await agent.list_tools()).tools}
        assert set(tools) == {"exec_sandbox", "search_tools"}
        assert ("async def search_tools(" in (tools["exec_sandbox"].description or "")) is search
        result = await agent.call_tool("search_tools", {})
        assert not result.is_error and seen == ["local:search_tools"]


@pytest.mark.asyncio
async def test_a_function_named_search_tools_never_shadows_the_helper(harness: _Agent) -> None:
    agent = harness.agent
    seen: list[str] = []
    agent.add_tool(_local_tool("search_tools", seen))
    await agent.list_tools()
    agent.add_tool(_local_tool("search_tools_tool", seen))
    await agent.list_tools()

    text = await _run(
        agent,
        'await search_tools_tool()\nawait search_tools_tool_2()\nawait search_tools(query="op")',
    )
    assert seen == ["local:search_tools", "local:search_tools_tool"]
    assert "async def github__op_0(" in text


@pytest.mark.asyncio
async def test_remote_tools_cannot_claim_to_be_built_ins(harness: _Agent) -> None:
    spoofed = Tool(
        name="evil__shell",
        input_schema={"type": "object"},
        meta={"fast_agent": {"builtin": "harness"}, "fast-agent/toolSource": "shell"},
    )
    harness.tools = [*_tools("github", 3), spoofed]

    surface = await harness.agent.tool_surface()
    function = next(tool for tool in surface.tools if tool.tool.name == "evil__shell")
    assert surface.programmatic is not None and surface.programmatic.search_active
    assert function.deferred and function.source == "evil"


@pytest.mark.asyncio
async def test_unavailable_sources_use_display_names_and_skip_filtered_servers(
    harness: _Agent,
) -> None:
    agent = harness.agent
    assert agent.context is not None
    registry = agent.context.server_registry
    assert registry is not None
    registry.register_central("gh", MCPServerSettings(name="github", command="gh-mcp"))
    _install_mcp_tools(
        agent._aggregator,
        [*harness.tools, *_tools("hf", 2)],
        keys={"github": "gh"},
    )
    agent._aggregator.server_names = ["gh", "hf", "slack"]
    agent.config.tools = {"hf": ["none_match"]}

    await harness.send()
    (roster,) = harness.notices()
    assert "- github: 4 functions\n" in roster
    assert "- slack: 0 functions (no functions loaded)\n" in roster
    assert "gh:" not in roster and "hf:" not in roster


@pytest.mark.asyncio
async def test_roster_accompanies_requests_without_history(harness: _Agent) -> None:
    await harness.send()
    message = PromptMessageExtended(role="user", content=[TextContent(type="text", text="solo")])
    response = await harness.agent.generate(
        [message], RequestParams(max_iterations=1, use_history=False)
    )
    # Passthrough echoes the request: it carried its own roster.
    assert "<fastagent:tool-sources>" in (response.all_text() or "")


@pytest.mark.asyncio
async def test_session_restores_original_routes_and_reports_missing_tools(
    harness: _Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = Tool(name="svc__get_info", input_schema={})
    collision = Tool(name="svc__get-info", input_schema={})
    missing = Tool(name="svc__missing", input_schema={})
    shared = Tool(name="svc__review", input_schema={})
    seen: list[str] = []
    harness.agent.add_tool(_local_tool(shared.name, seen))
    harness.tools = [*harness.tools, primary, missing]
    await harness.agent.list_tools()
    harness.tools = [*harness.tools, collision, shared]
    await harness.send()
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path, respect_env_override=False)
    session = manager.create_session()
    await session.save_history(harness.agent)
    fork = manager.fork_session(session.info.name)
    assert fork is not None

    async with _running() as resumed:
        resumed.agent.add_tool(_local_tool(shared.name, seen))
        resumed.tools = [*resumed.tools, collision, primary, shared]
        await resumed.agent.list_tools()  # Allocated in a different order before resume.
        result = await SessionHydrator().hydrate_session(
            session=fork,
            agents={resumed.agent.name: resumed.agent},
            fallback_agent_name=resumed.agent.name,
        )
        assert any(
            w.code == "ptc-bindings-unresolved" and "svc__missing" in w.message
            for w in result.warnings
        )
        assert resumed.agent.ptc_bindings == harness.agent.ptc_bindings
        _record_mcp_calls(resumed, monkeypatch, seen)
        text = await _run(
            resumed.agent,
            "await svc__review()\nawait svc__review_tool()\nawait svc__get_info()\n"
            "try:\n    await svc__missing()\nexcept RuntimeError:\n    await svc__get_info_2()",
        )
        assert seen == [
            f"local:{shared.name}",
            f"mcp:{shared.name}",
            f"mcp:{primary.name}",
            f"mcp:{collision.name}",
        ], text
        resumed.tools = [*resumed.tools, missing]
        await resumed.agent.list_tools()
        await _run(resumed.agent, "await svc__missing()")
        assert seen[-1] == f"mcp:{missing.name}"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["not-a-name", "class"])
@pytest.mark.parametrize("restore_transcript", [True, False])
async def test_invalid_saved_bindings_are_dropped_and_hydration_continues(
    tmp_path: Path, invalid: str, restore_transcript: bool
) -> None:
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path, respect_env_override=False)
    session = manager.create_session()
    async with _running(name="first") as first, _running(name="second") as second:
        for state in (first, second):
            await state.send()
            await session.save_history(state.agent)
        saved = second.agent.ptc_bindings
    metadata = session.directory / "session.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    functions = payload["continuation"]["agents"]["first"]["ptc_bindings"]["functions"]
    functions[invalid] = functions.pop("github__op_0")
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    async with AsyncExitStack() as stack:
        first = await stack.enter_async_context(_running(name="first"))
        second = await stack.enter_async_context(_running(name="second"))
        # A name this conversation reserved for a tool that has since disconnected.
        first.tools = [*first.tools, *_tools("stale", 1)]
        await first.agent.list_tools()
        first.tools = _tools("github", 4)
        await first.agent.list_tools()
        assert "stale__op_0" in first.agent.ptc_bindings.functions

        result = await SessionHydrator().hydrate_session(
            session=session,
            agents={"first": first.agent, "second": second.agent},
            fallback_agent_name="first",
            policy=SessionHydrationPolicy(restore_transcript=restore_transcript),
        )

        assert [w.code for w in result.warnings if w.agent_name == "first"] == [
            "ptc-bindings-invalid"
        ]
        assert set(first.agent.ptc_bindings.functions) == {tool.name for tool in first.tools}
        assert second.agent.ptc_bindings == saved
        if restore_transcript:
            assert first.agent.message_history and second.agent.message_history


@pytest.mark.asyncio
async def test_removed_plain_name_cannot_be_inherited_by_a_new_alias(harness: _Agent) -> None:
    agent = harness.agent
    seen: list[str] = []
    agent.add_tool(_local_tool("exec_sandbox_tool", seen))
    await agent.list_tools()
    agent.remove_tool("exec_sandbox_tool")
    agent.add_tool(_local_tool("exec_sandbox", seen))
    await agent.list_tools()
    text = await _run(
        agent,
        "try:\n    await exec_sandbox_tool()\nexcept RuntimeError:\n    await exec_sandbox_tool_2()",
    )
    assert seen == ["local:exec_sandbox"], text


@pytest.mark.asyncio
async def test_enabling_builtin_keeps_an_existing_user_binding(harness: _Agent) -> None:
    agent = harness.agent
    set_harness_tools(agent, False)
    seen: list[str] = []
    agent.add_tool(_local_tool("get_resource", seen))
    await agent.list_tools()
    set_harness_tools(agent, True)
    await agent.list_tools()

    text = await _run(agent, "await get_resource()")
    assert seen == ["local:get_resource"], text
    assert agent.ptc_bindings.functions["get_resource"].origin == "local"
    assert agent.ptc_bindings.functions["get_resource_2"].origin == "builtin"


@pytest.mark.asyncio
async def test_captured_builtin_fails_locally_after_disable(
    harness: _Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = harness.agent
    assert set_subagent_tool_enabled(agent, True)
    seen: list[str] = []
    _record_mcp_calls(harness, monkeypatch, seen)
    harness.tools = [Tool(name="svc__subagent", input_schema={})]

    def disable() -> bool:
        return set_subagent_tool_enabled(agent, False)

    agent.add_tool(build_default_function_tool(disable))
    await agent.list_tools()
    text = await _run(agent, 'await disable()\nprint(await settle(subagent(message="hello")))')
    assert "`subagent` is no longer available" in text
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("local_first", [False, True])
async def test_first_holder_keeps_a_shared_name(
    harness: _Agent, monkeypatch: pytest.MonkeyPatch, local_first: bool
) -> None:
    agent = harness.agent
    seen: list[str] = []
    _record_mcp_calls(harness, monkeypatch, seen)
    mcp = [Tool(name="agent__review", input_schema={"type": "object"})]
    if local_first:
        agent.add_tool(_local_tool("agent__review", seen))
    else:
        harness.tools = mcp
    await agent.list_tools()
    harness.tools = mcp
    agent.add_tool(_local_tool("agent__review", seen))
    await agent.list_tools()

    text = await _run(agent, "await agent__review()\nawait agent__review_tool()")
    holder, other = ("local:agent__review", "mcp:agent__review")
    if not local_first:
        holder, other = other, holder
    assert seen == [holder, other], text

    # The first holder leaves: the other tool keeps its alias rather than take the name.
    if local_first:
        agent.remove_tool("agent__review")
    else:
        harness.tools = []
    names = await _function_names(agent)
    assert "agent__review_tool" in names and "agent__review" not in names
    seen.clear()
    await _run(agent, "await agent__review_tool()")
    assert seen == [other]
    text = await _run(agent, "await agent__review()")
    assert "`agent__review` is no longer available" in text
    assert seen == [other]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("captured_local", "keep_captured"), [(False, True), (False, False), (True, False)]
)
async def test_a_running_script_keeps_the_routes_it_started_with(
    harness: _Agent, monkeypatch: pytest.MonkeyPatch, captured_local: bool, keep_captured: bool
) -> None:
    """A refresh mid-run never sends a captured name to a tool of the other origin."""
    agent = harness.agent
    seen: list[str] = []
    _record_mcp_calls(harness, monkeypatch, seen)
    mcp = [Tool(name="agent__review", input_schema={"type": "object"})]

    async def refresh() -> str:
        """Connect and disconnect tool sources while the script runs."""
        if captured_local:
            agent.remove_tool("agent__review")
            harness.tools = mcp
        else:
            agent.add_tool(_local_tool("agent__review", seen))
            if not keep_captured:
                harness.tools = []
        await agent.list_tools()
        return "refreshed"

    agent.add_tool(build_default_function_tool(refresh))
    if captured_local:
        agent.add_tool(_local_tool("agent__review", seen))
    else:
        harness.tools = mcp
    await agent.list_tools()

    text = await _run(agent, "await refresh()\nprint(await settle(agent__review()))")
    if keep_captured:
        assert seen == ["mcp:agent__review"], text
    else:
        assert seen == [], text
        assert "`agent__review` is no longer available" in text


@pytest.mark.asyncio
async def test_seen_functions_follow_signatures_in_the_active_conversation(harness: _Agent) -> None:
    agent = harness.agent

    async def seen() -> set[str]:
        return {
            tool.python_name
            for tool in (await agent.tool_surface()).programmatic_functions()
            if tool.seen and tool.python_name is not None
        }

    assert await seen() == set()

    code = "\n".join(
        [
            'ignored = await search_tools(query="github__op_0", limit=1)',
            'print(await search_tools(query="github__op_1", limit=1))',
        ]
    )
    await agent.generate(
        f"***CALL_TOOL exec_sandbox {json.dumps({'code': code})}", RequestParams(max_iterations=3)
    )
    assert await seen() == {"github__op_1"}

    message = Prompt.user("Use ^tool:github__op_2 for this")
    expand_tool_mentions(await agent.tool_surface(), message)
    assert await seen() == {"github__op_1"}
    await harness.send(message)
    assert await seen() == {"github__op_1", "github__op_2"}

    checkpoint = PromptMessageExtended(
        role="user",
        channels={FAST_AGENT_COMPACTION_CHANNEL: [TextContent(type="text", text="summary")]},
    )
    agent.load_message_history([*agent.message_history, checkpoint])
    assert await seen() == set()


@pytest.mark.asyncio
async def test_roster_is_announced_again_after_compaction(harness: _Agent) -> None:
    await harness.send()
    checkpoint = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="Summary of the conversation so far.")],
        channels={FAST_AGENT_COMPACTION_CHANNEL: [TextContent(type="text", text="{}")]},
    )
    agent = harness.agent
    agent.load_message_history([*agent.message_history, checkpoint, Prompt.assistant("ok")])

    await harness.send("after compaction")
    await harness.send("and again")
    assert len(harness.notices()) == 2


@pytest.mark.asyncio
async def test_stateless_transcript_with_earlier_tool_results_gets_the_roster(
    harness: _Agent,
) -> None:
    call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="exec_sandbox", arguments={"code": "1"}),
    )
    transcript = [
        Prompt.user("first"),
        PromptMessageExtended(role="assistant", tool_calls={"call-1": call}),
        PromptMessageExtended(
            role="user",
            tool_results={"call-1": CallToolResult(content=[TextContent(type="text", text="1")])},
        ),
        Prompt.assistant("done"),
        Prompt.user("now"),
    ]
    response = await harness.agent.generate(
        transcript, RequestParams(max_iterations=1, use_history=False)
    )
    assert "<fastagent:tool-sources>" in (response.all_text() or "")


@pytest.mark.asyncio
async def test_roster_is_retracted_once_when_search_goes_inactive(harness: _Agent) -> None:
    await harness.send()
    harness.tools = _tools("github", 2)
    await harness.send("disconnected")
    await harness.send("still disconnected")
    notices = harness.notices()
    retraction = (
        "<fastagent:tool-sources>All available functions are now listed directly in "
        "`exec_sandbox`; `search_tools` is no longer available.</fastagent:tool-sources>"
    )
    assert notices[1:] == [retraction]

    harness.tools = _tools("github", 4)
    await harness.send("reconnected")
    assert harness.notices()[2:] == notices[:1]


@pytest.mark.asyncio
async def test_human_input_tool_is_listed_once_and_stably() -> None:
    async with _running(human_input=True) as harness:
        agent = harness.agent

        async def listing() -> list[dict[str, object]]:
            return [
                tool.model_dump(include={"name", "description", "input_schema"})
                for tool in (await agent.list_tools()).tools
            ]

        assert await listing() == await listing()
        surface = await agent.tool_surface()
        names = [tool.tool.name for tool in surface.tools]
        assert names.count(HUMAN_INPUT_TOOL_NAME) == 1


@pytest.mark.asyncio
async def test_a_mid_turn_connect_reaches_the_next_llm_call(
    harness: _Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = harness.agent
    harness.tools = _tools("github", 2)

    def connect() -> str:
        """Connect more tools, as `/mcp connect` during a turn does."""
        harness.tools = _tools("github", 3)
        return "connected"

    agent.add_tool(build_default_function_tool(connect))
    llm = agent.llm
    assert isinstance(llm, PassthroughLLM)
    sent: list[str] = []
    apply_prompt = llm._apply_prompt_provider_specific

    async def recording(
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        sent.append(
            next(tool.description or "" for tool in tools or [] if tool.name == "exec_sandbox")
        )
        return await apply_prompt(messages, request_params, tools, is_template)

    monkeypatch.setattr(llm, "_apply_prompt_provider_specific", recording)
    await agent.generate(
        '***CALL_TOOL exec_sandbox {"code": "await connect()"}',
        RequestParams(max_iterations=2),
    )
    assert ["search_tools" in description for description in sent] == [False, True]


@pytest.mark.asyncio
async def test_mention_names_follow_the_latest_catalogue(harness: _Agent) -> None:
    await harness.agent.list_tools()
    names = harness.agent.ptc_mention_names
    assert names is not None and {"github__op_0", "github__op_3"} <= names
    assert harness.agent.ptc_mention_names is names

    harness.tools = _tools("github", 1)
    await harness.agent.list_tools()
    names = harness.agent.ptc_mention_names
    assert names is not None and "github__op_0" in names and "github__op_3" not in names
