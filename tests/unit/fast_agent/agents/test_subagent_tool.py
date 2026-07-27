import asyncio
import io
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from fastmcp.exceptions import ValidationError
from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams, CallToolResult
from rich.console import Console, Group
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.subagent_tool import (
    SUBAGENT_TOOL_NAME,
    _default_progress_display,
    _finalize_subagent_run,
    _SubagentMonitorCoordinator,
    install_subagent_tool,
)
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunner, ToolRunnerHooks
from fast_agent.constants import FAST_AGENT_SUBAGENT_RESULT_METADATA, FAST_AGENT_TOOL_METADATA
from fast_agent.context import Context
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.event_progress import ProgressAction, ProgressEvent
from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageSchema,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt import Prompt
from fast_agent.session import (
    Session,
    SessionChildLinkSnapshot,
    SessionManager,
    load_session_snapshot,
    reset_session_manager,
    set_session_manager,
)
from fast_agent.session.trace_export_atif import AtifRunSource, build_atif_trajectory
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.progress.display import RichProgressDisplay
from fast_agent.ui.progress_display import progress_display

if TYPE_CHECKING:
    from fast_agent.mcp.skybridge import SkybridgeServerConfig
    from fast_agent.ui.terminal_images.renderer import ImageRenderItem


class SubagentDisplayRecorder(ConsoleDisplay):
    def __init__(self) -> None:
        super().__init__(config=None)
        self.events: list[tuple[str, dict[str, object]]] = []

    def show_user_message(
        self,
        message: str | Text,
        model: str | None = None,
        chat_turn: int = 0,
        total_turns: int | None = None,
        turn_range: tuple[int, int] | None = None,
        name: str | None = None,
        attachments: list[str] | None = None,
        image_previews: list["ImageRenderItem"] | None = None,
        part_count: int | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        del (
            model,
            chat_turn,
            total_turns,
            turn_range,
            attachments,
            image_previews,
            part_count,
            show_hook_indicator,
        )
        self.events.append(("user", {"message": message, "name": name}))

    async def show_assistant_message(
        self,
        message_text: str | Text | PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_indexes: list[int] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        pre_content: Text | Group | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
        show_reprint_banner: bool = False,
    ) -> None:
        del (
            highlight_indexes,
            max_item_length,
            additional_message,
            pre_content,
            render_markdown,
            show_hook_indicator,
            show_reprint_banner,
        )
        self.events.append(
            (
                "assistant",
                {
                    "message": message_text,
                    "name": name,
                    "model": model,
                    "bottom_items": bottom_items,
                },
            )
        )

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        bottom_items: list[str] | None = None,
        highlight_indexes: list[int] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        type_label: str | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        del (
            tool_args,
            bottom_items,
            highlight_indexes,
            max_item_length,
            name,
            metadata,
            tool_call_id,
            type_label,
            show_hook_indicator,
        )
        self.events.append(("tool_call", {"tool_name": tool_name}))

    def show_tool_result(
        self,
        result: CallToolResult,
        name: str | None = None,
        tool_name: str | None = None,
        skybridge_config: "SkybridgeServerConfig | None" = None,
        timing_ms: float | None = None,
        tool_call_id: str | None = None,
        type_label: str | None = None,
        truncate_content: bool = True,
        show_hook_indicator: bool = False,
    ) -> None:
        del (
            result,
            name,
            skybridge_config,
            timing_ms,
            tool_call_id,
            type_label,
            truncate_content,
            show_hook_indicator,
        )
        self.events.append(("tool_result", {"tool_name": tool_name}))


class RecordingProgressDisplay(RichProgressDisplay):
    def __init__(self) -> None:
        super().__init__(console=Console(file=io.StringIO(), force_terminal=False))
        self.events: list[ProgressEvent] = []

    def update(self, event: ProgressEvent) -> None:
        self.events.append(event)
        super().update(event)


@pytest.mark.unit
def test_subagent_monitor_uses_shared_progress_display_by_default() -> None:
    assert _default_progress_display() is progress_display


def _subagent_call(message: str, label: str | None = None) -> CallToolRequest:
    arguments: dict[str, str] = {"message": message}
    if label is not None:
        arguments["label"] = label
    return CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=SUBAGENT_TOOL_NAME, arguments=arguments),
    )


class InspectingLLM(PassthroughLLM):
    def __init__(self, agent: ToolAgent, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del request_params, is_template
        tool_names = sorted(tool.name for tool in tools or [])
        hooks = self.agent.tool_runner_hooks
        self.usage_accumulator.add_turn(
            TurnUsage(
                provider=Provider.FAST_AGENT,
                usage_schema=UsageSchema.OPENAI_CHAT,
                model="passthrough",
                prompt=PromptTokenUsage(total=3),
                completion=CompletionTokenUsage(total=2),
            )
        )
        return Prompt.assistant(
            f"{multipart_messages[-1].last_text()} | tools={tool_names} | hooks={hooks is not None}"
        )


class BlockingLLM(PassthroughLLM):
    def __init__(self, entered: asyncio.Event, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entered = entered
        self.release = asyncio.Event()

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del multipart_messages, request_params, tools, is_template
        self.usage_accumulator.add_turn(
            TurnUsage(
                provider=Provider.FAST_AGENT,
                usage_schema=UsageSchema.OPENAI_CHAT,
                model="passthrough",
                prompt=PromptTokenUsage(total=3, cache_read=1),
                completion=CompletionTokenUsage(total=2),
            )
        )
        self.entered.set()
        await self.release.wait()
        return Prompt.assistant("done")


class ParallelBlockingLLM(PassthroughLLM):
    def __init__(
        self,
        started: asyncio.Queue[str],
        release: asyncio.Event,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.started = started
        self.release = release

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del multipart_messages, request_params, tools, is_template
        self.started.put_nowait(self.name)
        await self.release.wait()
        return Prompt.assistant("done")


class FailingLLM(PassthroughLLM):
    def _resolve_retry_count(self) -> int:
        return 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del multipart_messages, request_params, tools, is_template
        raise RuntimeError("simulated failure")


class ToolUsingLLM(PassthroughLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._turn = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del multipart_messages, request_params, tools, is_template
        self._turn += 1
        self.usage_accumulator.add_turn(
            TurnUsage(
                provider=Provider.FAST_AGENT,
                usage_schema=UsageSchema.OPENAI_CHAT,
                model="passthrough",
                prompt=PromptTokenUsage(total=3, cache_read=1),
                completion=CompletionTokenUsage(total=2),
            )
        )
        if self._turn == 1:
            return Prompt.assistant(
                "use lookup",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "lookup-call": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="lookup", arguments={}),
                    )
                },
            )
        return Prompt.assistant("done")


class TrackingToolAgent(ToolAgent):
    instances: ClassVar[list["TrackingToolAgent"]] = []

    def __init__(self, config: AgentConfig, context: Context | None = None) -> None:
        super().__init__(config, context=context)
        self.shutdown_called = False
        self.instances.append(self)

    async def shutdown(self) -> None:
        self.shutdown_called = True
        await super().shutdown()


class SlowSaveSession(Session):
    async def save_history(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def inspecting_factory(created: list[InspectingLLM]) -> Callable[..., FastAgentLLMProtocol]:
    def factory(
        agent: ToolAgent,
        request_params: RequestParams | None = None,
        **kwargs,
    ) -> FastAgentLLMProtocol:
        llm = InspectingLLM(
            agent,
            request_params=request_params,
            name=agent.name,
            instructions=agent.instruction,
            **kwargs,
        )
        created.append(llm)
        return llm

    return factory


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_inherits_tools_without_recursion_or_parent_hooks() -> None:
    async def read_note() -> str:
        return "note"

    parent_hook_calls = 0

    async def before_llm_call(
        _runner: ToolRunner,
        _messages: list[PromptMessageExtended],
    ) -> None:
        nonlocal parent_hook_calls
        parent_hook_calls += 1

    created: list[InspectingLLM] = []
    parent = ToolAgent(AgentConfig("parent", subagents=True), [read_note])
    parent.tool_runner_hooks = ToolRunnerHooks(before_llm_call=before_llm_call)
    await parent.attach_llm(inspecting_factory(created))

    assert install_subagent_tool(parent)
    assert install_subagent_tool(parent)

    result = await parent.call_tool(SUBAGENT_TOOL_NAME, {"message": "inspect"})
    text = get_text(result.content[0])

    assert text is not None
    assert "inspect" in text
    assert "read_note" in text
    assert SUBAGENT_TOOL_NAME not in text.split("tools=", 1)[1].split(" |", 1)[0]
    assert "hooks=True" in text
    assert parent_hook_calls == 0
    assert len(created) == 2
    assert parent.usage_accumulator is not None
    assert parent.usage_accumulator.summary.prompt.total == 3
    assert parent.subagent_usage_accumulator.summary.prompt.total == 3
    assert parent.subagent_usage_accumulator.summary.completion.total == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detached_instance_accepts_model_override() -> None:
    created: list[InspectingLLM] = []
    parent = ToolAgent(AgentConfig("parent"))
    await parent.attach_llm(inspecting_factory(created))

    clone = await parent.spawn_isolated_instance(model="playback")
    try:
        assert clone.llm is not None
        assert clone.llm.resolved_model.selected_model_name == "playback"
    finally:
        await clone.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_isolated_clone_rebuilds_instruction_for_child_context_and_model() -> None:
    parent = McpAgent(
        AgentConfig(
            "parent",
            instruction="environment={{env}} model={{model_specific}}",
        )
    )
    await parent.attach_llm(lambda agent, **kwargs: InspectingLLM(agent, **kwargs))
    parent.set_instruction_context({"env": "sandbox", "model_specific": "parent-only"})
    await rebuild_agent_instruction(parent)
    assert parent.instruction == "environment=sandbox model=parent-only"

    clone = await parent.spawn_isolated_instance(model="playback")
    try:
        assert clone.instruction == "environment=sandbox model="
        assert "{{env}}" not in clone.instruction
        assert "parent-only" not in clone.instruction
    finally:
        await clone.shutdown()
        await parent.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_isolated_clone_does_not_run_parent_lifecycle_hooks(tmp_path) -> None:
    marker = tmp_path / "isolated-hook-ran"
    hook_file = tmp_path / "isolated_hooks.py"
    hook_file.write_text(
        f"""
from pathlib import Path

async def record_hook(ctx):
    Path({str(marker)!r}).write_text(ctx.hook_type, encoding="utf-8")
""",
        encoding="utf-8",
    )
    parent = ToolAgent(
        AgentConfig(
            "parent",
            lifecycle_hooks={
                "on_start": f"{hook_file}:record_hook",
                "on_shutdown": f"{hook_file}:record_hook",
            },
        )
    )
    await parent.attach_llm(lambda agent, **kwargs: InspectingLLM(agent, **kwargs))

    clone = await parent.spawn_isolated_instance()
    await clone.shutdown()

    assert not marker.exists()
    await parent.shutdown()
    assert marker.read_text(encoding="utf-8") == "on_shutdown"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_incomplete_isolated_clone_is_shutdown() -> None:
    TrackingToolAgent.instances.clear()
    parent = TrackingToolAgent(AgentConfig("parent"))

    def factory(agent: AgentProtocol, **kwargs) -> FastAgentLLMProtocol:
        assert isinstance(agent, ToolAgent)
        if agent.name != "parent":
            raise RuntimeError("attach failed")
        return InspectingLLM(agent, name=agent.name, instructions=agent.instruction, **kwargs)

    await parent.attach_llm(factory)

    with pytest.raises(RuntimeError, match="attach failed"):
        await parent.spawn_isolated_instance(name="child")

    child = TrackingToolAgent.instances[-1]
    assert child is not parent
    assert child.shutdown_called


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalization_timeout_still_releases_clone_and_merges_usage(tmp_path) -> None:
    created: list[InspectingLLM] = []
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()
    persisted_child = manager.create_child_session(
        parent_session,
        SessionChildLinkSnapshot(
            parent_session_id=parent_session.info.name,
            parent_agent_name="parent",
            parent_tool_call_id="parent-call",
        ),
    )
    child_session = SlowSaveSession(
        persisted_child.info,
        persisted_child.directory,
        manager=manager,
    )
    parent = ToolAgent(AgentConfig("parent"))
    clone = TrackingToolAgent(AgentConfig("parent[child]"))
    await parent.attach_llm(inspecting_factory(created))
    await clone.attach_llm(inspecting_factory(created))
    clone.load_message_history([Prompt.user("persist me")])
    assert clone.usage_accumulator is not None
    clone.usage_accumulator.add_turn(
        TurnUsage(
            provider=Provider.FAST_AGENT,
            usage_schema=UsageSchema.OPENAI_CHAT,
            model="passthrough",
            prompt=PromptTokenUsage(total=3),
            completion=CompletionTokenUsage(total=2),
        )
    )
    monitor = _SubagentMonitorCoordinator(
        display=RecordingProgressDisplay(),
        parent_name=parent.name,
    )
    progress = monitor.start(
        label="child",
        child_name=clone.name,
        parent_tool_call_id="parent-call",
    )

    with pytest.raises(TimeoutError):
        await _finalize_subagent_run(
            parent=parent,
            clone=clone,
            child_session=child_session,
            child_name=clone.name,
            status="completed",
            progress=progress,
            message="persist me",
            requested_model=None,
            label="child",
            parent_tool_call_id="parent-call",
            started_at="2026-07-26T00:00:00+00:00",
            finalization_timeout_seconds=0.01,
        )

    assert clone.shutdown_called
    assert parent.usage_accumulator is not None
    assert parent.usage_accumulator.summary.prompt.total == 3
    assert parent.subagent_usage_accumulator.summary.completion.total == 2
    await parent.shutdown()


@pytest.mark.unit
def test_subagent_tool_name_is_reserved() -> None:
    def subagent() -> str:
        return "custom"

    agent = ToolAgent(AgentConfig("parent", subagents=True), [subagent])

    with pytest.raises(AgentConfigError, match="reserved"):
        install_subagent_tool(agent)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subagents", "installed"),
    [(None, False), (True, True), (False, False)],
)
async def test_subagent_tool_respects_normal_agent_config(
    subagents: bool | None,
    installed: bool,
) -> None:
    agent = ToolAgent(AgentConfig("normal", subagents=subagents))

    assert install_subagent_tool(agent) is installed
    tools = await agent.list_tools()

    assert (SUBAGENT_TOOL_NAME in {tool.name for tool in tools.tools}) is installed


@pytest.mark.unit
def test_subagent_model_does_not_enable_builtin_subagents() -> None:
    agent = ToolAgent(AgentConfig("normal", subagent_model="passthrough"))

    assert install_subagent_tool(agent) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_only_agents_cannot_enable_builtin_subagents() -> None:
    tool_only = ToolAgent(AgentConfig("tool_only", tool_only=True, subagents=True))

    assert install_subagent_tool(tool_only) is False
    assert SUBAGENT_TOOL_NAME not in {tool.name for tool in (await tool_only.list_tools()).tools}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_label_schema_and_tool_boundary_validation() -> None:
    parent = ToolAgent(AgentConfig("parent", subagents=True))
    assert install_subagent_tool(parent, label_generator=lambda: "brisk-otter")
    tool = parent._execution_tools[SUBAGENT_TOOL_NAME]

    schema = tool.parameters
    label_schema = schema["properties"]["label"]
    assert set(schema["properties"]) == {"message", "model", "label"}
    assert "display label" in label_schema["description"]
    assert schema["$defs"]["SubagentLabel"] == {
        "maxLength": 32,
        "minLength": 1,
        "pattern": "^[A-Za-z0-9](?:[A-Za-z0-9 _-]*[A-Za-z0-9])?$",
        "type": "string",
    }

    trimmed = await tool.run({"message": "trim this", "label": "  focused  "})
    assert trimmed.meta is not None
    details = trimmed.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
    assert details["requested_label"] == "focused"
    assert details["label"] == "focused"

    for invalid_label in ("<em>markup</em>", "line\nbreak", "a" * 33, "ends-"):
        with pytest.raises(ValidationError):
            await tool.run({"message": "reject this", "label": invalid_label})


@pytest.mark.unit
def test_subagent_schema_and_description_follow_forced_model_config() -> None:
    normal = ToolAgent(AgentConfig("normal", subagents=True))
    forced = ToolAgent(AgentConfig("forced", subagents=True, subagent_model="passthrough"))

    assert install_subagent_tool(normal)
    assert install_subagent_tool(forced)

    normal_tool = normal._execution_tools[SUBAGENT_TOOL_NAME]
    forced_tool = forced._execution_tools[SUBAGENT_TOOL_NAME]
    assert set(normal_tool.parameters["properties"]) == {"message", "model", "label"}
    assert set(forced_tool.parameters["properties"]) == {"message", "label"}
    assert forced_tool.description is not None
    assert "fixed model `passthrough`" in forced_tool.description


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_model_sources_and_forced_model_ignore_raw_model() -> None:
    created: list[InspectingLLM] = []
    normal = ToolAgent(AgentConfig("normal", model="passthrough", subagents=True))
    await normal.attach_llm(inspecting_factory(created))
    assert install_subagent_tool(normal)

    inherited = await normal.call_tool(SUBAGENT_TOOL_NAME, {"message": "inherit"})
    override = await normal.call_tool(
        SUBAGENT_TOOL_NAME,
        {"message": "override", "model": "passthrough"},
    )
    assert inherited.meta is not None
    assert override.meta is not None
    assert inherited.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]["model_source"] == "parent"
    assert override.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]["model_source"] == "tool_override"
    assert override.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]["model_spec"] == "passthrough"

    forced = ToolAgent(
        AgentConfig(
            "forced",
            model="playback",
            subagents=True,
            subagent_model="passthrough",
        )
    )
    await forced.attach_llm(inspecting_factory(created))
    assert install_subagent_tool(forced)

    result = await forced.call_tool(
        SUBAGENT_TOOL_NAME,
        {"message": "fixed", "model": "playback"},
    )

    assert result.meta is not None
    details = result.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
    assert details["model_source"] == "agent_card"
    assert details["model_spec"] == "passthrough"
    assert details["provider"] == "fast-agent"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_labels_are_generated_supplied_and_disambiguated() -> None:
    created: list[InspectingLLM] = []
    parent = ToolAgent(AgentConfig("parent", model="passthrough", subagents=True))
    await parent.attach_llm(inspecting_factory(created))
    assert install_subagent_tool(parent, label_generator=lambda: "brisk-otter")

    results = [
        await parent.call_tool(SUBAGENT_TOOL_NAME, {"message": "generated"}),
        await parent.call_tool(SUBAGENT_TOOL_NAME, {"message": "supplied", "label": " scout "}),
        await parent.call_tool(SUBAGENT_TOOL_NAME, {"message": "duplicate", "label": "scout"}),
        await parent.call_tool(SUBAGENT_TOOL_NAME, {"message": "long", "label": "a" * 32}),
        await parent.call_tool(
            SUBAGENT_TOOL_NAME, {"message": "long duplicate", "label": "a" * 32}
        ),
    ]
    details = [
        result.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
        for result in results
        if result.meta is not None
    ]

    assert [detail["requested_label"] for detail in details] == [
        None,
        "scout",
        "scout",
        "a" * 32,
        "a" * 32,
    ]
    assert [detail["label"] for detail in details] == [
        "brisk-otter",
        "scout",
        "scout-2",
        "a" * 32,
        f"{'a' * 30}-2",
    ]
    assert [detail["child_agent_name"] for detail in details] == [
        "parent[brisk-otter]",
        "parent[scout]",
        "parent[scout-2]",
        f"parent[{'a' * 32}]",
        f"parent[{'a' * 30}-2]",
    ]
    assert get_text(results[0].content[0]) == "generated | tools=[] | hooks=True"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_persists_nested_child_with_call_correlation(tmp_path) -> None:
    created: list[InspectingLLM] = []
    global_manager = SessionManager(
        cwd=tmp_path / "global",
        home_override=tmp_path / "global" / ".fast-agent",
        respect_env_override=False,
    )
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()
    set_session_manager(global_manager)

    try:
        parent = ToolAgent(
            AgentConfig("parent", model="passthrough", subagents=True),
            context=Context(session_manager=manager),
        )
        await parent.attach_llm(inspecting_factory(created))
        assert install_subagent_tool(parent, label_generator=lambda: "brisk-otter")

        result = await parent.call_tool(
            SUBAGENT_TOOL_NAME,
            {"message": "persist this"},
            tool_use_id="tool-123",
        )

        assert get_text(result.content[0]) is not None
        assert global_manager.current_session is None
        assert manager.current_session is parent_session
        assert [info.name for info in global_manager.list_sessions()] == []
        assert [info.name for info in manager.list_sessions()] == [parent_session.info.name]

        children = manager.list_child_sessions(parent_session)
        assert len(children) == 1
        child = children[0]
        snapshot = load_session_snapshot(
            json.loads((child.directory / "session.json").read_text(encoding="utf-8"))
        )

        assert snapshot.execution.resumable is False
        assert snapshot.execution.child_link is not None
        assert snapshot.execution.child_link.parent_session_id == parent_session.info.name
        assert snapshot.execution.child_link.parent_agent_name == "parent"
        assert snapshot.execution.child_link.parent_tool_call_id == "tool-123"
        assert snapshot.execution.status == "completed"
        assert snapshot.execution.started_at is not None
        assert snapshot.execution.completed_at is not None
        active_agent = snapshot.continuation.active_agent
        assert active_agent is not None
        agent_snapshot = snapshot.continuation.agents[active_agent]
        assert agent_snapshot.model == "passthrough"
        assert agent_snapshot.history_file is not None
        history = (child.directory / agent_snapshot.history_file).read_text(encoding="utf-8")
        assert "persist this" in history
        assert snapshot.analysis.usage_summary is not None
        assert snapshot.analysis.usage_summary.total_tokens == 5
    finally:
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_persists_last_turn_when_history_is_disabled(tmp_path) -> None:
    created: list[InspectingLLM] = []
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()

    parent = ToolAgent(
        AgentConfig(
            "parent",
            model="passthrough",
            subagents=True,
            use_history=False,
        ),
        context=Context(session_manager=manager),
    )
    await parent.attach_llm(inspecting_factory(created))
    assert install_subagent_tool(parent)

    result = await parent.call_tool(
        SUBAGENT_TOOL_NAME,
        {"message": "persist transient child turn"},
        tool_use_id="tool-transient",
    )

    assert not result.isError
    clone = created[-1].agent
    assert clone.message_history == []
    assert clone.last_turn_messages

    child = manager.list_child_sessions(parent_session)[0]
    snapshot = load_session_snapshot(
        json.loads((child.directory / "session.json").read_text(encoding="utf-8"))
    )
    active_agent = snapshot.continuation.active_agent
    assert active_agent is not None
    history_file = snapshot.continuation.agents[active_agent].history_file
    assert history_file is not None
    history = (child.directory / history_file).read_text(encoding="utf-8")
    assert "persist transient child turn" in history

    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id=parent_session.info.name,
            agent_name="parent",
            model_name="passthrough",
            provider="fast-agent",
            history=[
                Prompt.assistant(
                    tool_calls={"tool-transient": _subagent_call("persist transient child turn")}
                )
            ],
            message_timestamps=(None,),
            parent_session_dir=parent_session.directory,
        )
    )
    assert trajectory.subagent_trajectories is not None
    assert len(trajectory.subagent_trajectories) == 1
    assert any(
        "persist transient child turn" in str(step.message)
        for step in trajectory.subagent_trajectories[0].steps
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sessionless_subagent_is_embedded_from_transient_capture() -> None:
    async def lookup() -> str:
        return "found"

    parent = ToolAgent(
        AgentConfig("parent", model="passthrough", subagents=True),
        [lookup],
    )
    await parent.attach_llm(lambda agent, **kwargs: ToolUsingLLM(name=agent.name, **kwargs))
    parent.enable_subagent_trajectory_capture()
    assert install_subagent_tool(parent)
    request = Prompt.assistant(
        tool_calls={"parent-call": _subagent_call("inspect without a session", "inspector")}
    )

    result = await parent.run_tools(request)

    assert len(parent.subagent_trajectory_records) == 1
    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id="run-sessionless",
            agent_name=parent.name,
            model_name="passthrough",
            provider="fast-agent",
            history=[request, result],
            message_timestamps=(None, None),
            transient_child_trajectories=parent.subagent_trajectory_records,
        )
    )
    assert trajectory.subagent_trajectories is not None
    assert len(trajectory.subagent_trajectories) == 1
    child = trajectory.subagent_trajectories[0]
    assert child.agent.model_name == "passthrough"
    assert child.extra is not None
    assert child.extra["persistence"] == "transient"
    assert child.extra["parent_tool_call_id"] == "parent-call"
    assert child.extra["model"] == "passthrough"
    assert child.extra["provider"] == "fast-agent"
    assert any(
        step.tool_calls is not None
        and any(call.function_name == "lookup" for call in step.tool_calls)
        for step in child.steps
    )
    parent_observation = next(
        step.observation for step in trajectory.steps if step.observation is not None
    )
    assert parent_observation.results[0].subagent_trajectory_ref is not None
    assert (
        parent_observation.results[0].subagent_trajectory_ref[0].trajectory_id
        == child.trajectory_id
    )
    await parent.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_isolated_subagent_does_not_persist_to_parent_session(tmp_path) -> None:
    instances: list[ToolAgent] = []
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()

    async def lookup() -> str:
        return "found"

    parent = ToolAgent(
        AgentConfig("parent", model="passthrough", subagents=True),
        [lookup],
        context=Context(session_manager=manager),
    )
    await parent.attach_llm(
        lambda agent, **kwargs: instances.append(agent) or ToolUsingLLM(name=agent.name, **kwargs)
    )
    assert install_subagent_tool(parent)

    result = await parent.call_tool(SUBAGENT_TOOL_NAME, {"message": "persist only as a child"})

    assert not result.isError
    clone = instances[-1]
    assert clone.session_history_persistence_enabled is False

    root_snapshot = load_session_snapshot(
        json.loads((parent_session.directory / "session.json").read_text(encoding="utf-8"))
    )
    assert clone.name not in root_snapshot.continuation.agents
    assert list(parent_session.directory.glob("history_*.json")) == []
    assert len(manager.list_child_sessions(parent_session)) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_persists_terminal_failure_and_cancellation(tmp_path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()
    set_session_manager(manager)

    try:
        failed_progress = RecordingProgressDisplay()
        failing_parent = ToolAgent(
            AgentConfig("failing", model="passthrough", subagents=True),
            context=Context(session_manager=manager),
        )
        await failing_parent.attach_llm(
            lambda agent, **kwargs: FailingLLM(name=agent.name, **kwargs)
        )
        assert install_subagent_tool(
            failing_parent,
            progress_display=failed_progress,
            label_generator=lambda: "brisk-otter",
        )

        failed = await failing_parent.call_tool(
            SUBAGENT_TOOL_NAME,
            {"message": "fail"},
            tool_use_id="tool-failed",
        )
        assert failed.isError
        failed_child_events = [
            event for event in failed_progress.events if event.agent_name == "failing[brisk-otter]"
        ]
        assert failed_child_events[-1].action == ProgressAction.READY
        assert failed_child_events[-1].details == "failed"
        assert failed_progress.events[-1].agent_name == "failing"
        assert failed_progress.events[-1].action == ProgressAction.READY
        assert failed_progress._taskmap == {}

        entered = asyncio.Event()
        cancelled_progress = RecordingProgressDisplay()
        blocking_parent = ToolAgent(
            AgentConfig("blocking", model="passthrough", subagents=True),
            context=Context(session_manager=manager),
        )
        await blocking_parent.attach_llm(
            lambda agent, **kwargs: BlockingLLM(entered, name=agent.name, **kwargs)
        )
        assert install_subagent_tool(
            blocking_parent,
            progress_display=cancelled_progress,
            label_generator=lambda: "brisk-otter",
        )
        task = asyncio.create_task(
            blocking_parent.call_tool(
                SUBAGENT_TOOL_NAME,
                {"message": "cancel"},
                tool_use_id="tool-cancelled",
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cancelled_child_events = [
            event
            for event in cancelled_progress.events
            if event.agent_name == "blocking[brisk-otter]"
        ]
        assert cancelled_child_events[-1].action == ProgressAction.READY
        assert cancelled_child_events[-1].details == "cancelled"
        assert cancelled_progress.events[-1].agent_name == "blocking"
        assert cancelled_progress.events[-1].action == ProgressAction.READY
        assert cancelled_progress._taskmap == {}

        children = manager.list_child_sessions(parent_session)
        statuses = {}
        for child in children:
            snapshot = load_session_snapshot(
                json.loads((child.directory / "session.json").read_text(encoding="utf-8"))
            )
            assert snapshot.execution.child_link is not None
            call_id = snapshot.execution.child_link.parent_tool_call_id
            statuses[call_id] = snapshot.execution.status
            if call_id == "tool-cancelled":
                active_agent = snapshot.continuation.active_agent
                assert active_agent == "blocking[brisk-otter]"
                history_file = snapshot.continuation.agents[active_agent].history_file
                assert history_file is not None
                assert (child.directory / history_file).exists()
                assert snapshot.analysis.usage_summary is not None
                assert snapshot.analysis.usage_summary.total_tokens == 5
        assert statuses == {"tool-failed": "failed", "tool-cancelled": "cancelled"}
    finally:
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_live_display_uses_chat_panels_and_preserves_result_metadata(
    tmp_path,
) -> None:
    created: list[InspectingLLM] = []
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()
    set_session_manager(manager)

    try:
        parent = ToolAgent(AgentConfig("parent", model="passthrough", subagents=True))
        await parent.attach_llm(inspecting_factory(created))
        assert install_subagent_tool(parent, label_generator=lambda: "brisk-otter")
        display = SubagentDisplayRecorder()
        parent.display = display

        result_message = await parent.run_tools(
            Prompt.assistant(
                tool_calls={"call-1": _subagent_call("research this", "  research-pal  ")}
            )
        )

        assert result_message.tool_results is not None
        result = result_message.tool_results["call-1"]
        text = get_text(result.content[0])
        assert text is not None
        assert text.startswith("research this")
        assert result_message.channels is not None
        assert result_message.channels["fast-agent-tool-metadata"]
        assert result.meta is not None
        details = result.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
        assert details["child_agent_name"] == "parent[research-pal]"
        assert details["requested_label"] == "research-pal"
        assert details["label"] == "research-pal"
        assert details["model_spec"] == "passthrough"
        assert details["provider"] == "fast-agent"
        assert details["status"] == "completed"
        assert details["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        assert details["turn_count"] == 1

        children = manager.list_child_sessions(parent_session)
        assert len(children) == 1
        assert details["child_session_id"] == children[0].info.name
        assert display.events == [
            (
                "user",
                {"message": "research this", "name": "parent → research-pal"},
            ),
            (
                "assistant",
                {
                    "message": text,
                    "name": "subagent: research-pal",
                    "model": "passthrough",
                    "bottom_items": [f"session {children[0].info.name}"],
                },
            ),
        ]
    finally:
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parallel_subagent_live_display_keeps_call_and_session_identity(tmp_path) -> None:
    created: list[InspectingLLM] = []
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()
    set_session_manager(manager)

    try:
        parent = ToolAgent(AgentConfig("parent", model="passthrough", subagents=True))
        await parent.attach_llm(inspecting_factory(created))
        assert install_subagent_tool(parent, label_generator=lambda: "brisk-otter")
        display = SubagentDisplayRecorder()
        parent.display = display

        result_message = await parent.run_tools(
            Prompt.assistant(
                tool_calls={
                    "call-a": _subagent_call("first task"),
                    "call-b": _subagent_call("second task"),
                }
            )
        )

        assert result_message.tool_results is not None
        results = result_message.tool_results
        assert all(result.meta is not None for result in results.values())
        assert {get_text(result.content[0]) for result in results.values()} == {
            "first task | tools=[] | hooks=True",
            "second task | tools=[] | hooks=True",
        }
        details_by_call = {
            call_id: result.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
            for call_id, result in results.items()
            if result.meta is not None
        }
        assert {
            call_id: details["child_agent_name"] for call_id, details in details_by_call.items()
        } == {
            "call-a": "parent[brisk-otter]",
            "call-b": "parent[brisk-otter-2]",
        }
        assert len({details["child_session_id"] for details in details_by_call.values()}) == 2
        assert not [event for event, _ in display.events if event.startswith("tool_")]
        assert [payload["name"] for event, payload in display.events if event == "user"] == [
            "parent → subagent",
            "parent → subagent",
        ]
        assert {payload["name"] for event, payload in display.events if event == "assistant"} == {
            "subagent: brisk-otter",
            "subagent: brisk-otter-2",
        }

        children = manager.list_child_sessions(parent_session)
        child_links = [
            load_session_snapshot(
                json.loads((child.directory / "session.json").read_text(encoding="utf-8"))
            ).execution.child_link
            for child in children
        ]
        assert {link.parent_tool_call_id for link in child_links if link is not None} == {
            "call-a",
            "call-b",
        }
    finally:
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_subagent_live_display_uses_chat_panels_for_supplied_label(tmp_path) -> None:
    created: list[InspectingLLM] = []
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()
    set_session_manager(manager)

    try:
        parent = McpAgent(AgentConfig("parent", model="passthrough", subagents=True))
        await parent.attach_llm(inspecting_factory(created))
        assert install_subagent_tool(parent)
        display = SubagentDisplayRecorder()
        parent.display = display

        result_message = await parent.run_tools(
            Prompt.assistant(
                tool_calls={"call-1": _subagent_call("research this", "  research-pal  ")}
            )
        )

        assert result_message.tool_results is not None
        result = result_message.tool_results["call-1"]
        assert result.isError is False
        assert result.meta is not None
        details = result.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
        assert details["label"] == "research-pal"
        assert details["parent_tool_call_id"] == "call-1"

        assert result_message.channels is not None
        metadata_channel = result_message.channels[FAST_AGENT_TOOL_METADATA]
        assert json.loads(get_text(metadata_channel[0]) or "") == {
            "call-1": {"fast_agent": {"builtin": SUBAGENT_TOOL_NAME}}
        }

        children = manager.list_child_sessions(parent_session)
        assert [child.info.name for child in children] == [details["child_session_id"]]
        text = get_text(result.content[0])
        assert text is not None
        assert display.events == [
            ("user", {"message": "research this", "name": "parent → research-pal"}),
            (
                "assistant",
                {
                    "message": text,
                    "name": "subagent: research-pal",
                    "model": "passthrough",
                    "bottom_items": [f"session {details['child_session_id']}"],
                },
            ),
        ]
        assert not [event for event, _ in display.events if event.startswith("tool_")]
    finally:
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_parallel_subagent_live_display_keeps_generated_call_identity(tmp_path) -> None:
    created: list[InspectingLLM] = []
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()
    set_session_manager(manager)

    try:
        parent = McpAgent(AgentConfig("parent", model="passthrough", subagents=True))
        await parent.attach_llm(inspecting_factory(created))
        assert install_subagent_tool(parent)
        display = SubagentDisplayRecorder()
        parent.display = display

        result_message = await parent.run_tools(
            Prompt.assistant(
                tool_calls={
                    "call-generated": _subagent_call("generated task"),
                    "call-supplied": _subagent_call("supplied task", "reviewer"),
                }
            )
        )

        assert result_message.tool_results is not None
        generated = result_message.tool_results["call-generated"]
        supplied = result_message.tool_results["call-supplied"]
        assert generated.meta is not None
        assert supplied.meta is not None
        generated_details = generated.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
        supplied_details = supplied.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
        assert generated_details["parent_tool_call_id"] == "call-generated"
        assert supplied_details["parent_tool_call_id"] == "call-supplied"
        assert supplied_details["label"] == "reviewer"

        assert result_message.channels is not None
        metadata_channel = result_message.channels[FAST_AGENT_TOOL_METADATA]
        assert json.loads(get_text(metadata_channel[0]) or "") == {
            "call-generated": {"fast_agent": {"builtin": SUBAGENT_TOOL_NAME}},
            "call-supplied": {"fast_agent": {"builtin": SUBAGENT_TOOL_NAME}},
        }

        children = manager.list_child_sessions(parent_session)
        assert {child.info.name for child in children} == {
            generated_details["child_session_id"],
            supplied_details["child_session_id"],
        }
        generated_text = get_text(generated.content[0])
        supplied_text = get_text(supplied.content[0])
        assert generated_text is not None
        assert supplied_text is not None
        assert display.events == [
            ("user", {"message": "generated task", "name": "parent → subagent"}),
            ("user", {"message": "supplied task", "name": "parent → reviewer"}),
            (
                "assistant",
                {
                    "message": generated_text,
                    "name": f"subagent: {generated_details['label']}",
                    "model": "passthrough",
                    "bottom_items": [f"session {generated_details['child_session_id']}"],
                },
            ),
            (
                "assistant",
                {
                    "message": supplied_text,
                    "name": "subagent: reviewer",
                    "model": "passthrough",
                    "bottom_items": [f"session {supplied_details['child_session_id']}"],
                },
            ),
        ]
        assert not [event for event, _ in display.events if event.startswith("tool_")]
    finally:
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_subagent_error_result_uses_assistant_panel(tmp_path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    parent_session = manager.create_session()
    set_session_manager(manager)

    try:
        parent = McpAgent(AgentConfig("parent", model="passthrough", subagents=True))
        await parent.attach_llm(lambda agent, **kwargs: FailingLLM(name=agent.name, **kwargs))
        assert install_subagent_tool(parent)
        display = SubagentDisplayRecorder()
        parent.display = display

        result_message = await parent.run_tools(
            Prompt.assistant(tool_calls={"call-error": _subagent_call("fail", "reviewer")})
        )

        assert result_message.tool_results is not None
        result = result_message.tool_results["call-error"]
        assert result.isError is True
        assert result.meta is not None
        details = result.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
        assert details["status"] == "failed"
        assert details["parent_tool_call_id"] == "call-error"

        children = manager.list_child_sessions(parent_session)
        assert [child.info.name for child in children] == [details["child_session_id"]]
        assert display.events == [
            ("user", {"message": "fail", "name": "parent → reviewer"}),
            (
                "assistant",
                {
                    "message": "Error: simulated failure",
                    "name": "subagent: reviewer",
                    "model": "passthrough",
                    "bottom_items": [f"session {details['child_session_id']}"],
                },
            ),
        ]
        assert not [event for event, _ in display.events if event.startswith("tool_")]
    finally:
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unmanaged_mcp_tool_named_subagent_keeps_generic_display() -> None:
    def subagent(message: str) -> str:
        return f"ordinary tool: {message}"

    agent = McpAgent(
        AgentConfig("unmanaged"),
        context=Context(),
        tools=[subagent],
    )
    display = SubagentDisplayRecorder()
    agent.display = display

    result = await agent.run_tools(
        Prompt.assistant(tool_calls={"ordinary-call": _subagent_call("not a built-in")})
    )

    assert result.tool_results is not None
    assert get_text(result.tool_results["ordinary-call"].content[0]) == (
        "ordinary tool: not a built-in"
    )
    assert display.events == [
        ("tool_call", {"tool_name": SUBAGENT_TOOL_NAME}),
        ("tool_result", {"tool_name": SUBAGENT_TOOL_NAME}),
    ]
    await agent._aggregator.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subagent_monitor_row_reports_turn_usage_and_tool_lifecycle() -> None:
    async def lookup() -> str:
        return "found"

    progress = RecordingProgressDisplay()
    parent = ToolAgent(
        AgentConfig("parent", model="passthrough", subagents=True),
        [lookup],
    )
    await parent.attach_llm(lambda agent, **kwargs: ToolUsingLLM(name=agent.name, **kwargs))
    assert install_subagent_tool(
        parent,
        progress_display=progress,
        label_generator=lambda: "brisk-otter",
    )

    result = await parent.call_tool(
        SUBAGENT_TOOL_NAME,
        {"message": "research"},
        tool_use_id="parent-call",
    )

    assert result.meta is not None
    details = result.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
    assert details["parent_tool_call_id"] == "parent-call"

    events = [event for event in progress.events if event.agent_name == "parent[brisk-otter]"]
    assert "parent[brisk-otter]" in progress._folded_agent_progress
    assert events[0].action == ProgressAction.RUNNING
    assert events[0].target == "brisk-otter"
    assert events[0].details == "turn 0 · starting · in 0 out 0 cache 0 · tools 0"
    assert {event.correlation_id for event in events} == {"parent-call"}
    assert any(
        event.action == ProgressAction.RUNNING
        and event.details is not None
        and "turn 1" in event.details
        and "in 3" in event.details
        and "out 2" in event.details
        and "cache 1" in event.details
        for event in events
    )
    assert any(
        event.action == ProgressAction.RUNNING
        and event.details is not None
        and "tools 1 (lookup)" in event.details
        for event in events
    )
    assert any(
        event.action == ProgressAction.RUNNING
        and event.details is not None
        and "turn 2" in event.details
        and "in 6" in event.details
        and "out 4" in event.details
        and "cache 2" in event.details
        for event in events
    )
    assert events[-1].action == ProgressAction.READY
    assert progress._taskmap == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parallel_subagent_monitor_rows_have_distinct_identity_and_cleanup() -> None:
    progress = RecordingProgressDisplay()
    parent = ToolAgent(AgentConfig("parent", model="passthrough", subagents=True))
    started: asyncio.Queue[str] = asyncio.Queue()
    release = asyncio.Event()
    await parent.attach_llm(
        lambda agent, **kwargs: ParallelBlockingLLM(
            started,
            release,
            name=agent.name,
            **kwargs,
        )
    )
    assert install_subagent_tool(
        parent,
        progress_display=progress,
        label_generator=lambda: "brisk-otter",
    )

    run = asyncio.create_task(
        parent.run_tools(
            Prompt.assistant(
                tool_calls={
                    "call-a": _subagent_call("first task"),
                    "call-b": _subagent_call("second task"),
                }
            )
        )
    )
    assert {await started.get(), await started.get()} == {
        "parent[brisk-otter]",
        "parent[brisk-otter-2]",
    }
    assert set(progress._taskmap) == {
        "parent",
        "parent::subagent::call-a",
        "parent::subagent::call-b",
    }

    release.set()
    result_message = await run

    assert result_message.tool_results is not None
    rows_by_name: dict[str, list[ProgressEvent]] = {}
    for event in progress.events:
        if event.agent_name is not None and event.agent_name != "parent":
            rows_by_name.setdefault(event.agent_name, []).append(event)

    assert set(rows_by_name) == {"parent[brisk-otter]", "parent[brisk-otter-2]"}
    assert {events[0].correlation_id for events in rows_by_name.values()} == {"call-a", "call-b"}
    assert all(events[0].action == ProgressAction.RUNNING for events in rows_by_name.values())
    assert all(
        events[0].details == "turn 0 · starting · in 0 out 0 cache 0 · tools 0"
        for events in rows_by_name.values()
    )
    assert all(events[-1].action == ProgressAction.READY for events in rows_by_name.values())
    parent_events = [event for event in progress.events if event.agent_name == "parent"]
    assert any(
        event.action == ProgressAction.MONITORING
        and event.details == "2 subagents · brisk-otter, brisk-otter-2"
        for event in parent_events
    )
    assert parent_events[-1].action == ProgressAction.READY
    assert progress._taskmap == {}
