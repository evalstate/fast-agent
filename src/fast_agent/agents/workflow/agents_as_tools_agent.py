"""
Agents as Tools Pattern Implementation
=======================================

Overview
--------
This module implements the "Agents as Tools" pattern, inspired by OpenAI's Agents SDK
(https://openai.github.io/openai-agents-python/tools). It allows child agents to be
exposed as callable tools to a parent agent, enabling hierarchical agent composition
without the complexity of traditional orchestrator patterns. The current implementation
goes a step further by spawning **detached per-call clones** of every child so that each
parallel execution has its own LLM + MCP stack, eliminating name overrides and shared
state hacks.

Rationale
---------
Traditional approaches to multi-agent systems often require:
1. Complex orchestration logic with explicit routing rules
2. Iterative planning mechanisms that add cognitive overhead
3. Tight coupling between parent and child agent implementations

The "Agents as Tools" pattern simplifies this by:
- **Treating agents as first-class tools**: Each child agent becomes a tool that the
  parent LLM can call naturally via function calling
- **Delegation, not orchestration**: The parent LLM decides which child agents to invoke
  based on its instruction and context, without hardcoded routing logic
- **Parallel execution**: Multiple child agents can run concurrently when the LLM makes
  parallel tool calls
- **Clean abstraction**: Child agents expose tool schemas to the parent LLM. Cards can
  provide a child-owned `tool_input_schema`; otherwise a minimal default schema is used.

Benefits over iterative_planner/orchestrator:
- Simpler codebase: No custom planning loops or routing tables
- Better LLM utilization: Modern LLMs excel at function calling
- Natural composition: Agents nest cleanly without special handling
- Parallel by default: Leverage asyncio.gather for concurrent execution
- Optional concurrency bound: `max_parallel` limits concurrent child calls without
  dropping requested calls

Algorithm
---------
1. **Initialization**
   - `AgentsAsToolsAgent` is itself an `McpAgent` (with its own MCP servers + tools) and receives a list of **child agents**.
   - Each child agent is mapped to a synthetic tool name: `agent__{child_name}`.
   - Child tool schemas come from child cards (`tool_input_schema`) when set;
     otherwise the fallback schema accepts a single `message` string.

2. **Tool Discovery (list_tools)**
   - `list_tools()` starts from the base `McpAgent.list_tools()` (MCP + local tools).
   - Synthetic child tools `agent__ChildName` are added on top when their names do not collide with existing tools.
   - The parent LLM therefore sees a **merged surface**: MCP tools and agent-tools in a single list.

3. **Tool Execution (call_tool)**
   - If the requested tool name resolves to a child agent (either `child_name` or `agent__child_name`):
     - Convert the `message` argument to a child user message.
     - Execute via detached clones created inside `run_tools` (see below).
     - Responses are converted to `CallToolResult` objects (errors propagate as `isError=True`).
   - Otherwise, delegate to the base `McpAgent.call_tool` implementation (MCP tools, shell, human-input, etc.).

4. **Parallel Execution (run_tools)**
   - Collect all tool calls from the parent LLM response.
   - Partition them into **child-agent tools** and **regular MCP/local tools**.
   - Child-agent tools are executed in parallel:
     - For each child tool call, spawn a detached clone with its own LLM + MCP aggregator and suffixed name.
     - Emit `ProgressAction.SENDING` / `ProgressAction.READY` events for each instance and keep parent status untouched.
     - Merge each clone's usage back into the template child after shutdown.
   - Remaining MCP/local tools are delegated to `McpAgent.run_tools()`.
   - Child and MCP results (and their error text from `FAST_AGENT_ERROR_CHANNEL`) are merged into a single `PromptMessageExtended` that is returned to the parent LLM.

Progress Panel Behavior
-----------------------
To provide clear visibility into parallel executions, the progress panel (left status
table) undergoes dynamic updates:

**Before parallel execution:**
```
▎▶ Sending      ▎ PM-1-DayStatusSummarizer     gpt-5 turn 1
```

**During parallel execution (2+ instances):**
- Parent line stays in whatever lifecycle state it already had; no forced "Ready" flips.
- New lines appear for each detached instance with suffixed names:
```
▎▶ Sending      ▎ PM-1-DayStatusSummarizer[1]   gpt-5 turn 2
▎▶ Calling tool  ▎ PM-1-DayStatusSummarizer[2]   tg-ro (list_messages)
```

**Key implementation details:**
- Each clone advertises its own `agent_name` (e.g., `OriginalName[instance_number]`).
- MCP progress events originate from the clone's aggregator, so tool activity always shows under the suffixed name.
- Parent status lines remain visible for context while children run.

**As each instance completes:**
- We emit `ProgressAction.READY` to mark completion, keeping the line in the panel for auditability.
- Other instances continue showing their independent progress until they also finish.

**After all parallel executions complete:**
- Ready instance lines remain until the parent agent moves on, giving a full record of what ran.
- Parent and child template names stay untouched because clones carry the suffixed identity.

- **Instance line visibility**: We now leave finished instance lines visible (marked `READY`)
  instead of hiding them immediately, preserving a full audit trail of parallel runs.
- **Chat log separation**: Each parallel instance gets its own tool request/result headers
  with instance numbers [1], [2], etc. for traceability.

Stats and Usage Semantics
-------------------------
- Each detached clone accrues usage on its own `UsageAccumulator`; after shutdown we
  call `child.merge_usage_from(clone)` so template agents retain consolidated totals.
- Runtime events (logs, MCP progress, chat headers) use the suffixed clone names,
  ensuring per-instance traceability even though usage rolls up to the template.
- The CLI *Usage Summary* table still reports one row per template agent
  (for example, `PM-1-DayStatusSummarizer`), not per `[i]` instance; clones are
  runtime-only and do not appear as separate agents in that table.

**Chat log display:**
Tool headers show instance numbers for clarity:
```
▎▶ orchestrator    [tool request - agent__PM-1-DayStatusSummarizer[1]]
▎◀ orchestrator    [tool result - agent__PM-1-DayStatusSummarizer[1]]
▎▶ orchestrator    [tool request - agent__PM-1-DayStatusSummarizer[2]]
▎◀ orchestrator    [tool result - agent__PM-1-DayStatusSummarizer[2]]
```

Bottom status bar shows all instances:
```
| agent__PM-1-DayStatusSummarizer[1] · running | agent__PM-1-DayStatusSummarizer[2] · running |
```

Implementation Notes
--------------------
- **Instance naming**: `run_tools` computes `instance_name = f"{child.name}[i]"` inside the
  per-call wrapper and passes it into `spawn_detached_instance`, so the template child object
  keeps its original name while each detached clone owns the suffixed identity.
- **Progress event routing**: Because each clone's `MCPAggregator` is constructed with the
  suffixed `agent_name`, all MCP/tool progress events naturally use
  `PM-1-DayStatusSummarizer[i]` without mutating base agent fields or using `ContextVar` hacks.
- **Display suppression with reference counting**: Multiple parallel instances of the same
  child agent share a single agent object. Use reference counting to track active instances:
  - `_display_suppression_count[child_id]`: Count of active parallel instances
  - `_original_display_configs[child_id]`: Stored original config
  - Only modify display config when first instance starts (count 0→1)
  - Only restore display config when last instance completes (count 1→0)
  - Prevents race condition where early-finishing instances restore config while others run
- **Child agents**
  - Existing agents (typically `McpAgent`-based) with their own MCP servers, skills, tools, etc.
  - Serve as **templates**; `run_tools` now clones them before every tool call via
    `spawn_detached_instance`, so runtime work happens inside short-lived replicas.

- **Detached instances**
  - Each tool call gets an actual cloned agent with suffixed name `Child[i]`.
  - Clones own their MCP aggregator/LLM stacks and merge usage back into the template after shutdown.
- **Chat log separation**: Each parallel instance gets its own tool request/result headers
  with instance numbers [1], [2], etc. for traceability

Usage Example
-------------
```python
from fast_agent import FastAgent

fast = FastAgent("parent")

# Define child agents
@fast.agent(name="researcher", instruction="Research topics")
async def researcher(): pass

@fast.agent(name="writer", instruction="Write content")
async def writer(): pass

# Define parent with agents-as-tools
@fast.agent(
    name="coordinator",
    instruction="Coordinate research and writing",
    agents=["researcher", "writer"],  # Exposes children as tools
)
async def coordinator(): pass
```

The parent LLM can now naturally call researcher and writer as tools.

References
----------
- Design doc: ``agetns_as_tools_plan_scratch.md`` (repo root).
- Docs: `docs/agents/workflows.md` (Agents-as-Tools section).
- OpenAI Agents SDK: <https://openai.github.io/openai-agents-python/tools>
- GitHub Issue: [#458](https://github.com/evalstate/fast-agent/issues/458)
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, assert_never

from mcp import ListToolsResult, Tool
from mcp.types import CallToolResult

from fast_agent.acp.tool_call_context import acp_tool_call_context
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.agents.workflow.request_params import child_request_params
from fast_agent.constants import (
    FAST_AGENT_ERROR_CHANNEL,
    FORCE_SEQUENTIAL_TOOL_CALLS,
    should_parallelize_tool_calls,
)
from fast_agent.context import get_current_context
from fast_agent.core.agent_tool_shape import (
    render_agent_tool_arguments,
    resolved_agent_tool_schema,
    response_mode_control_enabled,
    split_response_mode_control,
    tool_result_mode_allows_agent_response_mode,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import ToolRunnerHookCapable
from fast_agent.mcp.helpers.content_helpers import get_text, text_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.session import get_session_manager
from fast_agent.session.identity import (
    SessionSaveContext,
    normalize_session_store_scope,
    resolve_session_for_save,
)
from fast_agent.session.trajectory import (
    TrajectoryRecord,
    new_trajectory_id,
    save_trajectory_record,
)
from fast_agent.tools.invocation_context import agent_tool_invocation_context
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.async_utils import gather_with_cancel

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.llm_agent import LlmAgent
    from fast_agent.agents.tool_runner import ToolRunner
    from fast_agent.llm.request_params import ToolResultMode
    from fast_agent.session.session_manager import Session, SessionManager

logger = get_logger(__name__)

_ChildToolStatus = Literal["pending", "done", "error", "missing"]


@dataclass(slots=True)
class _ChildToolDescriptor:
    id: str
    tool: str
    args: dict[str, Any]
    status: _ChildToolStatus = "pending"
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class _ChildToolResultRecord:
    descriptor: _ChildToolDescriptor
    result: CallToolResult


@dataclass(frozen=True, slots=True)
class _ChildHookInstall:
    installed: bool
    previous_hooks: ToolRunnerHooks | None = None


@dataclass(slots=True)
class _ChildTrajectoryCapture:
    started_at: str
    messages: list[PromptMessageExtended] | None = None


@dataclass(frozen=True, slots=True)
class _ChildInvocationTrace:
    tool_input_schema: dict[str, Any]
    tool_arguments: dict[str, Any]
    effective_tool_arguments: dict[str, Any]
    rendered_child_input: str
    messages: list[PromptMessageExtended] | None
    started_at: str
    completed_at: str


@dataclass(slots=True)
class _ChildToolRunPlan:
    tool_results: dict[str, CallToolResult]
    tool_loop_error: str | None
    call_descriptors: list[_ChildToolDescriptor]
    descriptor_by_id: dict[str, _ChildToolDescriptor]
    id_list: list[str]

def _trajectory_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _resolved_path(raw_path: object | None) -> Path | None:
    if not raw_path:
        return None
    return Path(str(raw_path)).expanduser().resolve()


def _resolve_active_session_manager(
    manager: "SessionManager",
    cwd: Path | None,
) -> "SessionManager":
    if cwd is not None and cwd.resolve() != manager.workspace_dir:
        raise RuntimeError(
            "Trajectory persistence requested a different cwd than the active session manager."
        )
    return manager


class HistorySource(str, Enum):
    """History sources for detached child instances."""

    NONE = "none"
    MESSAGES = "messages"
    CHILD = "child"
    ORCHESTRATOR = "orchestrator"

    @classmethod
    def from_input(cls, value: Any | None) -> HistorySource:
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            valid_values = ", ".join(member.value for member in cls)
            raise ValueError(f"history_source must be one of: {valid_values}") from exc


class HistoryMergeTarget(str, Enum):
    """Merge targets for detached child history."""

    NONE = "none"
    CHILD = "child"
    ORCHESTRATOR = "orchestrator"

    @classmethod
    def from_input(cls, value: Any | None) -> HistoryMergeTarget:
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        if str(value) == "messages":
            raise ValueError(
                "history_merge_target=messages is not supported; use child or orchestrator"
            )
        try:
            return cls(str(value))
        except ValueError as exc:
            valid_values = ", ".join(member.value for member in cls)
            raise ValueError(f"history_merge_target must be one of: {valid_values}") from exc


@dataclass(kw_only=True)
class AgentsAsToolsOptions:
    """Configuration knobs for the Agents-as-Tools wrapper.

    Defaults:
    - history_source: none (child starts with empty history)
    - history_merge_target: none (no merge back)
    - max_parallel: None (no cap; caller may set an explicit limit)
    - child_timeout_sec: None (no per-child timeout)
    - max_display_instances: 20 (show first N lines, collapse the rest)
    """

    history_source: HistorySource = HistorySource.NONE
    history_merge_target: HistoryMergeTarget = HistoryMergeTarget.NONE
    max_parallel: int | None = None
    child_timeout_sec: float | None = None
    max_display_instances: int = 20

    def __post_init__(self) -> None:
        self.history_source = HistorySource.from_input(self.history_source)
        self.history_merge_target = HistoryMergeTarget.from_input(self.history_merge_target)
        if self.max_parallel is not None and self.max_parallel <= 0:
            raise ValueError("max_parallel must be > 0 when set")
        if self.max_display_instances is not None and self.max_display_instances <= 0:
            raise ValueError("max_display_instances must be > 0")
        if self.child_timeout_sec is not None and self.child_timeout_sec <= 0:
            raise ValueError("child_timeout_sec must be > 0 when set")


class AgentsAsToolsAgent(McpAgent):
    """MCP-enabled agent that exposes child agents as additional tools.

    This hybrid agent:

    - Inherits all MCP behavior from :class:`McpAgent` (servers, MCP tool discovery, local tools).
    - Exposes each child agent as an additional synthetic tool (`agent__ChildName`).
    - Merges **MCP tools** and **agent-tools** into a single `list_tools()` surface.
    - Routes `call_tool()` to child agents when the name matches a child, otherwise delegates
      to the base `McpAgent.call_tool` implementation.
    - Overrides `run_tools()` to fan out child-agent tools in parallel using detached clones,
      while delegating any remaining MCP/local tools to the base `McpAgent.run_tools` and
      merging all results into a single tool-loop response.
    """

    def __init__(
        self,
        config: AgentConfig,
        agents: list[LlmAgent],
        options: AgentsAsToolsOptions | None = None,
        context: Any | None = None,
        child_message_files: dict[str, list[Path]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AgentsAsToolsAgent.

        Args:
            config: Agent configuration for this parent agent (including MCP servers/tools)
            agents: List of child agents to expose as tools
            context: Optional context for agent execution
            **kwargs: Additional arguments passed through to :class:`McpAgent` and its bases
        """
        super().__init__(config=config, context=context, **kwargs)
        self._options = options or AgentsAsToolsOptions()
        self._child_agents: dict[str, LlmAgent] = {}
        self._child_message_files = child_message_files or {}
        self._history_merge_lock = asyncio.Lock()
        self._display_suppression_count: dict[int, int] = {}
        self._original_display_logger_settings: dict[int, Any] = {}

        for child in agents:
            tool_name = self._make_tool_name(child.name)
            if tool_name in self._child_agents:
                existing_child = self._child_agents[tool_name]
                raise AgentConfigError(
                    f"Duplicate Agents-as-Tools tool name '{tool_name}' for child agents "
                    f"'{existing_child.name}' and '{child.name}'"
                )
            self._child_agents[tool_name] = child

    def _make_tool_name(self, child_name: str) -> str:
        """Generate a tool name for a child agent.

        Args:
            child_name: Name of the child agent

        Returns:
            Prefixed tool name to avoid collisions with MCP tools
        """
        return f"agent__{child_name}"

    @property
    def agent_backed_tools(self) -> Mapping[str, LlmAgent]:
        """Return all child agents exposed as tool invocations."""
        if not self._agent_tools:
            return self._child_agents
        if not self._child_agents:
            return self._agent_tools
        return {**self._agent_tools, **self._child_agents}

    async def initialize(self) -> None:
        """Initialize this agent and all child agents."""
        if self.initialized:
            return

        await super().initialize()
        for agent in self._child_agents.values():
            if not agent.initialized:
                await agent.initialize()

    async def shutdown(self) -> None:
        """Shutdown this agent and all child agents."""
        await super().shutdown()
        for agent in self._child_agents.values():
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down child agent {agent.name}: {e}")

    def _clone_constructor_kwargs(self) -> dict[str, Any]:
        """Provide kwargs needed to clone this AgentsAsToolsAgent."""
        kwargs = super()._clone_constructor_kwargs()
        kwargs["agents"] = list(self._child_agents.values())
        kwargs["options"] = self._options
        if self._child_message_files:
            kwargs["child_message_files"] = self._child_message_files
        return kwargs

    @staticmethod
    def _child_tool_result_mode(child: LlmAgent) -> ToolResultMode:
        request_params = child.config.default_request_params
        if request_params is None:
            return "postprocess"
        return request_params.tool_result_mode

    @classmethod
    def _child_response_mode_enabled(cls, child: LlmAgent) -> bool:
        return tool_result_mode_allows_agent_response_mode(cls._child_tool_result_mode(child))

    @staticmethod
    def _configured_child_tool_schema(child: LlmAgent) -> dict[str, Any] | None:
        schema = child.config.tool_input_schema
        return schema if isinstance(schema, dict) else None

    @classmethod
    def _child_response_mode_control_enabled(cls, child: LlmAgent) -> bool:
        return response_mode_control_enabled(
            cls._configured_child_tool_schema(child),
            response_mode_enabled=cls._child_response_mode_enabled(child),
        )

    @classmethod
    def _resolved_child_tool_schema(cls, child: LlmAgent) -> dict[str, Any]:
        return resolved_agent_tool_schema(
            cls._configured_child_tool_schema(child),
            response_mode_control=cls._child_response_mode_control_enabled(child),
        )

    @staticmethod
    def _build_child_request_params(
        request_params: RequestParams | None,
        tool_result_mode_override: ToolResultMode | None,
    ) -> RequestParams | None:
        if tool_result_mode_override is not None:
            if request_params is None:
                return RequestParams(tool_result_mode=tool_result_mode_override)
            forwarded_params = child_request_params(request_params)
            if forwarded_params is None:
                return RequestParams(tool_result_mode=tool_result_mode_override)
            return forwarded_params.model_copy(
                update={"tool_result_mode": tool_result_mode_override}
            )

        return child_request_params(request_params)

    async def list_tools(self) -> ListToolsResult:
        """List MCP tools plus child agents exposed as tools."""

        base = await super().list_tools()
        tools = list(base.tools)
        existing_names = {tool.name for tool in tools}

        for tool_name, agent in self._child_agents.items():
            if tool_name in existing_names:
                continue

            description = agent.config.description
            if not description:
                description = agent.instruction

            input_schema = self._resolved_child_tool_schema(agent)
            tools.append(
                Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=input_schema,
                )
            )
            existing_names.add(tool_name)

        return ListToolsResult(tools=tools)

    @contextmanager
    def _child_display_suppressed(self, child: LlmAgent) -> Iterator[None]:
        """Context manager to hide child chat while keeping tool logs visible."""
        child_id = id(child)
        count = self._display_suppression_count.get(child_id, 0)
        if count == 0:
            original_settings = child.display.logger_settings
            self._original_display_logger_settings[child_id] = original_settings
            child.display.update_logger_settings(
                original_settings.model_copy(
                    update={
                        "show_chat": False,
                        "show_tools": True,
                    }
                )
            )
        self._display_suppression_count[child_id] = count + 1
        try:
            yield
        finally:
            self._display_suppression_count[child_id] -= 1
            if self._display_suppression_count[child_id] <= 0:
                del self._display_suppression_count[child_id]
                original_settings = self._original_display_logger_settings.pop(
                    child_id,
                    None,
                )
                if original_settings is not None:
                    child.display.update_logger_settings(original_settings)

    async def _merge_history(self, target: LlmAgent, clone: LlmAgent, start_index: int) -> None:
        """Append clone history from start_index into target with a global merge lock."""
        async with self._history_merge_lock:
            new_messages = clone.message_history[start_index:]
            target.append_history(new_messages)

    def _history_for_source(
        self,
        child: LlmAgent,
        source: HistorySource,
    ) -> list[PromptMessageExtended]:
        match source:
            case HistorySource.NONE:
                return []
            case HistorySource.MESSAGES:
                return self._load_child_message_history(child.name)
            case HistorySource.CHILD:
                return list(child.message_history)
            case HistorySource.ORCHESTRATOR:
                return list(self.message_history)
            case _:
                assert_never(source)

    async def _merge_history_for_target(
        self,
        *,
        target: HistoryMergeTarget,
        child: LlmAgent,
        clone: LlmAgent,
        start_index: int,
    ) -> None:
        match target:
            case HistoryMergeTarget.NONE:
                return
            case HistoryMergeTarget.CHILD:
                await self._merge_history(target=child, clone=clone, start_index=start_index)
            case HistoryMergeTarget.ORCHESTRATOR:
                await self._merge_history(target=self, clone=clone, start_index=start_index)
            case _:
                assert_never(target)

    def _load_child_message_history(self, child_name: str) -> list[PromptMessageExtended]:
        message_files = self._child_message_files.get(child_name, [])
        if not message_files:
            return []
        messages: list[PromptMessageExtended] = []
        for path in message_files:
            try:
                messages.extend(load_prompt(path))
            except Exception as exc:
                logger.warning(
                    "Failed to load child message history",
                    data={"agent_name": child_name, "path": str(path), "error": str(exc)},
                )
        return messages

    def _child_input_text(self, child: LlmAgent, args: dict[str, Any]) -> str:
        return render_agent_tool_arguments(
            args,
            configured_schema=self._configured_child_tool_schema(child),
            response_mode_control=self._child_response_mode_control_enabled(child),
        )

    @staticmethod
    def _child_tool_result_from_response(
        response: PromptMessageExtended,
    ) -> tuple[CallToolResult, list[Any] | None]:
        content_blocks = list(response.content or [])
        error_blocks = None
        if response.channels and FAST_AGENT_ERROR_CHANNEL in response.channels:
            error_blocks = list(response.channels.get(FAST_AGENT_ERROR_CHANNEL) or [])
            if error_blocks:
                content_blocks.extend(error_blocks)

        if not content_blocks and response.stop_reason == LlmStopReason.TOOL_USE:
            error_blocks = [
                text_content(
                    "Runtime budget exhausted: child agent stopped while requesting another tool "
                    "and produced no final content. Return a best-effort answer or retry with a "
                    "larger budget."
                )
            ]
            content_blocks.extend(error_blocks)

        return (
            CallToolResult(
                content=content_blocks,
                isError=bool(error_blocks),
            ),
            error_blocks,
        )

    @staticmethod
    async def _complete_child_tool_call(
        *,
        tool_handler: Any,
        tool_call_id: str | None,
        tool_result: CallToolResult,
        error_blocks: list[Any] | None,
    ) -> None:
        if not tool_handler or not tool_call_id:
            return
        try:
            if tool_result.isError:
                error_text = get_text(error_blocks[0]) if error_blocks else None
                with acp_tool_call_context():
                    await tool_handler.on_tool_complete(
                        tool_call_id,
                        False,
                        None,
                        error_text,
                    )
                return

            with acp_tool_call_context():
                await tool_handler.on_tool_complete(
                    tool_call_id,
                    True,
                    tool_result.content,
                    None,
                )
        except Exception:
            pass

    @staticmethod
    async def _fail_child_tool_call(
        *,
        tool_handler: Any,
        tool_call_id: str | None,
        error: str,
    ) -> None:
        if not tool_handler or not tool_call_id:
            return
        try:
            with acp_tool_call_context():
                await tool_handler.on_tool_complete(tool_call_id, False, None, error)
        except Exception:
            pass

    @staticmethod
    async def _start_child_tool_call(
        *,
        tool_handler: Any,
        child: LlmAgent,
        args: dict[str, Any],
        tool_use_id: str | None,
    ) -> str | None:
        if not tool_handler:
            return None
        try:
            with acp_tool_call_context():
                return await tool_handler.on_tool_start(
                    child.name,
                    "agent",
                    args,
                    tool_use_id,
                )
        except Exception:
            return None

    @staticmethod
    async def _emit_child_tool_progress(
        *,
        tool_handler: Any,
        tool_call_id: str | None,
        progress_step: int,
    ) -> int:
        if not tool_handler or not tool_call_id:
            return progress_step
        next_step = progress_step + 1
        try:
            # Title already includes the agent instance name; keep progress updates minimal.
            # The progress counter itself is shown as `[N]` in ACP tool titles.
            with acp_tool_call_context():
                await tool_handler.on_tool_progress(
                    tool_call_id,
                    float(next_step),
                    None,
                    None,
                )
        except Exception:
            pass
        return next_step

    async def _install_child_progress_hooks(
        self,
        *,
        child: LlmAgent,
        tool_handler: Any,
        tool_call_id: str | None,
        emit_progress: Callable[[], Any],
        trajectory_capture: _ChildTrajectoryCapture | None = None,
    ) -> _ChildHookInstall:
        capture_enabled = trajectory_capture is not None
        progress_enabled = bool(tool_handler and tool_call_id)
        if not (progress_enabled or capture_enabled) or not isinstance(
            child, ToolRunnerHookCapable
        ):
            return _ChildHookInstall(installed=False)

        previous_hooks = child.tool_runner_hooks
        before_llm_call = previous_hooks.before_llm_call if previous_hooks else None
        before_tool_call = previous_hooks.before_tool_call if previous_hooks else None
        after_llm_call = previous_hooks.after_llm_call if previous_hooks else None
        after_tool_call = previous_hooks.after_tool_call if previous_hooks else None
        after_turn_complete = previous_hooks.after_turn_complete if previous_hooks else None

        async def handle_before_llm_call(runner, messages):
            if before_llm_call:
                await before_llm_call(runner, messages)
            if progress_enabled:
                await emit_progress()

        async def handle_before_tool_call(runner, message):
            if before_tool_call:
                await before_tool_call(runner, message)
            if progress_enabled:
                await emit_progress()

        async def handle_after_turn_complete(
            runner: "ToolRunner",
            message: PromptMessageExtended,
        ) -> None:
            if after_turn_complete:
                await after_turn_complete(runner, message)
            if trajectory_capture is None:
                return
            trajectory_capture.messages = [
                *[item.model_copy(deep=True) for item in runner.delta_messages],
                message.model_copy(deep=True),
            ]

        child.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=handle_before_llm_call,
            after_llm_call=after_llm_call,
            before_tool_call=handle_before_tool_call,
            after_tool_call=after_tool_call,
            after_turn_complete=handle_after_turn_complete,
        )
        return _ChildHookInstall(installed=True, previous_hooks=previous_hooks)

    async def _generate_child_tool_response(
        self,
        *,
        child: LlmAgent,
        child_request: PromptMessageExtended,
        child_request_params: RequestParams | None,
        child_arguments: dict[str, Any],
        child_tool_name: str | None,
        suppress_display: bool,
        tool_handler: Any,
        tool_call_id: str | None,
        hooks_set: bool,
        emit_progress: Callable[[], Any],
    ) -> PromptMessageExtended:
        scope = (
            acp_tool_call_context(parent_tool_call_id=tool_call_id)
            if tool_handler and tool_call_id
            else acp_tool_call_context()
        )
        display_scope = self._child_display_suppressed(child) if suppress_display else nullcontext()
        invocation_scope = agent_tool_invocation_context(
            agent_name=child.name,
            arguments=child_arguments,
            tool_name=child_tool_name,
            tool_use_id=tool_call_id,
        )
        with scope, display_scope, invocation_scope:
            if tool_handler and tool_call_id and not hooks_set:
                await emit_progress()
            return await child.generate(
                [child_request],
                request_params=child_request_params,
            )

    async def _invoke_child_agent(
        self,
        child: LlmAgent,
        arguments: dict[str, Any] | None = None,
        *,
        suppress_display: bool = True,
        tool_name: str | None = None,
        tool_use_id: str | None = None,
        request_params: RequestParams | None = None,
        trace_sink: list[_ChildInvocationTrace] | None = None,
    ) -> CallToolResult:
        """Shared helper to execute a child agent with standard serialization and display rules."""

        raw_args = arguments or {}
        response_mode_control = split_response_mode_control(
            raw_args,
            enabled=self._child_response_mode_control_enabled(child),
        )
        if response_mode_control.error is not None:
            return CallToolResult(
                content=[text_content(response_mode_control.error)],
                isError=True,
            )
        child_request_params = self._build_child_request_params(
            request_params,
            response_mode_control.tool_result_mode_override,
        )
        args = response_mode_control.arguments
        rendered_child_input = self._child_input_text(child, args)
        child_request = Prompt.user(rendered_child_input)
        trajectory_capture = (
            _ChildTrajectoryCapture(started_at=_trajectory_timestamp())
            if trace_sink is not None
            else None
        )

        tool_handler = self._get_tool_handler(request_params)
        progress_step = 0
        tool_call_id = await self._start_child_tool_call(
            tool_handler=tool_handler,
            child=child,
            args=args,
            tool_use_id=tool_use_id,
        )

        async def emit_progress() -> None:
            nonlocal progress_step
            progress_step = await self._emit_child_tool_progress(
                tool_handler=tool_handler,
                tool_call_id=tool_call_id,
                progress_step=progress_step,
            )

        hook_install = await self._install_child_progress_hooks(
            child=child,
            tool_handler=tool_handler,
            tool_call_id=tool_call_id,
            emit_progress=emit_progress,
            trajectory_capture=trajectory_capture,
        )
        hooks_set = hook_install.installed

        try:
            response = await self._generate_child_tool_response(
                child=child,
                child_request=child_request,
                child_request_params=child_request_params,
                child_arguments=args,
                child_tool_name=tool_name,
                suppress_display=suppress_display,
                tool_handler=tool_handler,
                tool_call_id=tool_call_id,
                hooks_set=hooks_set,
                emit_progress=emit_progress,
            )
            tool_result, error_blocks = self._child_tool_result_from_response(response)
            await self._complete_child_tool_call(
                tool_handler=tool_handler,
                tool_call_id=tool_call_id,
                tool_result=tool_result,
                error_blocks=error_blocks,
            )
            if trace_sink is not None and trajectory_capture is not None:
                trace_sink.append(
                    _ChildInvocationTrace(
                        tool_input_schema=self._resolved_child_tool_schema(child),
                        tool_arguments=deepcopy(raw_args),
                        effective_tool_arguments=deepcopy(args),
                        rendered_child_input=rendered_child_input,
                        messages=trajectory_capture.messages,
                        started_at=trajectory_capture.started_at,
                        completed_at=_trajectory_timestamp(),
                    )
                )
            return tool_result
        except asyncio.CancelledError:
            await self._fail_child_tool_call(
                tool_handler=tool_handler,
                tool_call_id=tool_call_id,
                error="Child agent tool call cancelled",
            )
            raise
        except Exception as exc:
            import traceback

            logger.error(
                "Child agent tool call failed",
                data={
                    "agent_name": child.name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            await self._fail_child_tool_call(
                tool_handler=tool_handler,
                tool_call_id=tool_call_id,
                error=str(exc),
            )
            return CallToolResult(content=[text_content(f"Error: {exc}")], isError=True)
        finally:
            if hooks_set and isinstance(child, ToolRunnerHookCapable):
                child.tool_runner_hooks = hook_install.previous_hooks

    def _resolve_child_agent(self, name: str) -> LlmAgent | None:
        return self._child_agents.get(name) or self._child_agents.get(self._make_tool_name(name))

    async def _base_tool_names(self) -> set[str]:
        base = await super().list_tools()
        return {tool.name for tool in base.tools}

    async def _resolve_child_agent_for_call(self, name: str) -> LlmAgent | None:
        if name in await self._base_tool_names():
            return None
        return self._resolve_child_agent(name)

    async def _available_child_tool_names(self) -> set[str]:
        try:
            listed = await self.list_tools()
            return {tool.name for tool in listed.tools}
        except Exception as exc:
            logger.warning(f"Failed to list tools before execution: {exc}")
            return set(self._child_agents.keys())

    def _child_tool_is_available(self, tool_name: str, available_tools: set[str]) -> bool:
        return tool_name in available_tools or self._make_tool_name(tool_name) in available_tools

    async def _build_child_tool_run_plan(
        self,
        *,
        request: PromptMessageExtended,
        target_ids: set[str],
    ) -> _ChildToolRunPlan:
        available_tools = await self._available_child_tool_names()
        tool_results: dict[str, CallToolResult] = {}
        call_descriptors: list[_ChildToolDescriptor] = []
        descriptor_by_id: dict[str, _ChildToolDescriptor] = {}
        id_list: list[str] = []
        tool_loop_error: str | None = None

        for correlation_id, tool_request in (request.tool_calls or {}).items():
            if correlation_id not in target_ids:
                continue

            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}
            descriptor = _ChildToolDescriptor(
                id=correlation_id,
                tool=tool_name,
                args=tool_args,
            )
            call_descriptors.append(descriptor)
            descriptor_by_id[correlation_id] = descriptor

            if self._child_tool_is_available(tool_name, available_tools):
                descriptor.status = "pending"
                id_list.append(correlation_id)
                continue

            error_message = f"Tool '{tool_name}' is not available"
            tool_results[correlation_id] = CallToolResult(
                content=[text_content(error_message)], isError=True
            )
            tool_loop_error = tool_loop_error or error_message
            descriptor.status = "error"
            descriptor.error_message = error_message

        return _ChildToolRunPlan(
            tool_results=tool_results,
            tool_loop_error=tool_loop_error,
            call_descriptors=call_descriptors,
            descriptor_by_id=descriptor_by_id,
            id_list=id_list,
        )

    async def _run_child_tool_clone(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        instance: int,
        correlation_id: str,
        request_params: RequestParams | None,
    ) -> CallToolResult:
        child = self._resolve_child_agent(tool_name)
        if not child:
            error_msg = f"Unknown agent-tool: {tool_name}"
            return CallToolResult(content=[text_content(error_msg)], isError=True)

        instance_name = f"{child.name}[{instance}]"
        try:
            clone = await child.spawn_detached_instance(name=instance_name)
        except Exception as exc:
            logger.error(
                "Failed to spawn dedicated child instance",
                data={
                    "tool_name": tool_name,
                    "agent_name": child.name,
                    "error": str(exc),
                },
            )
            return CallToolResult(content=[text_content(f"Spawn failed: {exc}")], isError=True)

        fork_index = self._load_history_into_clone(child, clone, instance_name)
        progress_started = self._start_child_clone_progress(
            instance_name=instance_name,
            correlation_id=correlation_id,
            tool_name=tool_name,
        )
        trace_sink: list[_ChildInvocationTrace] | None = (
            [] if child.config.save_trajectory else None
        )
        try:
            call_coro = self._invoke_child_agent(
                clone,
                tool_args,
                tool_name=tool_name,
                tool_use_id=correlation_id,
                request_params=request_params,
                trace_sink=trace_sink,
            )
            timeout = self._options.child_timeout_sec
            if timeout:
                result = await asyncio.wait_for(call_coro, timeout=timeout)
            else:
                result = await call_coro
            await self._save_child_trajectory(
                trace_sink=trace_sink,
                child=child,
                clone=clone,
                instance_name=instance_name,
                tool_name=tool_name,
                correlation_id=correlation_id,
            )
            return result
        finally:
            await self._cleanup_child_clone(
                child=child,
                clone=clone,
                instance_name=instance_name,
                fork_index=fork_index,
                progress_started=progress_started,
                correlation_id=correlation_id,
                tool_name=tool_name,
            )

    async def _save_child_trajectory(
        self,
        *,
        trace_sink: list[_ChildInvocationTrace] | None,
        child: LlmAgent,
        clone: LlmAgent,
        instance_name: str,
        tool_name: str,
        correlation_id: str,
    ) -> None:
        if trace_sink is None or not trace_sink:
            return
        trace = trace_sink[-1]
        if trace.messages is None:
            return
        try:
            session = self._resolve_trajectory_session(child.name)
            await save_trajectory_record(
                session,
                TrajectoryRecord(
                    trajectory_id=new_trajectory_id(),
                    session_id=session.info.name,
                    parent_agent_name=self.name,
                    agent_name=instance_name,
                    template_agent_name=child.name,
                    tool_name=tool_name,
                    parent_tool_call_id=correlation_id,
                    use_history=child.config.use_history,
                    started_at=trace.started_at,
                    completed_at=trace.completed_at,
                    tool_input_schema=trace.tool_input_schema,
                    tool_arguments=trace.tool_arguments,
                    effective_tool_arguments=trace.effective_tool_arguments,
                    rendered_child_input=trace.rendered_child_input,
                    messages=trace.messages,
                    usage_summary=(
                        clone.usage_accumulator.get_summary()
                        if clone.usage_accumulator is not None
                        else None
                    ),
                ),
            )
        except Exception as exc:
            logger.warning(
                "Failed to save child trajectory",
                data={
                    "parent_agent_name": self.name,
                    "agent_name": instance_name,
                    "tool_name": tool_name,
                    "correlation_id": correlation_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    def _resolve_trajectory_session(self, child_name: str) -> "Session":
        current_context = get_current_context()
        agent_context = self.context
        acp_context = agent_context.acp if agent_context else None
        session_store_scope = normalize_session_store_scope(
            getattr(acp_context, "session_store_scope", "workspace")
            if acp_context is not None
            else "workspace"
        )
        session_store_cwd = _resolved_path(
            getattr(acp_context, "session_store_cwd", None) if acp_context is not None else None
        )
        session_cwd = _resolved_path(
            getattr(acp_context, "session_cwd", None) if acp_context is not None else None
        )
        manager = agent_context.session_manager if agent_context else current_context.session_manager
        if manager is None:
            manager = get_session_manager()
        identity = resolve_session_for_save(
            current_session=None,
            get_manager=lambda cwd: _resolve_active_session_manager(manager, cwd),
            context=SessionSaveContext(
                acp_session_id=(
                    getattr(acp_context, "session_id", None) if acp_context is not None else None
                ),
                session_cwd=session_cwd,
                session_store_scope=session_store_scope,
                session_store_cwd=session_store_cwd,
            ),
            seed_metadata={
                "agent_name": self.name,
                "trajectory_agent_name": child_name,
            },
        )
        if current_context is not None and current_context.config is not None:
            app_name = getattr(current_context.config, "name", None)
            if app_name:
                identity.session.info.metadata.setdefault("application", app_name)
        return identity.session

    def _load_history_into_clone(
        self,
        child: LlmAgent,
        clone: LlmAgent,
        instance_name: str,
    ) -> int:
        try:
            base_history = self._history_for_source(child, self._options.history_source)
            clone.load_message_history(base_history)
            return len(base_history)
        except Exception as hist_exc:
            logger.warning(
                "Failed to load history into clone",
                data={"instance_name": instance_name, "error": str(hist_exc)},
            )
            return 0

    @staticmethod
    def _start_child_clone_progress(
        *,
        instance_name: str,
        correlation_id: str,
        tool_name: str,
    ) -> bool:
        from fast_agent.event_progress import ProgressAction, ProgressEvent
        from fast_agent.ui.progress_display import (
            progress_display as outer_progress_display,
        )

        outer_progress_display.update(
            ProgressEvent(
                action=ProgressAction.SENDING,
                target=instance_name,
                details="",
                agent_name=instance_name,
                correlation_id=correlation_id,
                instance_name=instance_name,
                tool_name=tool_name,
            )
        )
        return True

    @staticmethod
    def _finish_child_clone_progress(
        *,
        instance_name: str,
        correlation_id: str,
        tool_name: str,
    ) -> None:
        from fast_agent.event_progress import ProgressAction, ProgressEvent
        from fast_agent.ui.progress_display import (
            progress_display as outer_progress_display,
        )

        outer_progress_display.update(
            ProgressEvent(
                action=ProgressAction.READY,
                target=instance_name,
                details=None,
                agent_name=instance_name,
                correlation_id=correlation_id,
                instance_name=instance_name,
                tool_name=tool_name,
            )
        )

    async def _cleanup_child_clone(
        self,
        *,
        child: LlmAgent,
        clone: LlmAgent,
        instance_name: str,
        fork_index: int,
        progress_started: bool,
        correlation_id: str,
        tool_name: str,
    ) -> None:
        try:
            await clone.shutdown()
        except Exception as shutdown_exc:
            logger.warning(
                "Error shutting down dedicated child instance",
                data={"instance_name": instance_name, "error": str(shutdown_exc)},
            )
        try:
            child.merge_usage_from(clone)
        except Exception as merge_exc:
            logger.warning(
                "Failed to merge usage from child instance",
                data={"instance_name": instance_name, "error": str(merge_exc)},
            )
        try:
            await self._merge_history_for_target(
                target=self._options.history_merge_target,
                child=child,
                clone=clone,
                start_index=fork_index,
            )
        except Exception as merge_hist_exc:
            logger.warning(
                "Failed to merge child history",
                data={
                    "instance_name": instance_name,
                    "target": self._options.history_merge_target.value,
                    "error": str(merge_hist_exc),
                },
            )
        if progress_started and instance_name:
            self._finish_child_clone_progress(
                instance_name=instance_name,
                correlation_id=correlation_id,
                tool_name=tool_name,
            )

    def _close_streaming_for_parallel_child_tools(self, tool_call_count: int) -> None:
        if tool_call_count <= 1:
            return
        did_close = self.close_active_streaming_display(reason="parallel tool calls")
        if did_close:
            logger.info(
                "Closing streaming display due to parallel subagent tool calls",
                tool_call_count=tool_call_count,
                agent_name=self.name,
            )

    async def _execute_child_tool_calls(
        self,
        *,
        plan: _ChildToolRunPlan,
        request_params: RequestParams | None,
    ) -> list[CallToolResult | BaseException]:
        if not plan.id_list:
            return []
        if FORCE_SEQUENTIAL_TOOL_CALLS:
            return await self._execute_child_tool_calls_sequential(
                plan=plan,
                request_params=request_params,
            )
        return await self._execute_child_tool_calls_parallel(
            plan=plan,
            request_params=request_params,
        )

    async def _execute_child_tool_calls_sequential(
        self,
        *,
        plan: _ChildToolRunPlan,
        request_params: RequestParams | None,
    ) -> list[CallToolResult | BaseException]:
        results: list[CallToolResult | BaseException] = []
        for instance, correlation_id in enumerate(plan.id_list, 1):
            descriptor = plan.descriptor_by_id[correlation_id]
            try:
                results.append(
                    await self._run_child_tool_clone(
                        tool_name=descriptor.tool,
                        tool_args=descriptor.args,
                        instance=instance,
                        correlation_id=correlation_id,
                        request_params=request_params,
                    )
                )
            except Exception as exc:
                results.append(exc)
        return results

    async def _execute_child_tool_calls_parallel(
        self,
        *,
        plan: _ChildToolRunPlan,
        request_params: RequestParams | None,
    ) -> list[CallToolResult | BaseException]:
        semaphore = (
            asyncio.Semaphore(self._options.max_parallel) if self._options.max_parallel else None
        )

        async def bounded_call(
            descriptor: _ChildToolDescriptor,
            instance: int,
        ) -> CallToolResult:
            if semaphore is None:
                return await self._run_child_tool_clone(
                    tool_name=descriptor.tool,
                    tool_args=descriptor.args,
                    instance=instance,
                    correlation_id=descriptor.id,
                    request_params=request_params,
                )
            async with semaphore:
                return await self._run_child_tool_clone(
                    tool_name=descriptor.tool,
                    tool_args=descriptor.args,
                    instance=instance,
                    correlation_id=descriptor.id,
                    request_params=request_params,
                )

        return await gather_with_cancel(
            bounded_call(plan.descriptor_by_id[correlation_id], instance)
            for instance, correlation_id in enumerate(plan.id_list, 1)
        )

    @staticmethod
    def _merge_child_tool_execution_results(
        *,
        plan: _ChildToolRunPlan,
        results: list[CallToolResult | BaseException],
    ) -> None:
        for index, result in enumerate(results):
            correlation_id = plan.id_list[index]
            descriptor = plan.descriptor_by_id[correlation_id]
            if isinstance(result, BaseException):
                msg = f"Tool execution failed: {result}"
                plan.tool_results[correlation_id] = CallToolResult(
                    content=[text_content(msg)], isError=True
                )
                plan.tool_loop_error = plan.tool_loop_error or msg
                descriptor.status = "error"
                descriptor.error_message = msg
                continue

            plan.tool_results[correlation_id] = result
            descriptor.status = "error" if result.isError else "done"

    @staticmethod
    def _ordered_child_tool_records(
        plan: _ChildToolRunPlan,
    ) -> list[_ChildToolResultRecord]:
        ordered_records: list[_ChildToolResultRecord] = []
        for correlation_id in plan.id_list:
            result = plan.tool_results.get(correlation_id)
            if result is None:
                continue
            descriptor = plan.descriptor_by_id[correlation_id]
            ordered_records.append(_ChildToolResultRecord(descriptor=descriptor, result=result))
        return ordered_records

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult:
        """Route tool execution to child agents first, then MCP/local tools.

        The signature matches :meth:`McpAgent.call_tool` so that upstream tooling
        can safely pass the LLM's ``tool_use_id`` as a positional argument.
        """

        child = await self._resolve_child_agent_for_call(name)
        if child is not None:
            # Child agents don't currently use tool_use_id, they operate via
            # a plain PromptMessageExtended tool call.
            return await self._invoke_child_agent(
                child,
                arguments,
                tool_name=name,
                tool_use_id=tool_use_id,
                request_params=request_params,
            )

        return await super().call_tool(name, arguments, tool_use_id, request_params=request_params)

    def _show_parallel_tool_calls(
        self,
        descriptors: list[_ChildToolDescriptor],
        *,
        show_tool_call_id: bool = False,
    ) -> None:
        """Display tool call headers for parallel agent execution.

        Args:
            descriptors: List of tool call descriptors with metadata
        """
        if not descriptors:
            return

        status_labels = {
            "pending": "running",
            "error": "error",
            "missing": "missing",
        }

        total = len(descriptors)
        limit = self._options.max_display_instances or total

        # Show detailed call information for each agent
        for desc in descriptors[:limit]:
            tool_name = desc.tool
            corr_id = desc.id
            args = desc.args
            status = desc.status

            if status == "error":
                continue  # Skip display for error tools, will show in results

            base_tool_name = tool_name.removeprefix("agent__")
            display_tool_name = base_tool_name

            # Build bottom item for THIS instance only (not all instances)
            status_label = status_labels.get(status, "pending")
            bottom_item = f"{display_tool_name} · {status_label}"

            # Show individual tool call with arguments
            self.display.show_tool_call(
                name=self.name,
                tool_name=display_tool_name,
                tool_args=args,
                bottom_items=[bottom_item],  # Only this instance's label
                max_item_length=28,
                metadata={"correlation_id": corr_id, "instance_name": display_tool_name},
                tool_call_id=corr_id if show_tool_call_id else None,
                type_label="subagent",
                show_hook_indicator=self.has_external_hooks,
            )
        if total > limit:
            collapsed = total - limit
            label = f"[{limit + 1}..{total}]"
            self.display.show_tool_call(
                name=self.name,
                tool_name=label,
                tool_args={"collapsed": collapsed},
                bottom_items=[f"{label} · {collapsed} more"],
                max_item_length=28,
                type_label="subagent",
                show_hook_indicator=self.has_external_hooks,
            )

    def _show_parallel_tool_results(
        self,
        records: list[_ChildToolResultRecord],
        *,
        show_tool_call_id: bool = False,
    ) -> None:
        """Display tool result panels for parallel agent execution.

        Args:
            records: List of result records with descriptor and result data
        """
        if not records:
            return

        total = len(records)
        limit = self._options.max_display_instances or total

        # Show detailed result for each agent
        for record in records[:limit]:
            descriptor = record.descriptor
            result = record.result
            tool_name = descriptor.tool
            corr_id = descriptor.id

            if result:
                base_tool_name = tool_name.removeprefix("agent__")
                display_tool_name = base_tool_name

                # Show individual tool result with full content
                self.display.show_tool_result(
                    name=self.name,
                    tool_name=display_tool_name,
                    type_label="subagent response",
                    result=result,
                    tool_call_id=corr_id if show_tool_call_id else None,
                    show_hook_indicator=self.has_external_hooks,
                )
        if total > limit:
            collapsed = total - limit
            label = f"[{limit + 1}..{total}]"
            self.display.show_tool_result(
                name=self.name,
                tool_name=label,
                type_label="subagent response",
                show_hook_indicator=self.has_external_hooks,
                result=CallToolResult(
                    content=[text_content(f"{collapsed} more results (collapsed)")],
                    isError=False,
                ),
            )

    async def run_tools(
        self,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
        """Handle mixed MCP + agent-tool batches."""

        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        child_ids: list[str] = []
        base_tool_names = await self._base_tool_names()
        for correlation_id, tool_request in request.tool_calls.items():
            tool_name = tool_request.params.name
            if tool_name not in base_tool_names and self._resolve_child_agent(tool_name):
                child_ids.append(correlation_id)

        if not child_ids:
            return await super().run_tools(request, request_params=request_params)

        child_results, child_error = await self._run_child_tools(
            request,
            set(child_ids),
            request_params=request_params,
        )

        if len(child_ids) == len(request.tool_calls):
            return self._finalize_tool_results(child_results, tool_loop_error=child_error)

        # Execute remaining MCP/local tools via base implementation
        remaining_ids = [cid for cid in request.tool_calls if cid not in child_ids]
        mcp_request = PromptMessageExtended(
            role=request.role,
            content=request.content,
            tool_calls={cid: request.tool_calls[cid] for cid in remaining_ids},
        )
        mcp_message = await super().run_tools(mcp_request, request_params=request_params)
        mcp_results = mcp_message.tool_results or {}
        mcp_error = self._extract_error_text(mcp_message)

        combined_results = {
            correlation_id: result
            for correlation_id in request.tool_calls
            if (result := child_results.get(correlation_id) or mcp_results.get(correlation_id))
            is not None
        }

        tool_loop_error = child_error or mcp_error
        return self._finalize_tool_results(combined_results, tool_loop_error=tool_loop_error)

    async def _run_child_tools(
        self,
        request: PromptMessageExtended,
        target_ids: set[str],
        request_params: RequestParams | None = None,
    ) -> tuple[dict[str, CallToolResult], str | None]:
        """Run only the child-agent tool calls from the request."""

        if not target_ids:
            return {}, None

        plan = await self._build_child_tool_run_plan(
            request=request,
            target_ids=target_ids,
        )
        show_tool_call_id = should_parallelize_tool_calls(len(plan.id_list))
        self._close_streaming_for_parallel_child_tools(len(plan.id_list))

        self._show_parallel_tool_calls(
            plan.call_descriptors,
            show_tool_call_id=show_tool_call_id,
        )

        results = await self._execute_child_tool_calls(
            plan=plan,
            request_params=request_params,
        )
        self._merge_child_tool_execution_results(plan=plan, results=results)
        self._show_parallel_tool_results(
            self._ordered_child_tool_records(plan),
            show_tool_call_id=show_tool_call_id,
        )

        return plan.tool_results, plan.tool_loop_error

    def _extract_error_text(self, message: PromptMessageExtended) -> str | None:
        if not message.channels:
            return None

        error_blocks = message.channels.get(FAST_AGENT_ERROR_CHANNEL)
        if not error_blocks:
            return None

        for block in error_blocks:
            text = get_text(block)
            if text:
                return text

        return None
