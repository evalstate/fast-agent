from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Protocol,
    Union,
)

from mcp.types import ListToolsResult, TextContent

from fast_agent.constants import DEFAULT_MAX_ITERATIONS, FAST_AGENT_ERROR_CHANNEL
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from mcp import Tool


class _AgentConfig(Protocol):
    use_history: bool


class _ToolLoopAgent(Protocol):
    config: _AgentConfig

    async def _tool_runner_llm_step(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended: ...

    async def run_tools(
        self,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended: ...

    async def list_tools(self) -> ListToolsResult: ...


@dataclass(frozen=True)
class ToolRunnerHooks:
    """
    Optional hook points for customizing the tool loop.

    These hooks are intentionally low-level and mutation-friendly: they can
    inspect and modify the agent history (via agent.load_message_history),
    tweak request params, or append extra messages via the runner.
    """

    before_llm_call: (
        Callable[["ToolRunner", list[PromptMessageExtended]], Awaitable[None]] | None
    ) = None
    after_llm_call: Callable[["ToolRunner", PromptMessageExtended], Awaitable[None]] | None = None
    before_tool_call: Callable[["ToolRunner", PromptMessageExtended], Awaitable[None]] | None = None
    after_tool_call: Callable[["ToolRunner", PromptMessageExtended], Awaitable[None]] | None = None


class ToolRunner:
    """
    Async-iterable tool runner.

    Yields assistant messages (LLM responses). If the response requests tools,
    a tool response is prepared and sent on the next iteration.
    """

    def __init__(
        self,
        *,
        agent: _ToolLoopAgent,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        hooks: ToolRunnerHooks | None = None,
    ) -> None:
        self._agent = agent
        self._delta_messages: list[PromptMessageExtended] = list(messages)
        self._request_params = request_params
        self._tools = tools
        self._hooks = hooks or ToolRunnerHooks()

        self._iteration = 0
        self._done = False
        self._last_message: PromptMessageExtended | None = None

        self._pending_tool_request: PromptMessageExtended | None = None
        self._pending_tool_response: PromptMessageExtended | None = None

    def __aiter__(self) -> "ToolRunner":
        return self

    async def __anext__(self) -> PromptMessageExtended:
        if self._done:
            raise StopAsyncIteration

        await self._ensure_tool_response_staged()
        if self._done:
            raise StopAsyncIteration

        await self._ensure_tools_ready()

        if self._hooks.before_llm_call is not None:
            await self._hooks.before_llm_call(self, self._delta_messages)

        assistant_message = await self._agent._tool_runner_llm_step(
            self._delta_messages,
            request_params=self._request_params,
            tools=self._tools,
        )

        self._last_message = assistant_message
        if self._hooks.after_llm_call is not None:
            await self._hooks.after_llm_call(self, assistant_message)

        if assistant_message.stop_reason == LlmStopReason.TOOL_USE:
            self._pending_tool_request = assistant_message
            self._pending_tool_response = None  # Clear cache for new request
        else:
            self._done = True

        return assistant_message

    async def until_done(self) -> PromptMessageExtended:
        last: PromptMessageExtended | None = None
        async for message in self:
            last = message
        if last is None:
            raise RuntimeError("ToolRunner produced no messages")
        return last

    async def generate_tool_call_response(self) -> PromptMessageExtended | None:
        if self._pending_tool_request is None:
            return None
        if self._pending_tool_response is not None:
            return self._pending_tool_response

        if self._hooks.before_tool_call is not None:
            await self._hooks.before_tool_call(self, self._pending_tool_request)

        tool_message = await self._agent.run_tools(
            self._pending_tool_request, request_params=self._request_params
        )
        self._pending_tool_response = tool_message

        if self._hooks.after_tool_call is not None:
            await self._hooks.after_tool_call(self, tool_message)

        self._stage_tool_response(tool_message)
        self._pending_tool_request = None

        return tool_message

    def set_request_params(self, params: RequestParams) -> None:
        self._request_params = params

    def append_messages(self, *messages: Union[str, PromptMessageExtended]) -> None:
        for message in messages:
            if isinstance(message, str):
                self._delta_messages.append(
                    PromptMessageExtended(
                        role="user",
                        content=[TextContent(type="text", text=message)],
                    )
                )
            else:
                self._delta_messages.append(message)

    @property
    def delta_messages(self) -> list[PromptMessageExtended]:
        """Messages to be sent in the next LLM call (not full history)."""
        return self._delta_messages

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def last_message(self) -> PromptMessageExtended | None:
        return self._last_message

    @property
    def has_pending_tool_response(self) -> bool:
        return self._pending_tool_request is not None

    def _stage_tool_response(self, tool_message: PromptMessageExtended) -> None:
        if self._agent.config.use_history:
            self._delta_messages = [tool_message]
        else:
            if self._last_message is not None:
                self._delta_messages.append(self._last_message)
            self._delta_messages.append(tool_message)

    async def _ensure_tools_ready(self) -> None:
        if self._tools is None:
            self._tools = (await self._agent.list_tools()).tools

    async def _ensure_tool_response_staged(self) -> None:
        if self._pending_tool_request is None:
            return

        tool_message = await self.generate_tool_call_response()
        if tool_message is None:
            return

        error_channel_messages = (tool_message.channels or {}).get(FAST_AGENT_ERROR_CHANNEL)
        if error_channel_messages and self._last_message is not None:
            tool_result_contents = [
                content
                for tool_result in (tool_message.tool_results or {}).values()
                for content in tool_result.content
            ]
            if tool_result_contents:
                if self._last_message.content is None:
                    self._last_message.content = []
                self._last_message.content.extend(tool_result_contents)
            self._last_message.stop_reason = LlmStopReason.ERROR
            self._done = True
            return

        self._iteration += 1
        max_iterations = (
            self._request_params.max_iterations
            if self._request_params is not None
            else DEFAULT_MAX_ITERATIONS
        )
        if self._iteration > max_iterations:
            self._done = True
