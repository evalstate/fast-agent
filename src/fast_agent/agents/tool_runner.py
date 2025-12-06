"""
Iterable tool execution loop for fine-grained control over agent tool calls.

This module provides `ToolRunner`, an async-iterable class that yields each
message from Claude during a tool-calling loop. It allows callers to:
- Observe intermediate messages as they're generated
- Break out of the loop early
- Customize parameters between iterations
- Manually handle tool responses before they're sent back

Inspired by the Anthropic SDK's tool_runner() pattern.
"""

from typing import TYPE_CHECKING, AsyncIterator, Callable, List

from mcp.types import Tool

from fast_agent.constants import DEFAULT_MAX_ITERATIONS, FAST_AGENT_ERROR_CHANNEL
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class ToolRunner:
    """
    An async-iterable tool execution loop.

    Each iteration yields a `PromptMessageExtended` from Claude. Between iterations,
    the runner automatically handles tool calls unless you intervene.

    Example:
        ```python
        runner = agent.tool_runner(messages=[...])

        # Option 1: Iterate manually
        async for message in runner:
            print(f"Claude said: {message.first_text()}")
            if should_stop:
                break

        # Option 2: Just get the final result
        final = await runner.until_done()
        ```

    Advanced usage:
        ```python
        async for message in runner:
            # Get the tool response before it's sent
            tool_response = await runner.generate_tool_call_response()
            if tool_response:
                print(f"Tool results: {tool_response.tool_results}")

            # Modify next request
            runner.set_request_params(lambda p: RequestParams(**(p.model_dump() | {"maxTokens": 4096})))

            # Add guidance for next turn
            runner.append_messages(
                PromptMessageExtended(role="user", content=[TextContent(type="text", text="Be concise.")])
            )
        ```
    """

    def __init__(
        self,
        agent: "AgentProtocol",
        messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        use_history: bool = True,
    ):
        """
        Initialize a tool runner.

        Args:
            agent: The agent to use for generation and tool execution
            messages: Initial conversation messages
            request_params: Optional parameters for LLM requests
            tools: Optional list of available tools (uses agent's tools if not provided)
            max_iterations: Maximum number of tool-call iterations before stopping
            use_history: If True, only send tool results (agent manages history).
                         If False, accumulate full message history locally.
        """
        self._agent = agent
        self._messages = list(messages)
        self._request_params = request_params
        self._tools = tools
        self._max_iterations = max_iterations
        self._use_history = use_history

        # State
        self._iterations = 0
        self._current_message: PromptMessageExtended | None = None
        self._pending_tool_response: PromptMessageExtended | None = None
        self._tool_response_generated = False
        self._done = False

        # Customization hooks
        self._params_modifier: Callable[[RequestParams | None], RequestParams | None] | None = (
            None
        )
        self._pending_messages: List[PromptMessageExtended] = []

    def __aiter__(self) -> AsyncIterator[PromptMessageExtended]:
        """Return self as the async iterator."""
        return self

    async def __anext__(self) -> PromptMessageExtended:
        """
        Yield the next message from Claude.

        The runner will automatically handle tool calls between iterations
        unless you've already called `generate_tool_call_response()` manually.

        Raises:
            StopAsyncIteration: When the loop is complete (no more tool calls,
                                max iterations reached, or error occurred).
        """
        if self._done:
            raise StopAsyncIteration

        # Handle tool response from previous iteration
        if self._pending_tool_response is not None:
            if self._use_history:
                # Agent manages history - just send tool results
                self._messages = [self._pending_tool_response]
            else:
                # We manage full history
                self._messages.extend([self._current_message, self._pending_tool_response])
            self._pending_tool_response = None
            self._tool_response_generated = False

        # Add any custom appended messages
        if self._pending_messages:
            self._messages.extend(self._pending_messages)
            self._pending_messages.clear()

        # Apply custom params modifier
        params = self._request_params
        if self._params_modifier:
            params = self._params_modifier(params)
            self._params_modifier = None

        # Generate next response from the LLM
        # We call the underlying LLM directly to avoid the agent's own tool loop
        result = await self._agent.llm.generate(
            self._messages,
            request_params=params,
            tools=self._tools,
        )

        self._current_message = result
        self._iterations += 1

        # Check termination conditions
        if result.stop_reason == LlmStopReason.ERROR:
            self._done = True
        elif result.stop_reason != LlmStopReason.TOOL_USE:
            self._done = True
        elif self._iterations >= self._max_iterations:
            self._done = True
        else:
            # Pre-generate tool response for next iteration
            # (unless user calls generate_tool_call_response() manually)
            await self._auto_generate_tool_response()

        return result

    async def _auto_generate_tool_response(self) -> None:
        """Automatically generate tool response if not already done."""
        if not self._tool_response_generated and self._current_message:
            if self._current_message.stop_reason == LlmStopReason.TOOL_USE:
                self._pending_tool_response = await self._agent.run_tools(self._current_message)
                self._tool_response_generated = True

                # Check for errors in tool execution
                error_channel = (self._pending_tool_response.channels or {}).get(
                    FAST_AGENT_ERROR_CHANNEL
                )
                if error_channel:
                    # Copy error content to the current message and mark as error
                    tool_result_contents = [
                        content
                        for tool_result in (
                            self._pending_tool_response.tool_results or {}
                        ).values()
                        for content in tool_result.content
                    ]
                    if tool_result_contents:
                        if self._current_message.content is None:
                            self._current_message.content = []
                        self._current_message.content.extend(tool_result_contents)
                    self._current_message.stop_reason = LlmStopReason.ERROR
                    self._done = True

    async def generate_tool_call_response(self) -> PromptMessageExtended | None:
        """
        Manually generate and retrieve the tool call response.

        This gives you access to the tool result before it's sent back to Claude.
        If you don't call this, it happens automatically at the start of the next iteration.

        Returns:
            The tool response message, or None if there was no tool call.

        Example:
            ```python
            async for message in runner:
                tool_response = await runner.generate_tool_call_response()
                if tool_response:
                    # Inspect or modify tool results before they're sent
                    print(f"Tool results: {tool_response.tool_results}")
            ```
        """
        if self._current_message and self._current_message.stop_reason == LlmStopReason.TOOL_USE:
            if not self._tool_response_generated:
                self._pending_tool_response = await self._agent.run_tools(self._current_message)
                self._tool_response_generated = True
            return self._pending_tool_response
        return None

    def set_request_params(
        self, modifier: Callable[[RequestParams | None], RequestParams | None]
    ) -> None:
        """
        Set a modifier function for the next request's parameters.

        The modifier receives the current RequestParams (or None) and should
        return the modified params to use for the next LLM call.

        Args:
            modifier: Function that transforms request parameters

        Example:
            ```python
            # Increase max tokens for next request
            runner.set_request_params(
                lambda p: RequestParams(**(p.model_dump() if p else {}) | {"maxTokens": 4096})
            )
            ```
        """
        self._params_modifier = modifier

    def append_messages(self, *messages: PromptMessageExtended) -> None:
        """
        Add additional messages to be included in the next request.

        These messages are appended after the tool response (if any) and
        before calling the LLM.

        Args:
            *messages: Messages to append

        Example:
            ```python
            from mcp.types import TextContent

            runner.append_messages(
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="Please be concise.")]
                )
            )
            ```
        """
        self._pending_messages.extend(messages)

    async def until_done(self) -> PromptMessageExtended:
        """
        Consume all iterations and return the final message.

        Use this when you don't care about intermediate messages.

        Returns:
            The final message from Claude after all tool calls are complete.

        Example:
            ```python
            runner = agent.tool_runner(messages=[...])
            final = await runner.until_done()
            print(final.last_text())
            ```
        """
        final = None
        async for message in self:
            final = message
        return final

    @property
    def current_message(self) -> PromptMessageExtended | None:
        """The most recently yielded message from Claude."""
        return self._current_message

    @property
    def messages(self) -> List[PromptMessageExtended]:
        """
        A copy of the current message history.

        Note: This reflects what will be sent in the next iteration,
        not necessarily the full conversation history if use_history=True.
        """
        return self._messages.copy()

    @property
    def is_done(self) -> bool:
        """Whether the runner has finished (no more tool calls expected)."""
        return self._done

    @property
    def iterations(self) -> int:
        """Number of LLM calls made so far."""
        return self._iterations
