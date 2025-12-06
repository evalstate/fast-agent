"""
Iterable Tool Runner for fine-grained control over the tool loop.

This module provides a ToolRunner class that allows users to iterate over
each message in a tool-calling conversation, similar to the Anthropic SDK's
tool_runner pattern.
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    List,
    Sequence,
    Union,
)

from mcp.types import ListToolsResult, PromptMessage, Tool

from fast_agent.constants import DEFAULT_MAX_ITERATIONS, FAST_AGENT_ERROR_CHANNEL
from fast_agent.mcp.helpers.content_helpers import normalize_to_extended_list
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    pass


# Type aliases for the callback functions
GenerateFn = Callable[
    [List[PromptMessageExtended], RequestParams | None, List[Tool] | None],
    Awaitable[PromptMessageExtended],
]
RunToolsFn = Callable[[PromptMessageExtended], Awaitable[PromptMessageExtended]]
ListToolsFn = Callable[[], Awaitable[ListToolsResult]]


@dataclass
class ToolRunner:
    """
    An async iterable tool runner that yields messages from the tool loop.

    This class provides fine-grained control over the tool-calling loop,
    allowing users to:
    - Iterate over each message from Claude (including intermediate tool_use messages)
    - Inspect and modify request parameters between iterations
    - Add additional messages to the conversation
    - Break out of the loop early
    - Or simply call until_done() to get the final message

    Example:
        ```python
        runner = agent.tool_runner("What's the weather in Paris?")

        # Iterate over intermediate messages
        async for message in runner:
            print(f"Claude: {message.first_text()}")
            if message.tool_calls:
                print(f"Tools requested: {list(message.tool_calls.keys())}")

        # Or just get the final result
        runner = agent.tool_runner("Calculate 15 + 27")
        final = await runner.until_done()
        print(final.first_text())
        ```
    """

    # Core dependencies (injected by the agent)
    _generate_fn: GenerateFn
    _run_tools_fn: RunToolsFn
    _list_tools_fn: ListToolsFn

    # Request state
    messages: List[PromptMessageExtended]
    request_params: RequestParams | None = None
    tools: List[Tool] | None = None

    # Configuration
    use_history: bool = True

    # Internal state (not settable at construction)
    _iterations: int = field(default=0, init=False)
    _current_message: PromptMessageExtended | None = field(default=None, init=False)
    _pending_tool_response: PromptMessageExtended | None = field(default=None, init=False)
    _done: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)

    @property
    def max_iterations(self) -> int:
        """Maximum number of tool-calling iterations allowed."""
        if self.request_params:
            return self.request_params.max_iterations
        return DEFAULT_MAX_ITERATIONS

    @property
    def current_message(self) -> PromptMessageExtended | None:
        """The most recently yielded message from Claude."""
        return self._current_message

    @property
    def is_done(self) -> bool:
        """Whether the tool loop has completed."""
        return self._done

    @property
    def iterations(self) -> int:
        """Number of tool-calling iterations completed so far."""
        return self._iterations

    def __aiter__(self) -> AsyncIterator[PromptMessageExtended]:
        """Return self as an async iterator."""
        return self

    async def __anext__(self) -> PromptMessageExtended:
        """
        Yield the next message from Claude.

        Each iteration:
        1. If there's a pending tool response from the previous iteration,
           add it to the messages
        2. Call the LLM to generate the next message
        3. If the message contains tool_use, pre-compute the tool response
           for the next iteration
        4. Yield the message

        Raises:
            StopAsyncIteration: When the loop is complete (no more tool_use
                or max iterations reached)
        """
        if self._done:
            raise StopAsyncIteration

        # Handle pending tool response from previous iteration
        if self._pending_tool_response is not None:
            if self.use_history:
                # Sliding window: only send the tool results
                self.messages = [self._pending_tool_response]
            else:
                # Full context: append both the assistant message and tool results
                if self._current_message:
                    self.messages.append(self._current_message)
                self.messages.append(self._pending_tool_response)
            self._pending_tool_response = None

        # Check iteration limit
        if self._iterations > self.max_iterations:
            self._done = True
            raise StopAsyncIteration

        # Get tools if not provided
        if self.tools is None:
            tools_result = await self._list_tools_fn()
            self.tools = tools_result.tools

        # Generate next message from LLM
        self._current_message = await self._generate_fn(
            self.messages,
            self.request_params,
            self.tools,
        )
        self._started = True

        # Check if Claude requested tool use
        if self._current_message.stop_reason == LlmStopReason.TOOL_USE:
            # Pre-compute tool response for next iteration
            self._pending_tool_response = await self._run_tools_fn(self._current_message)
            self._iterations += 1

            # Check for tool loop errors
            error_messages = (self._pending_tool_response.channels or {}).get(
                FAST_AGENT_ERROR_CHANNEL
            )
            if error_messages:
                # Copy error content to the current message
                tool_result_contents = [
                    content
                    for tool_result in (self._pending_tool_response.tool_results or {}).values()
                    for content in tool_result.content
                ]
                if tool_result_contents:
                    if self._current_message.content is None:
                        self._current_message.content = []
                    self._current_message.content.extend(tool_result_contents)
                self._current_message.stop_reason = LlmStopReason.ERROR
                self._done = True
        else:
            # No tool use requested, we're done after this message
            self._done = True

        return self._current_message

    async def until_done(self) -> PromptMessageExtended:
        """
        Consume all iterations and return the final message from Claude.

        This is equivalent to:
            async for message in runner:
                pass
            return runner.current_message

        Returns:
            The final PromptMessageExtended from Claude after all tool calls
            have been processed.

        Raises:
            RuntimeError: If called on an already-exhausted runner with no messages.
        """
        async for _ in self:
            pass

        if self._current_message is None:
            raise RuntimeError("Tool runner completed without producing any messages")

        return self._current_message

    # === Advanced API ===

    def get_pending_tool_response(self) -> PromptMessageExtended | None:
        """
        Get the tool response that will be sent on the next iteration.

        This allows inspection of tool results before they're sent back to Claude.
        Returns None if the current message doesn't have tool calls or if tools
        haven't been executed yet.

        Returns:
            The pending tool response message, or None.
        """
        return self._pending_tool_response

    def set_request_params(
        self, updater: Callable[[RequestParams | None], RequestParams]
    ) -> None:
        """
        Modify the request params for the next API request.

        Example:
            runner.set_request_params(
                lambda p: RequestParams(**(p.model_dump() if p else {}), maxTokens=4096)
            )

        Args:
            updater: A function that takes the current params and returns new params.
        """
        self.request_params = updater(self.request_params)

    def append_messages(
        self,
        *messages: Union[str, PromptMessage, PromptMessageExtended],
    ) -> None:
        """
        Add additional messages to the conversation.

        These will be included in the next API request. Useful for injecting
        system guidance or user clarifications mid-conversation.

        Args:
            *messages: Messages to append (strings, PromptMessage, or PromptMessageExtended)
        """
        for msg in messages:
            normalized = normalize_to_extended_list(msg)
            self.messages.extend(normalized)

    def set_messages(self, messages: List[PromptMessageExtended]) -> None:
        """
        Replace the current messages entirely.

        Use with caution - this replaces the entire conversation context.

        Args:
            messages: The new list of messages.
        """
        self.messages = messages

    def set_tools(self, tools: List[Tool]) -> None:
        """
        Override the tools available to Claude for subsequent iterations.

        Args:
            tools: The new list of tools.
        """
        self.tools = tools


def create_tool_runner(
    generate_fn: GenerateFn,
    run_tools_fn: RunToolsFn,
    list_tools_fn: ListToolsFn,
    messages: Union[
        str,
        PromptMessage,
        PromptMessageExtended,
        Sequence[Union[str, PromptMessage, PromptMessageExtended]],
    ],
    request_params: RequestParams | None = None,
    tools: List[Tool] | None = None,
    use_history: bool = True,
) -> ToolRunner:
    """
    Factory function to create a ToolRunner with normalized messages.

    This is the recommended way to create a ToolRunner, as it handles
    message normalization automatically.

    Args:
        generate_fn: Function to call the LLM (without tool loop)
        run_tools_fn: Function to execute tool calls
        list_tools_fn: Function to list available tools
        messages: Input messages in any supported format
        request_params: Optional request parameters
        tools: Optional list of tools (if None, will be fetched via list_tools_fn)
        use_history: Whether to use sliding window history mode

    Returns:
        A configured ToolRunner instance.
    """
    normalized = normalize_to_extended_list(messages)

    return ToolRunner(
        _generate_fn=generate_fn,
        _run_tools_fn=run_tools_fn,
        _list_tools_fn=list_tools_fn,
        messages=normalized,
        request_params=request_params,
        tools=tools,
        use_history=use_history,
    )
