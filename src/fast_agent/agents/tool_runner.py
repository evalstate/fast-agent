"""
Iterable tool runner for fine-grained control over the tool call loop.

This module provides a ToolRunner class that allows users to iterate over
each message in a tool call loop, similar to Anthropic's SDK tool_runner pattern.
"""

from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Callable,
    Sequence,
    Union,
)

from mcp.types import Tool

from fast_agent.constants import DEFAULT_MAX_ITERATIONS
from fast_agent.mcp.helpers.content_helpers import normalize_to_extended_list
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class ToolRunner:
    """
    Async iterator for fine-grained control over the tool call loop.

    Each iteration yields the message returned by the LLM. After yielding,
    if Claude requested a tool use, the runner automatically calls the tool
    and prepares the result for the next iteration.

    The loop continues until:
    - The LLM returns a message without tool use (natural completion)
    - max_iterations is reached
    - The caller breaks out of the loop

    Usage:
        runner = await agent.tool_runner("What's the weather?")
        async for message in runner:
            print(message.first_text())
            # Can break early if needed

        # Or run to completion:
        final = await runner.until_done()

    Advanced usage:
        async for message in runner:
            # Inspect the pending tool response
            tool_response = runner.generate_tool_call_response()
            if tool_response:
                print("Tool was called")

            # Modify request params for next iteration
            runner.update_request_params(lambda p: RequestParams(
                **(p.model_dump() if p else {}),
                max_tokens=2048
            ))

            # Add additional context
            runner.append_messages("Please be concise.")
    """

    def __init__(
        self,
        agent: "AgentProtocol",
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        """
        Initialize the tool runner.

        Args:
            agent: The agent to use for LLM calls and tool execution
            messages: Initial messages to send to the LLM
            request_params: Optional request parameters
            tools: Optional list of tools available to the LLM
            max_iterations: Maximum number of tool call iterations
        """
        self._agent = agent
        self._messages = list(messages)  # Copy to avoid mutation
        self._request_params = request_params
        self._tools = tools
        self._max_iterations = max_iterations

        # State
        self._iteration = 0
        self._done = False
        self._last_message: PromptMessageExtended | None = None
        self._pending_tool_response: PromptMessageExtended | None = None

    def __aiter__(self) -> AsyncIterator[PromptMessageExtended]:
        """Return self as the async iterator."""
        return self

    async def __anext__(self) -> PromptMessageExtended:
        """
        Get the next message from the LLM.

        Returns:
            The LLM's response as a PromptMessageExtended

        Raises:
            StopAsyncIteration: When the loop is complete
        """
        if self._done:
            raise StopAsyncIteration

        if self._iteration >= self._max_iterations:
            self._done = True
            raise StopAsyncIteration

        # If there's a pending tool response from a previous iteration, add it
        if self._pending_tool_response:
            self._messages.append(self._pending_tool_response)
            self._pending_tool_response = None

        # Call the LLM directly (bypassing the internal tool loop in generate_impl)
        assert self._agent.llm is not None, "Agent must have an attached LLM"
        result = await self._agent.llm.generate(
            self._messages,
            self._request_params,
            self._tools,
        )

        self._last_message = result
        self._messages.append(result)
        self._iteration += 1

        # Check if we should continue (tool use requested)
        if result.stop_reason == LlmStopReason.TOOL_USE:
            # Run tools and store response for next iteration
            tool_response = await self._agent.run_tools(result)
            self._pending_tool_response = tool_response
        else:
            self._done = True

        return result

    async def until_done(self) -> PromptMessageExtended:
        """
        Run the loop to completion, returning the final message.

        This is a convenience method for when you don't need to process
        intermediate messages.

        Returns:
            The final message from the LLM

        Raises:
            RuntimeError: If no messages were generated
        """
        last: PromptMessageExtended | None = None
        async for message in self:
            last = message
        if last is None:
            raise RuntimeError("No messages generated")
        return last

    # --- Advanced API ---

    def generate_tool_call_response(self) -> PromptMessageExtended | None:
        """
        Get the tool response that will be sent in the next iteration.

        This allows inspection of tool results before they are sent back
        to the LLM.

        Returns:
            The pending tool response, or None if no tool was called
        """
        return self._pending_tool_response

    def set_request_params(self, params: RequestParams) -> None:
        """
        Set request params for the next LLM call.

        Args:
            params: The new request parameters
        """
        self._request_params = params

    def update_request_params(
        self,
        updater: Callable[[RequestParams | None], RequestParams],
    ) -> None:
        """
        Update request params using a function.

        Args:
            updater: Function that takes current params and returns new params
        """
        self._request_params = updater(self._request_params)

    def append_messages(
        self,
        *messages: Union[str, PromptMessageExtended],
    ) -> None:
        """
        Add additional messages to the conversation.

        These messages will be included in the next LLM call.

        Args:
            messages: Messages to add (strings are converted to user messages)
        """
        for msg in messages:
            if isinstance(msg, str):
                self._messages.extend(normalize_to_extended_list(msg))
            else:
                self._messages.append(msg)

    def replace_pending_tool_response(
        self,
        response: PromptMessageExtended,
    ) -> None:
        """
        Replace the pending tool response with a custom one.

        This allows modifying or completely replacing tool results
        before they are sent to the LLM.

        Args:
            response: The replacement tool response
        """
        self._pending_tool_response = response

    def skip_tool_response(self) -> None:
        """
        Skip sending the pending tool response.

        The next iteration will not include the tool result.
        """
        self._pending_tool_response = None

    @property
    def messages(self) -> list[PromptMessageExtended]:
        """
        Get a copy of the current message history.

        Returns:
            List of messages exchanged so far
        """
        return list(self._messages)

    @property
    def iteration(self) -> int:
        """
        Get the current iteration number.

        Returns:
            Number of LLM calls made so far
        """
        return self._iteration

    @property
    def is_done(self) -> bool:
        """
        Check whether the runner has completed.

        Returns:
            True if the loop is finished
        """
        return self._done

    @property
    def last_message(self) -> PromptMessageExtended | None:
        """
        Get the most recently yielded message.

        Returns:
            The last message from the LLM, or None if not started
        """
        return self._last_message

    @property
    def has_pending_tool_response(self) -> bool:
        """
        Check if there's a pending tool response.

        Returns:
            True if a tool was called and the response is pending
        """
        return self._pending_tool_response is not None


async def create_tool_runner(
    agent: "AgentProtocol",
    messages: Union[
        str,
        PromptMessageExtended,
        Sequence[Union[str, PromptMessageExtended]],
    ],
    request_params: RequestParams | None = None,
    tools: list[Tool] | None = None,
    max_iterations: int | None = None,
) -> ToolRunner:
    """
    Factory function to create a ToolRunner.

    This handles the async initialization needed to fetch tools from the agent.

    Args:
        agent: The agent to use for LLM calls and tool execution
        messages: Initial message(s) to send
        request_params: Optional request parameters
        tools: Optional tools (if None, fetched from agent)
        max_iterations: Maximum iterations (if None, uses request_params or default)

    Returns:
        A configured ToolRunner ready for iteration
    """
    # Normalize messages
    normalized = normalize_to_extended_list(messages)

    # Get tools if not provided
    if tools is None:
        tools_result = await agent.list_tools()
        tools = tools_result.tools

    # Get effective request params
    effective_params = request_params
    if agent.llm:
        effective_params = agent.llm.get_request_params(request_params)

    # Determine max iterations
    if max_iterations is None:
        max_iterations = (
            effective_params.max_iterations
            if effective_params
            else DEFAULT_MAX_ITERATIONS
        )

    return ToolRunner(
        agent=agent,
        messages=normalized,
        request_params=effective_params,
        tools=tools,
        max_iterations=max_iterations,
    )
