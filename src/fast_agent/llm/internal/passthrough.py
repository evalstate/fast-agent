import json  # Import at the module level
from typing import Any, Dict, List, Optional

from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams, PromptMessage

from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.llm.fastagent_llm import (
    FastAgentLLM,
    RequestParams,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import create_turn_usage_from_messages
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

CALL_TOOL_INDICATOR = "***CALL_TOOL"
FIXED_RESPONSE_INDICATOR = "***FIXED_RESPONSE"


class PassthroughLLM(FastAgentLLM):
    """
    A specialized LLM implementation that simply passes through input messages without modification.

    This is useful for cases where you need an object with the AugmentedLLM interface
    but want to preserve the original message without any processing, such as in a
    parallel workflow where no fan-in aggregation is needed.
    """

    def __init__(
        self, provider=Provider.FAST_AGENT, name: str = "Passthrough", **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(name=name, provider=provider, **kwargs)
        self.logger = get_logger(__name__)
        self._messages = [PromptMessage]
        self._fixed_response: str | None = None
        self._correlation_id: int = 0

    async def initialize(self) -> None:
        pass

    def _parse_tool_command(self, command: str) -> tuple[str, Optional[dict]]:
        """
        Parse a tool command string into tool name and arguments.

        Args:
            command: The command string in format "***CALL_TOOL <tool_name> [arguments_json]"

        Returns:
            Tuple of (tool_name, arguments_dict)

        Raises:
            ValueError: If command format is invalid
        """
        parts = command.split(" ", 2)
        if len(parts) < 2:
            raise ValueError("Invalid format. Expected '***CALL_TOOL <tool_name> [arguments_json]'")

        tool_name = parts[1].strip()
        arguments = None

        if len(parts) > 2:
            try:
                arguments = json.loads(parts[2])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON arguments: {parts[2]}")

        self.logger.info(f"Calling tool {tool_name} with arguments {arguments}")
        return tool_name, arguments

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageExtended"],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        # Add messages to history with proper is_prompt flag
        self.history.extend(multipart_messages, is_prompt=is_template)

        last_message = multipart_messages[-1]
        tool_calls: Dict[str, CallToolRequest] = {}
        stop_reason: LlmStopReason = LlmStopReason.END_TURN
        if self.is_tool_call(last_message):
            tool_name, arguments = self._parse_tool_command(last_message.first_text())
            tool_calls["correlationId" + str(self._correlation_id)] = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=tool_name, arguments=arguments),
            )
            self._correlation_id += 1
            stop_reason = LlmStopReason.TOOL_USE

        if last_message.first_text().startswith(FIXED_RESPONSE_INDICATOR):
            self._fixed_response = (
                last_message.first_text().split(FIXED_RESPONSE_INDICATOR, 1)[1].strip()
            )

        if len(last_message.tool_results or {}) > 0:
            assert last_message.tool_results
            concatenated_content = " ".join(
                [
                    (get_text(tool_result.content[0]) or "<empty>")
                    for tool_result in last_message.tool_results.values()
                ]
            )
            result = Prompt.assistant(concatenated_content, stop_reason=stop_reason)

        elif self._fixed_response:
            result = Prompt.assistant(
                self._fixed_response, tool_calls=tool_calls, stop_reason=stop_reason
            )
        else:
            concatenated_content = "\n".join(
                [message.all_text() for message in multipart_messages if "user" == message.role]
            )
            result = Prompt.assistant(
                concatenated_content,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
            )

        turn_usage = create_turn_usage_from_messages(
            input_content=multipart_messages[-1].all_text(),
            output_content=result.all_text(),
            model="passthrough",
            model_type="passthrough",
            tool_calls=len(tool_calls),
            delay_seconds=0.0,
        )
        self.usage_accumulator.add_turn(turn_usage)

        return result

    def is_tool_call(self, message: PromptMessageExtended) -> bool:
        return message.first_text().startswith(CALL_TOOL_INDICATOR)
