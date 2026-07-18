"""Tests for converting logger events into progress display events."""

from fast_agent.core.logging.events import Event
from fast_agent.core.logging.listeners import convert_log_event
from fast_agent.event_progress import ProgressAction


def test_convert_log_event_extracts_tool_correlation_id() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.mcp.mcp_aggregator",
        message="Tool progress update",
        data={
            "data": {
                "progress_action": ProgressAction.TOOL_PROGRESS,
                "agent_name": "assistant",
                "tool_name": "search",
                "server_name": "web",
                "tool_call_id": "call-123",
                "tool_use_id": "use-123",
                "progress": 1.0,
                "total": 2.0,
                "details": "step 1",
            }
        },
    )

    progress_event = convert_log_event(event)

    assert progress_event is not None
    assert progress_event.correlation_id == "use-123"
    assert progress_event.tool_name == "search"
    assert progress_event.details == "search - step 1"


def test_convert_poll_start_uses_process_details_without_tool_prefix() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.custom.runtime",
        message="Shell tool lifecycle",
        data={
            "data": {
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": "assistant",
                "tool_name": "poll_process",
                "server_name": "local",
                "tool_use_id": "call-poll",
                "tool_event": "start",
                "details": "process-4",
                "process_elapsed_seconds": 65,
                "process_command": "uv run worker.py",
                "process_id": "process-4",
                "process_wait_seconds": 30,
                "process_has_observed_output": True,
                "process_seconds_since_last_output": 4,
            }
        },
    )

    progress_event = convert_log_event(event)

    assert progress_event is not None
    assert progress_event.details == "process-4"
    assert progress_event.process_elapsed_seconds == 65
    assert progress_event.process_command == "uv run worker.py"
    assert progress_event.process_id == "process-4"
    assert progress_event.process_wait_seconds == 30
    assert progress_event.process_has_observed_output is True
    assert progress_event.process_seconds_since_last_output == 4


def test_convert_log_event_uses_tool_use_id_when_call_id_missing() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.mcp.mcp_aggregator",
        message="Requesting tool call",
        data={
            "data": {
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": "assistant",
                "tool_name": "search",
                "server_name": "web",
                "tool_use_id": "use-789",
                "tool_event": "stop",
                "tool_terminal": True,
            }
        },
    )

    progress_event = convert_log_event(event)

    assert progress_event is not None
    assert progress_event.correlation_id == "use-789"
    assert progress_event.tool_event == "stop"
    assert progress_event.tool_terminal is True


def test_convert_log_event_skips_provider_web_tool_progress_events() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.llm.provider.anthropic.llm_anthropic",
        message="Anthropic server tool started",
        data={
            "data": {
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": "assistant",
                "model": "claude-sonnet-4-6",
                "tool_name": "web_search",
                "tool_use_id": "srv_123",
                "tool_event": "start",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is None


def test_convert_log_event_keeps_mcp_web_tool_progress_events() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.mcp.mcp_aggregator",
        message="Requesting tool call",
        data={
            "data": {
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": "assistant",
                "tool_name": "web_search",
                "server_name": "my_tools",
                "tool_use_id": "call_456",
                "tool_event": "start",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.tool_name == "web_search"
    assert progress_event.server_name == "my_tools"


def test_convert_log_event_mcp_resource_read_uses_resource_action_and_details() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.mcp.mcp_aggregator",
        message="Requesting resource",
        data={
            "data": {
                "progress_action": ProgressAction.READING_RESOURCE,
                "agent_name": "assistant",
                "server_name": "docs",
                "details": "file://README.md",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.action == ProgressAction.READING_RESOURCE
    assert progress_event.details == "docs - file://README.md"


def test_convert_log_event_mcp_resource_read_completion_preserves_details() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.mcp.mcp_aggregator",
        message="Resource read complete",
        data={
            "data": {
                "progress_action": ProgressAction.RESOURCE_READ,
                "agent_name": "assistant",
                "server_name": "docs",
                "details": "file://README.md",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.action == ProgressAction.RESOURCE_READ
    assert progress_event.details == "docs - file://README.md"


def test_convert_log_event_skips_loaded_progress_events() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.core.direct_factory",
        message="Loaded assistant",
        data={
            "data": {
                "progress_action": ProgressAction.LOADED,
                "agent_name": "assistant",
                "target": "assistant",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is None


def test_convert_log_event_llm_tool_context_uses_server_and_tool_when_available() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.llm.provider.openai.llm_openai",
        message="Model started streaming tool call",
        data={
            "data": {
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": "assistant",
                "model": "gpt-5",
                "tool_name": "search",
                "server_name": "research",
                "tool_use_id": "use-42",
                "tool_event": "start",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.details == "gpt-5 • search (start)"


def test_convert_log_event_ignores_non_string_tool_context_fields() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.llm.provider.openai.llm_openai",
        message="Model started streaming tool call",
        data={
            "data": {
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": "assistant",
                "model": "gpt-5",
                "tool_name": 123,
                "server_name": " research ",
                "tool_use_id": "use-42",
                "tool_event": object(),
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.details == "gpt-5 • research"


def test_convert_log_event_llm_tool_context_omits_blank_model_prefix() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.llm.provider.openai.llm_openai",
        message="Model started streaming tool call",
        data={
            "data": {
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": "assistant",
                "model": " ",
                "tool_name": "search",
                "tool_use_id": "use-42",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.details == "search"


def test_convert_log_event_llm_chat_turn_omits_blank_model_prefix() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.llm.provider.openai.llm_openai",
        message="Model turn",
        data={
            "data": {
                "progress_action": ProgressAction.READY,
                "agent_name": "assistant",
                "model": "",
                "chat_turn": 3,
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.details == "turn 3"


def test_convert_log_event_generic_tool_progress_includes_context_and_details() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.custom.runtime",
        message="Tool progress update",
        data={
            "data": {
                "progress_action": ProgressAction.TOOL_PROGRESS,
                "agent_name": "assistant",
                "tool_name": "index",
                "server_name": "fs",
                "tool_state": "completed",
                "tool_terminal": True,
                "details": "chunk 1",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.details == "index - chunk 1"
    assert progress_event.tool_state == "completed"
    assert progress_event.tool_terminal is True


def test_convert_log_event_fatal_error_uses_server_name_as_target_when_agent_missing() -> None:
    event = Event(
        type="error",
        namespace="fast_agent.mcp.mcp_connection_manager",
        message="Lifecycle task encountered an error",
        data={
            "data": {
                "progress_action": ProgressAction.FATAL_ERROR,
                "server_name": "127-0-0-1",
                "error_message": "Connection refused",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.target == "127-0-0-1"
    assert progress_event.details == "Connection refused"


def test_convert_log_event_ignores_invalid_progress_action() -> None:
    event = Event(
        type="info",
        namespace="fast_agent.custom.runtime",
        message="Invalid progress action",
        data={
            "data": {
                "progress_action": "not-a-progress-action",
                "agent_name": "assistant",
            }
        },
    )

    assert convert_log_event(event) is None
