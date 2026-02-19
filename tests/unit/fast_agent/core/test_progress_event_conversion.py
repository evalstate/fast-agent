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
            }
        },
    )

    progress_event = convert_log_event(event)

    assert progress_event is not None
    assert progress_event.correlation_id == "use-789"
    assert progress_event.tool_event == "stop"


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
                "details": "chunk 1",
            }
        },
    )

    progress_event = convert_log_event(event)
    assert progress_event is not None
    assert progress_event.details == "index - chunk 1"
