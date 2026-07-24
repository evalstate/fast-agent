from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from fast_agent.acp.server.session_runtime import ACPServerSessionRuntime

if TYPE_CHECKING:
    from fast_agent.acp.tool_progress import ACPToolProgressManager
    from fast_agent.interfaces import AgentProtocol


class _LiveToolStreamLLM:
    def __init__(self) -> None:
        self.listener: Callable[[str, dict[str, Any] | None], None] | None = None

    def add_tool_stream_listener(
        self,
        listener: Callable[[str, dict[str, Any] | None], None],
    ) -> None:
        self.listener = listener


class _AgentWithLLM:
    def __init__(self, llm: _LiveToolStreamLLM) -> None:
        self.llm = llm


class _ToolHandler:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any] | None]] = []

    def handle_tool_stream_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None,
    ) -> None:
        self.events.append((event_type, payload))


def test_acp_tool_stream_events_are_forwarded_live() -> None:
    llm = _LiveToolStreamLLM()
    handler = _ToolHandler()

    ACPServerSessionRuntime._register_tool_stream_listener(
        cast("AgentProtocol", _AgentWithLLM(llm)),
        tool_handler=cast("ACPToolProgressManager", handler),
    )

    assert llm.listener is not None
    llm.listener("delta", {"tool_use_id": "call-1", "chunk": '{"path":'})

    assert handler.events == [
        ("delta", {"tool_use_id": "call-1", "chunk": '{"path":'}),
    ]
