from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class ChainResponseAggregator:
    """Aggregate responses for a multi-agent chain."""

    def __init__(self, chain_name: str, total_agents: int) -> None:
        self.chain_name = chain_name
        self.total_agents = total_agents
        self.agent_responses: Dict[str, Any] = {}
        self.completed_agents = 0
        self._response_sent = False

    async def add_agent_response(self, agent_name: str, response: Any) -> None:
        """Record a response from an agent in the chain."""
        self.agent_responses[agent_name] = response
        self.completed_agents += 1

    async def should_send_response(self) -> bool:
        """Return ``True`` if the aggregated response should be sent."""
        return not self._response_sent and self.completed_agents >= self.total_agents

    async def get_aggregated_response(self) -> Dict[str, Any]:
        """Return the aggregated response for the chain."""
        self._response_sent = True
        return {"chain": self.chain_name, "responses": self.agent_responses}


class SSEEventType(Enum):
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETE = "agent_complete"
    CHAIN_COMPLETE = "chain_complete"
    ERROR = "error"


async def send_sse_event(event_type: SSEEventType, data: Dict[str, Any], stream: Any) -> None:
    """Send an SSE event to the provided stream if possible."""
    if stream is not None and hasattr(stream, "send"):
        await stream.send({"event": event_type.value, "data": data})
