"""Secret-bearing endpoint contract shared by the native Copilot broker and adapters."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

Transport = Literal["sse", "websocket"]


@dataclass(frozen=True)
class CopilotEndpoint:
    model_id: str
    wire_api: Literal["messages", "responses"]
    transport: Transport
    base_url: str = field(repr=False)
    headers: Mapping[str, str] = field(repr=False)
