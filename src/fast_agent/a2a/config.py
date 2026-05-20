"""Configuration for remote A2A agents."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class A2AAgentConfig:
    url: str
    transport: str | None = None
    streaming: bool = True
    polling: bool = False
    accepted_output_modes: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    relative_card_path: str | None = None

