from dataclasses import dataclass


@dataclass(frozen=True)
class StreamChunk:
    """Typed streaming chunk emitted by providers."""

    text: str
    is_reasoning: bool = False
