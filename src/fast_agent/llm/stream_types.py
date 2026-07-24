from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StreamChunk:
    """Typed streaming chunk emitted by providers."""

    text: str = ""
    is_reasoning: bool = False
    event: Literal["delta", "commit", "rollback"] = "delta"
