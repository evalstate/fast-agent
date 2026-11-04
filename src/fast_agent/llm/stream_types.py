from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StreamChunk:
    """Typed streaming event for a provider request's retry attempts.

    A ``delta`` carries provisional text. ``commit`` accepts text since the last
    boundary; ``rollback`` discards it while preserving earlier committed text.
    These boundaries apply to provider request attempts, not the whole agent run.
    Fatal errors and cancellation may propagate without either boundary, so
    consumers must also inspect the final result or raised exception.
    """

    text: str = ""
    is_reasoning: bool = False
    event: Literal["delta", "commit", "rollback"] = "delta"
