"""Assistant message phase metadata for multi-phase model responses."""

from enum import Enum
from typing import Union


class AssistantMessagePhase(str, Enum):
    """Phase labels emitted by providers for assistant message items."""

    COMMENTARY = "commentary"
    FINAL_ANSWER = "final_answer"

    def __eq__(self, other: object) -> bool:
        """Allow comparison with both enum members and raw strings."""
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    @classmethod
    def from_string(
        cls, value: Union[str, "AssistantMessagePhase"]
    ) -> "AssistantMessagePhase":
        """Convert a raw string to an assistant message phase."""
        if isinstance(value, cls):
            return value

        for member in cls:
            if member.value == value:
                return member

        raise ValueError(
            f"Invalid assistant message phase: {value}. "
            f"Valid values are: {[member.value for member in cls]}"
        )
