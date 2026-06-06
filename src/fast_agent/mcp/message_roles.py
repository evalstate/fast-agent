from __future__ import annotations

from typing import Final, Literal, TypeGuard

type MessageRole = Literal["user", "assistant"]

MESSAGE_ROLES: Final[tuple[MessageRole, ...]] = ("user", "assistant")
MESSAGE_ROLE_NAMES: Final[str] = ", ".join(MESSAGE_ROLES)
DEFAULT_MESSAGE_ROLE: Final[MessageRole] = "user"


def is_message_role(value: object) -> TypeGuard[MessageRole]:
    return isinstance(value, str) and value in MESSAGE_ROLES


def coerce_message_role(value: object, default: MessageRole = DEFAULT_MESSAGE_ROLE) -> MessageRole:
    if is_message_role(value):
        return value
    return default
