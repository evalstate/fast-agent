from fast_agent.mcp.message_roles import (
    DEFAULT_MESSAGE_ROLE,
    MESSAGE_ROLE_NAMES,
    coerce_message_role,
    is_message_role,
)


class _UserLike:
    def __eq__(self, other: object) -> bool:
        return other == "user"


def test_is_message_role_accepts_supported_role_strings() -> None:
    assert is_message_role("user")
    assert is_message_role("assistant")


def test_is_message_role_rejects_non_string_values() -> None:
    assert not is_message_role(_UserLike())
    assert not is_message_role(None)


def test_coerce_message_role_returns_supported_roles() -> None:
    assert coerce_message_role("assistant") == "assistant"
    assert coerce_message_role("user") == "user"


def test_coerce_message_role_uses_default_for_invalid_values() -> None:
    assert coerce_message_role("tool") == DEFAULT_MESSAGE_ROLE
    assert coerce_message_role(None, default="assistant") == "assistant"


def test_message_role_names_reflect_supported_roles() -> None:
    assert MESSAGE_ROLE_NAMES == "user, assistant"
