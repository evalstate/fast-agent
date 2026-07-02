from __future__ import annotations

from fast_agent.utils import huggingface_hub


class VerifiedHubIdentity:
    def whoami(self, token: bool | str | None = None, *, cache: bool = False) -> dict[str, str]:
        return {"name": "fast-agent"}


class UnverifiedHubIdentity:
    def whoami(self, token: bool | str | None = None, *, cache: bool = False) -> dict[str, str]:
        raise RuntimeError("not logged in")


def test_get_huggingface_hub_token_normalizes_token_string() -> None:
    assert huggingface_hub.get_huggingface_hub_token(lambda: " hf_token ") == "hf_token"


def test_get_huggingface_hub_token_returns_none_for_blank_token() -> None:
    assert huggingface_hub.get_huggingface_hub_token(lambda: "   ") is None


def test_get_huggingface_hub_token_returns_none_when_provider_fails() -> None:
    def failing_get_token() -> object:
        raise RuntimeError("token store unavailable")

    assert huggingface_hub.get_huggingface_hub_token(failing_get_token) is None


def test_is_huggingface_hub_logged_in_uses_whoami() -> None:
    assert huggingface_hub.is_huggingface_hub_logged_in(VerifiedHubIdentity())


def test_is_huggingface_hub_logged_in_returns_false_when_whoami_fails() -> None:
    assert not huggingface_hub.is_huggingface_hub_logged_in(UnverifiedHubIdentity())
