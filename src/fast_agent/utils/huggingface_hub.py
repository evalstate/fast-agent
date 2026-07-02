from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

from huggingface_hub import HfApi, get_token

from fast_agent.utils.text import strip_to_none


class HuggingFaceHubIdentity(Protocol):
    def whoami(self, token: bool | str | None = None, *, cache: bool = False) -> object: ...


def get_huggingface_hub_token(token_provider: Callable[[], object] = get_token) -> str | None:
    """Return the active Hugging Face Hub token."""
    with suppress(Exception):
        token = token_provider()
        return strip_to_none(token) if isinstance(token, str) else None
    return None


def is_huggingface_hub_logged_in(api: HuggingFaceHubIdentity | None = None) -> bool:
    """Return True when the Hugging Face Hub library can verify an active login."""
    with suppress(Exception):
        (api or HfApi()).whoami(cache=True)
        return True
    return False
