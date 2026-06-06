from collections.abc import Callable
from contextlib import suppress
from importlib import import_module

from fast_agent.utils.text import strip_to_none


def _huggingface_hub_get_token() -> Callable[[], object] | None:
    with suppress(Exception):
        module = import_module("huggingface_hub")
        get_token = getattr(module, "get_token", None)
        return get_token if callable(get_token) else None
    return None


def get_huggingface_hub_token() -> str | None:
    """Return the active Hugging Face Hub token when huggingface_hub is installed."""
    get_token = _huggingface_hub_get_token()
    if get_token is None:
        return None

    with suppress(Exception):
        token = get_token()
        return strip_to_none(token) if isinstance(token, str) else None
    return None
