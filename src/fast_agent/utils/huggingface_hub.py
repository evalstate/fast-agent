from importlib import import_module


def get_huggingface_hub_token() -> str | None:
    """Return the active Hugging Face Hub token when huggingface_hub is installed."""
    try:
        module = import_module("huggingface_hub")
    except Exception:
        return None

    get_token = getattr(module, "get_token", None)
    if not callable(get_token):
        return None

    try:
        token = get_token()
    except Exception:
        return None

    if not isinstance(token, str) or not token:
        return None
    return token
