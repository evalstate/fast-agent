"""hf-inference-acp - Hugging Face inference agent with ACP support."""

from importlib.metadata import version

__all__ = ["__version__"]


def _resolve_version() -> str:
    try:
        return version("hf-inference-acp")
    except Exception:
        return "0.0.0"


__version__ = _resolve_version()
