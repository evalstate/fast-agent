"""LLM module for Fast Agent.

Public API:
- RequestParams: main configuration object for LLM interactions.
- lookup_inference_providers: async lookup of HuggingFace inference providers.
- lookup_inference_providers_sync: sync wrapper for lookup_inference_providers.
- InferenceProviderLookupResult: result type for inference provider lookups.
- format_inference_lookup_message: format lookup results for display.
"""

from typing import TYPE_CHECKING

from .request_params import RequestParams

_HF_INFERENCE_EXPORTS = frozenset(
    {
        "InferenceProvider",
        "InferenceProviderLookupResult",
        "InferenceProviderStatus",
        "format_inference_lookup_message",
        "lookup_inference_providers",
        "lookup_inference_providers_sync",
    }
)


def __getattr__(name: str):
    if name not in _HF_INFERENCE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import hf_inference_lookup

    value = vars(hf_inference_lookup)[name]
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from .hf_inference_lookup import (
        InferenceProvider as InferenceProvider,
    )
    from .hf_inference_lookup import (
        InferenceProviderLookupResult as InferenceProviderLookupResult,
    )
    from .hf_inference_lookup import (
        InferenceProviderStatus as InferenceProviderStatus,
    )
    from .hf_inference_lookup import (
        format_inference_lookup_message as format_inference_lookup_message,
    )
    from .hf_inference_lookup import (
        lookup_inference_providers as lookup_inference_providers,
    )
    from .hf_inference_lookup import (
        lookup_inference_providers_sync as lookup_inference_providers_sync,
    )

__all__ = [
    "InferenceProvider",
    "InferenceProviderLookupResult",
    "InferenceProviderStatus",
    "RequestParams",
    "format_inference_lookup_message",
    "lookup_inference_providers",
    "lookup_inference_providers_sync",
]
