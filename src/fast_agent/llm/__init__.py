"""LLM module for Fast Agent.

Public API:
- RequestParams: main configuration object for LLM interactions.
- lookup_inference_providers: async lookup of HuggingFace inference providers.
- lookup_inference_providers_sync: sync wrapper for lookup_inference_providers.
- InferenceProviderLookupResult: result type for inference provider lookups.
- format_inference_lookup_message: format lookup results for display.
"""

from .hf_inference_lookup import (
    InferenceProvider,
    InferenceProviderLookupResult,
    InferenceProviderStatus,
    format_inference_lookup_message,
    lookup_inference_providers,
    lookup_inference_providers_sync,
)
from .request_params import RequestParams

__all__ = [
    "RequestParams",
    "lookup_inference_providers",
    "lookup_inference_providers_sync",
    "InferenceProvider",
    "InferenceProviderLookupResult",
    "InferenceProviderStatus",
    "format_inference_lookup_message",
]
