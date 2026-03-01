"""Compatibility shim for model picker prototypes.

This script module now re-exports the shared implementation from
``fast_agent.ui.model_picker_common`` so prototype scripts and runtime picker
stay in sync.
"""

from __future__ import annotations

from fast_agent.ui.model_picker_common import (  # noqa: F401
    DEFAULT_VALUE,
    KEEP_VALUE,
    PICKER_PROVIDER_ORDER,
    REFER_TO_DOCS_PROVIDERS,
    ModelCapabilities,
    ModelOption,
    ModelPickerSnapshot,
    ModelSource,
    ProviderOption,
    active_provider_names,
    apply_option_overrides,
    build_provider_label,
    build_snapshot,
    find_provider,
    model_capabilities,
    model_options_for_provider,
    web_search_display,
)

__all__ = [
    "DEFAULT_VALUE",
    "KEEP_VALUE",
    "PICKER_PROVIDER_ORDER",
    "REFER_TO_DOCS_PROVIDERS",
    "ModelCapabilities",
    "ModelOption",
    "ModelPickerSnapshot",
    "ModelSource",
    "ProviderOption",
    "active_provider_names",
    "apply_option_overrides",
    "build_provider_label",
    "build_snapshot",
    "find_provider",
    "model_capabilities",
    "model_options_for_provider",
    "web_search_display",
]
