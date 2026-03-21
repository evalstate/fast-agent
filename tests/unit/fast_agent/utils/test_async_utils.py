"""Tests for asyncio runtime helpers."""

import warnings

from fast_agent.utils.async_utils import (
    _UVLOOP_PROMPT_TOOLKIT_DEPRECATION_MESSAGE,
    _suppress_known_uvloop_prompt_toolkit_deprecation,
)


def test_suppress_known_uvloop_prompt_toolkit_deprecation_installs_targeted_filter() -> None:
    with warnings.catch_warnings():
        warnings.resetwarnings()
        _suppress_known_uvloop_prompt_toolkit_deprecation(version_info=(3, 14))

        assert any(
            action == "ignore"
            and category is DeprecationWarning
            and getattr(message, "pattern", None) == _UVLOOP_PROMPT_TOOLKIT_DEPRECATION_MESSAGE
            for action, message, category, module, _lineno in warnings.filters
        )
