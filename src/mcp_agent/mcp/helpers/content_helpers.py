"""Deprecated module. Use fast_agent.mcp.helpers.content_helpers instead."""

import warnings

from fast_agent.mcp.helpers.content_helpers import *  # noqa: F401,F403

warnings.warn(
    "mcp_agent.mcp.helpers.content_helpers is deprecated; use fast_agent.mcp.helpers.content_helpers",
    DeprecationWarning,
    stacklevel=2,
)
