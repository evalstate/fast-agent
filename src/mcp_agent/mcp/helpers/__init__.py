"""
Deprecated helpers package. Use fast_agent.mcp.helpers instead.
"""

import warnings

from fast_agent.mcp.helpers import *  # noqa: F401,F403

warnings.warn(
    "mcp_agent.mcp.helpers is deprecated; use fast_agent.mcp.helpers",
    DeprecationWarning,
    stacklevel=2,
)
