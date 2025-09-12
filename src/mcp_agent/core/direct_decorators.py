"""
Compatibility shim for direct decorators.

Canonical location: fast_agent.core.direct_decorators
Also exposes private helpers for backward compatibility.
"""

from fast_agent.core import direct_decorators as _dd
from fast_agent.core.direct_decorators import *  # noqa: F401,F403

# Explicitly re-export private helpers referenced by legacy code/tests
_apply_templates = _dd._apply_templates
_resolve_instruction = _dd._resolve_instruction
