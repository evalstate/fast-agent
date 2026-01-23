"""
Skills and tools helpers for interactive prompt (initial shim exporting existing functions).
"""
from fast_agent.ui.interactive_prompt import (
    InteractivePrompt,  # keep available while migrating
    # _list_tools and _list_skills live as methods - we'll call them via InteractivePrompt
)

__all__ = ["InteractivePrompt"]
