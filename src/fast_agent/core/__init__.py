"""
fast_agent.core package

Provides core subpackages (executor, logging) and lazily exposes the `Core`
class to avoid circular imports during initialization.
"""

__all__ = ["Core"]


def __getattr__(name: str):
    if name == "Core":
        # Lazy import to avoid importing heavy dependencies during package init
        from .core_app import Core

        return Core
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
