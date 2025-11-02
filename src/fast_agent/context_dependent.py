from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fast_agent.context import Context


class ContextDependent:
    """
    Mixin class for components that need context access.
    Provides both global fallback and instance-specific context support.
    """

    # Ensure the attribute always exists even if a subclass
    # does not call this mixin's __init__.
    _context: "Context | None" = None

    def __init__(self, context: "Context | None" = None, **kwargs: dict[str, Any]) -> None:
        self._context = context
        super().__init__()

    @property
    def context(self) -> "Context":
        """
        Get context, with graceful fallback to global context if needed.
        Raises clear error if no context is available.
        """
        # First try instance context
        if self._context is not None:
            return self._context

        try:
            # Fall back to global context if available
            from fast_agent.context import get_current_context

            return get_current_context()
        except Exception as e:
            raise RuntimeError(
                f"No context available for {self.__class__.__name__}. Either initialize Core first or pass context explicitly."
            ) from e

    @contextmanager
    def use_context(self, context: "Context"):
        """Temporarily use a different context."""
        old_context = self._context
        self._context = context
        try:
            yield
        finally:
            self._context = old_context
