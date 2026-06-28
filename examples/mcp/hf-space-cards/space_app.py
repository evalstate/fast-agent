from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fast_agent import AppOpenRequest, HarnessAppContext

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager


class SpaceHarnessApp:
    """Small entrypoint-loaded app wrapper for Space-wide metadata."""

    def __init__(self, context: HarnessAppContext) -> None:
        self._default_app = context.default_app

    def open(self, request: AppOpenRequest | None = None) -> AbstractAsyncContextManager[Any]:
        resolved = request or AppOpenRequest()
        return self._default_app.open(
            AppOpenRequest(
                session_id=resolved.session_id,
                agent=resolved.agent,
                metadata={
                    "example": "mcp/hf-space-cards",
                    **dict(resolved.metadata),
                },
            )
        )


def create_app(context: HarnessAppContext) -> SpaceHarnessApp:
    return SpaceHarnessApp(context)
