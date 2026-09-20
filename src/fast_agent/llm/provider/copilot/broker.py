"""Native Copilot endpoint broker: local credentials to secret-bearing endpoints."""

from __future__ import annotations

import asyncio
from importlib.metadata import version
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from fast_agent.config import CopilotSettings
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.copilot.endpoint import CopilotEndpoint, Transport
from fast_agent.llm.provider.copilot.models import get_copilot_model
from fast_agent.llm.provider.copilot.oauth import (
    CopilotAuthenticationError,
    get_copilot_access_token,
)

if TYPE_CHECKING:
    from fast_agent.context import Context

_USER_AGENT: Final = f"fast-agent/{version('fast-agent-mcp')}"


class CopilotBroker:
    """Resolve endpoints from local credentials without checking remote model entitlement.

    The broker holds no connections or cached credentials; every operation reloads
    validated local credentials, so instances are cheap and need no lifecycle.
    """

    def __init__(self, settings: CopilotSettings) -> None:
        self._settings = settings

    async def _access_token(self, timeout: float) -> str:
        # Load validated credentials afresh on every request. Only this read runs
        # in a worker, so cancellation cannot leave authentication state mutations.
        try:
            async with asyncio.timeout(timeout):
                token = await asyncio.to_thread(get_copilot_access_token)
        except TimeoutError:
            raise ProviderKeyError("Copilot operation timed out.") from None
        if token is None:
            raise CopilotAuthenticationError(
                "Copilot credentials are missing. Sign in with: fast-agent auth provider login copilot"
            )
        return token

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "authorization": f"Bearer {token}",
            "user-agent": _USER_AGENT,
            "copilot-integration-id": self._settings.integration_id,
            "x-github-api-version": "2026-06-01",
            "openai-intent": "conversation-edits",
        }

    async def has_credentials(self, timeout: float | None = None) -> bool:
        """Check local presence, expiry, and format without remote entitlement checks or login."""
        limit = self._settings.runtime_timeout_seconds
        if timeout is not None:
            limit = min(timeout, limit)
        try:
            await self._access_token(limit)
        except CopilotAuthenticationError:
            # Missing (raised above) and expired (raised by the loader) credentials
            # both mean "not signed in". Invalid overrides/stores propagate.
            return False
        return True

    async def resolve(
        self, model_id: str, *, owner_id: str, transport: Transport = "sse"
    ) -> CopilotEndpoint:
        spec = get_copilot_model(model_id)
        if transport not in spec.transports:
            raise ProviderKeyError("Requested Copilot transport is not supported for this model.")
        headers = self._headers(await self._access_token(self._settings.runtime_timeout_seconds))
        headers["x-interaction-id"] = owner_id
        if spec.wire_api == "messages":
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        return CopilotEndpoint(
            model_id=model_id,
            wire_api=spec.wire_api,
            transport=transport,
            base_url=self._settings.base_url,
            headers=MappingProxyType(headers),
        )


def get_copilot_broker(context: Context) -> CopilotBroker:
    return CopilotBroker(context.config.copilot if context.config else CopilotSettings())
