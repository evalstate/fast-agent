"""Native Copilot endpoint broker: local credentials to secret-bearing endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from importlib.metadata import version
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from fast_agent.config import CopilotSettings
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.copilot.models import get_copilot_model
from fast_agent.llm.provider.copilot.oauth import (
    CopilotAuthenticationError,
    get_copilot_access_token,
)
from fast_agent.utils.async_utils import run_in_daemon_thread

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.context import Context

_USER_AGENT: Final = f"fast-agent/{version('fast-agent-mcp')}"

Transport = Literal["sse", "websocket"]


@dataclass(frozen=True)
class CopilotEndpoint:
    """Secret-bearing endpoint binding shared by the broker and the protocol adapters."""

    model_id: str
    wire_api: Literal["messages", "responses"]
    transport: Transport
    base_url: str = field(repr=False)
    headers: Mapping[str, str] = field(repr=False)


def copilot_settings(context: Context) -> CopilotSettings:
    return context.config.copilot if context.config else CopilotSettings()


class CopilotBroker:
    """Resolve endpoints from local credentials without checking remote model entitlement.

    The broker holds no connections or cached credentials; every operation reloads
    validated local credentials, so instances are cheap and need no lifecycle.
    """

    def __init__(self, settings: CopilotSettings) -> None:
        self._settings = settings

    async def _access_token(self, timeout: float) -> str:
        # Resolve afresh in a daemon worker. Timeout/cancellation stops waiting,
        # not the worker: an in-flight refresh may still persist rotated tokens
        # under the shared lock. HTTP and the lock wait are bounded; the keyring
        # is not, so the worker must never block interpreter exit.
        try:
            async with asyncio.timeout(timeout):
                token = await run_in_daemon_thread(
                    get_copilot_access_token, name="fast-agent-copilot-token"
                )
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
        """Resolve credentials (refreshing if needed), without entitlement checks or login."""
        limit = self._settings.runtime_timeout_seconds
        if timeout is not None:
            limit = min(timeout, limit)
        try:
            await self._access_token(limit)
        except CopilotAuthenticationError:
            # Missing, expired, or failed refresh means "not signed in".
            # Invalid overrides/stores propagate.
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
    return CopilotBroker(copilot_settings(context))
