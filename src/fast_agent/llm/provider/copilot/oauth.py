"""Native GitHub device authorization, independent of GitHub/Copilot CLI state."""

from __future__ import annotations

import asyncio
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from fast_agent.auth.credentials import (
    OAuthCredential,
    StoredCredential,
    delete_oauth_credential,
    load_oauth_credential,
    save_oauth_credential,
)
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.ui import console

if TYPE_CHECKING:
    from collections.abc import Iterator

COPILOT_CLIENT_ID: Final = "Ov23li9BBH5sVoKopuI6"
COPILOT_PROVIDER_ID: Final = "copilot"
_DEVICE_URL: Final = "https://github.com/login/device/code"
_TOKEN_URL: Final = "https://github.com/login/oauth/access_token"
_VERIFICATION_URI: Final = "https://github.com/login/device"
_LOGIN_HINT: Final = "Run `fast-agent auth provider login copilot`."

_PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
_NonemptyString = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]


class CopilotAuthenticationError(ProviderKeyError):
    """Missing, expired, rejected, or otherwise unusable Copilot authentication."""


@dataclass(frozen=True, slots=True)
class CopilotDeviceCode:
    device_code: str = field(repr=False)
    user_code: str = field(repr=False)
    verification_uri: str
    expires_in: float
    interval: float
    # Monotonic deadline includes time spent requesting the code and displaying it.
    deadline: float = field(repr=False)


class _OAuthResponse(BaseModel):
    model_config = ConfigDict(strict=True, hide_input_in_errors=True)


class _DeviceResponse(_OAuthResponse):
    device_code: _NonemptyString = Field(repr=False)
    user_code: Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9-]+$")] = Field(repr=False)
    # Deliberately reject query strings, fragments, userinfo, and lookalike hosts.
    verification_uri: Literal["https://github.com/login/device"]
    expires_in: _PositiveSeconds
    interval: _PositiveSeconds = 5


class _TokenResponse(_OAuthResponse):
    error: None = None
    access_token: Annotated[str, Field(min_length=1, pattern=r"^[!-~]+$")] = Field(repr=False)
    token_type: Literal["bearer", "Bearer"]
    scope: str | None = None
    expires_in: _PositiveSeconds | None = None


class _ErrorResponse(_OAuthResponse):
    error: str
    interval: _PositiveSeconds | None = None


_TOKEN_RESPONSE = TypeAdapter(_ErrorResponse | _TokenResponse)


class _CopilotCredential(OAuthCredential):
    """Keep the shared store contract while redacting credential representations."""

    access_token: str = Field(repr=False)
    refresh_token: str | None = Field(default=None, repr=False)


def _expired_device() -> CopilotAuthenticationError:
    return CopilotAuthenticationError("Copilot device code expired.", _LOGIN_HINT)


def _http_error(response: httpx.Response) -> CopilotAuthenticationError:
    return CopilotAuthenticationError(
        f"GitHub device authorization failed (HTTP {response.status_code}).", _LOGIN_HINT
    )


async def request_copilot_device_code(client: httpx.AsyncClient) -> CopilotDeviceCode:
    started = time.monotonic()
    try:
        response = await client.post(
            _DEVICE_URL,
            data={"client_id": COPILOT_CLIENT_ID, "scope": "read:user"},
            headers={"Accept": "application/json"},
            follow_redirects=False,
        )
    except httpx.RequestError:
        raise CopilotAuthenticationError(
            "Unable to contact GitHub for device authorization."
        ) from None
    if not response.is_success:
        raise _http_error(response)
    try:
        payload = _DeviceResponse.model_validate_json(response.content)
    except ValidationError:
        raise CopilotAuthenticationError("Invalid GitHub device authorization response.") from None
    deadline = started + payload.expires_in
    if time.monotonic() >= deadline:
        raise _expired_device()
    return CopilotDeviceCode(
        device_code=payload.device_code,
        user_code=payload.user_code,
        verification_uri=payload.verification_uri,
        expires_in=payload.expires_in,
        interval=payload.interval,
        deadline=deadline,
    )


async def poll_copilot_device_code(
    client: httpx.AsyncClient, device: CopilotDeviceCode
) -> OAuthCredential:
    remaining = device.deadline - time.monotonic()
    if remaining <= 0:
        raise _expired_device()
    interval = device.interval
    try:
        # One deadline bounds both sleep and in-flight HTTP, not just each iteration.
        async with asyncio.timeout(remaining):
            while True:
                await asyncio.sleep(min(interval, device.deadline - time.monotonic()))
                if time.monotonic() >= device.deadline:
                    raise _expired_device()
                try:
                    response = await client.post(
                        _TOKEN_URL,
                        data={
                            "client_id": COPILOT_CLIENT_ID,
                            "device_code": device.device_code,
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        },
                        headers={"Accept": "application/json"},
                        follow_redirects=False,
                    )
                except httpx.TimeoutException:
                    # RFC 8628: reduce polling frequency on connection timeouts.
                    interval *= 2
                    continue
                except httpx.RequestError:
                    raise CopilotAuthenticationError(
                        "Unable to contact GitHub for device authorization."
                    ) from None
                if time.monotonic() >= device.deadline:
                    raise _expired_device()
                # GitHub returns OAuth errors with HTTP 200 as well as HTTP 400.
                if not response.is_success and response.status_code != 400:
                    raise _http_error(response)
                try:
                    payload = _TOKEN_RESPONSE.validate_json(response.content)
                except ValidationError:
                    raise CopilotAuthenticationError(
                        "Invalid GitHub device token response."
                    ) from None
                if isinstance(payload, _ErrorResponse):
                    if payload.error == "authorization_pending":
                        continue
                    if payload.error == "slow_down":
                        interval = max(interval + 5, payload.interval or 0)
                        continue
                    if payload.error in {"access_denied", "authorization_denied"}:
                        raise CopilotAuthenticationError("Copilot device authorization denied.")
                    if payload.error == "expired_token":
                        raise _expired_device()
                    raise CopilotAuthenticationError(
                        "GitHub rejected Copilot device authorization.", _LOGIN_HINT
                    )
                if not response.is_success:
                    raise _http_error(response)
                return _CopilotCredential(
                    access_token=payload.access_token,
                    token_type=payload.token_type,
                    scope=payload.scope,
                    expires_at=(
                        time.time() + payload.expires_in if payload.expires_in is not None else None
                    ),
                )
    except TimeoutError:
        raise _expired_device() from None


@contextmanager
def _store_errors() -> Iterator[None]:
    # ValidationError/JSONDecodeError/UnicodeError are ValueErrors. Never expose
    # their model dumps, input values, or file contents in user-facing failures.
    try:
        yield
    except (ValueError, OSError):
        raise ProviderKeyError(
            "Unable to read or update the Copilot credential store.",
            "Check the fast-agent credential file and its permissions.",
        ) from None


async def login_copilot_oauth_async() -> OAuthCredential:
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as client:
        device = await request_copilot_device_code(client)
        console.ensure_blocking_console()
        console.console.print(
            f"Open {_VERIFICATION_URI} and enter code {device.user_code}.", markup=False
        )
        credential = await poll_copilot_device_code(client, device)
    # Cancellation at any await above propagates without saving.
    with _store_errors():
        save_oauth_credential(COPILOT_PROVIDER_ID, credential)
    return credential


def login_copilot_oauth() -> OAuthCredential:
    return asyncio.run(login_copilot_oauth_async())


def _valid_access_token(token: str) -> bool:
    return bool(token) and all("!" <= character <= "~" for character in token)


def _environment_credential() -> OAuthCredential | None:
    token = os.environ.get("COPILOT_GITHUB_TOKEN")
    if token is None:
        return None
    if not _valid_access_token(token):
        raise ProviderKeyError(
            "Invalid COPILOT_GITHUB_TOKEN.",
            "Set a printable nonspace ASCII GitHub token or unset COPILOT_GITHUB_TOKEN "
            "to use saved credentials.",
        )
    return _CopilotCredential(access_token=token)


def _load_validated_credential() -> StoredCredential | None:
    with _store_errors():
        stored = load_oauth_credential(COPILOT_PROVIDER_ID)
    if stored is not None:
        credential = stored.credential
        if not _valid_access_token(credential.access_token) or (
            credential.expires_at is not None and not math.isfinite(credential.expires_at)
        ):
            raise ProviderKeyError(
                "Invalid saved Copilot OAuth credential.",
                "Check the fast-agent credential store.",
            )
    return stored


def get_copilot_credential() -> OAuthCredential | None:
    """Resolve validated credentials, preferring the environment and rejecting expiry."""
    environment = _environment_credential()
    if environment is not None:
        return environment
    stored = _load_validated_credential()
    if stored is None:
        return None
    credential = stored.credential
    if credential.expires_at is not None and time.time() >= credential.expires_at:
        raise CopilotAuthenticationError("Copilot OAuth token expired.", _LOGIN_HINT)
    return credential


def get_copilot_access_token() -> str | None:
    credential = get_copilot_credential()
    return credential.access_token if credential is not None else None


def get_copilot_token_status() -> dict[str, object]:
    if _environment_credential() is not None:
        return {"present": True, "source": "environment", "expires_at": None, "expired": False}
    stored = _load_validated_credential()
    expires_at = stored.credential.expires_at if stored else None
    return {
        "present": stored is not None,
        "source": stored.source if stored else None,
        "expires_at": expires_at,
        "expired": expires_at is not None and time.time() >= expires_at,
    }


def clear_copilot_tokens() -> bool:
    """Clear fast-agent's saved credential; environment variables are not modified."""
    with _store_errors():
        return delete_oauth_credential(COPILOT_PROVIDER_ID)
