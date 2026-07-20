from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from fast_agent.auth.credentials import (
    OAuthCredential,
    credential_refresh_lock,
    delete_oauth_credential,
    load_oauth_credential,
    save_oauth_credential,
)
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.ui import console

if TYPE_CHECKING:
    from collections.abc import Callable

XAI_PROVIDER_ID = "xai"
XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
XAI_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
REFRESH_SKEW_SECONDS = 300
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600
DEFAULT_POLL_INTERVAL_SECONDS = 5


@dataclass(frozen=True, slots=True)
class XaiDeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: float
    interval: float


class _InvalidGrantError(ProviderKeyError):
    pass


def _oauth_error(action: str, response: httpx.Response) -> ProviderKeyError:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    description = payload.get("error_description") if isinstance(payload, dict) else None
    detail = ": ".join(value for value in (error, description) if isinstance(value, str))
    suffix = f": {detail}" if detail else ""
    return ProviderKeyError(
        f"xAI OAuth {action} failed",
        f"The xAI OAuth server returned HTTP {response.status_code}{suffix}.",
    )


def _oauth_error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, str) else None


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ProviderKeyError(
            "Invalid xAI OAuth response",
            f"The xAI OAuth response did not include a valid {field}.",
        )
    return value


def _positive_number(payload: dict[str, Any], field: str, default: float) -> float:
    value = payload.get(field)
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        else default
    )


def _credential_from_payload(
    payload: dict[str, Any],
    *,
    previous_refresh_token: str | None = None,
) -> OAuthCredential:
    refresh_token = payload.get("refresh_token", previous_refresh_token)
    if not isinstance(refresh_token, str) or not refresh_token:
        raise ProviderKeyError(
            "Invalid xAI OAuth response",
            "The xAI OAuth response did not include a refresh token.",
        )
    expires_in = _positive_number(
        payload,
        "expires_in",
        DEFAULT_TOKEN_LIFETIME_SECONDS,
    )
    return OAuthCredential(
        access_token=_required_string(payload, "access_token"),
        refresh_token=refresh_token,
        expires_at=time.time() + expires_in - REFRESH_SKEW_SECONDS,
        token_type=str(payload.get("token_type") or "Bearer"),
        scope=payload.get("scope") if isinstance(payload.get("scope"), str) else None,
    )


def request_xai_device_code(client: httpx.Client) -> XaiDeviceCode:
    response = client.post(
        XAI_DEVICE_CODE_URL,
        data={"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPE, "referrer": "fast-agent"},
    )
    if not response.is_success:
        raise _oauth_error("device authorization", response)
    payload = response.json()
    if not isinstance(payload, dict):
        raise ProviderKeyError("Invalid xAI OAuth response", "Expected a JSON object.")
    verification_uri = _required_string(payload, "verification_uri")
    if not verification_uri.startswith("https://"):
        raise ProviderKeyError(
            "Invalid xAI OAuth response",
            "The verification URI was not a trusted HTTPS URL.",
        )
    return XaiDeviceCode(
        device_code=_required_string(payload, "device_code"),
        user_code=_required_string(payload, "user_code"),
        verification_uri=verification_uri,
        expires_in=_positive_number(payload, "expires_in", 900),
        interval=_positive_number(payload, "interval", DEFAULT_POLL_INTERVAL_SECONDS),
    )


def poll_xai_device_code(
    client: httpx.Client,
    device: XaiDeviceCode,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> OAuthCredential:
    deadline = monotonic() + device.expires_in
    interval = device.interval
    while monotonic() < deadline:
        sleep(interval)
        response = client.post(
            XAI_TOKEN_URL,
            data={
                "grant_type": XAI_DEVICE_GRANT,
                "client_id": XAI_CLIENT_ID,
                "device_code": device.device_code,
            },
        )
        if response.is_success:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ProviderKeyError("Invalid xAI OAuth response", "Expected a JSON object.")
            return _credential_from_payload(payload)
        try:
            payload = response.json()
        except ValueError:
            raise _oauth_error("device token polling", response) from None
        error = payload.get("error") if isinstance(payload, dict) else None
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval = _positive_number(payload, "interval", interval + 5)
            continue
        if error in {"access_denied", "authorization_denied"}:
            raise ProviderKeyError(
                "xAI OAuth login denied",
                "The xAI device authorization was denied.",
            )
        if error == "expired_token":
            break
        raise _oauth_error("device token polling", response)
    raise ProviderKeyError("xAI OAuth login expired", "The xAI device code expired.")


def refresh_xai_credential(
    credential: OAuthCredential,
    *,
    client: httpx.Client | None = None,
) -> OAuthCredential:
    if not credential.refresh_token:
        raise ProviderKeyError(
            "xAI OAuth token expired",
            "No refresh token is available. Run `fast-agent auth login xai`.",
        )
    owns_client = client is None
    resolved_client = client or httpx.Client(timeout=30)
    try:
        response = resolved_client.post(
            XAI_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": XAI_CLIENT_ID,
                "refresh_token": credential.refresh_token,
            },
        )
        if not response.is_success:
            error = _oauth_error("token refresh", response)
            if _oauth_error_code(response) == "invalid_grant":
                raise _InvalidGrantError(error.message, error.details)
            raise error
        payload = response.json()
        if not isinstance(payload, dict):
            raise ProviderKeyError("Invalid xAI OAuth response", "Expected a JSON object.")
        return _credential_from_payload(
            payload,
            previous_refresh_token=credential.refresh_token,
        )
    finally:
        if owns_client:
            resolved_client.close()


def login_xai_oauth(
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> OAuthCredential:
    owns_client = client is None
    resolved_client = client or httpx.Client(timeout=30)
    try:
        device = request_xai_device_code(resolved_client)
        console.ensure_blocking_console()
        console.console.print(
            f"Open [link={device.verification_uri}]{device.verification_uri}[/link] "
            f"and enter code [bold]{device.user_code}[/bold]."
        )
        credential = poll_xai_device_code(resolved_client, device, sleep=sleep)
    finally:
        if owns_client:
            resolved_client.close()
    save_oauth_credential(XAI_PROVIDER_ID, credential)
    return credential


def get_xai_access_token(
    *,
    force_refresh: bool = False,
    client: httpx.Client | None = None,
) -> str | None:
    stored = load_oauth_credential(XAI_PROVIDER_ID)
    if stored is None:
        return None
    credential = stored.credential
    expired = credential.expires_at is not None and time.time() >= credential.expires_at
    if force_refresh or expired:
        with credential_refresh_lock(XAI_PROVIDER_ID):
            current = load_oauth_credential(XAI_PROVIDER_ID)
            if current is None:
                return None
            credential = current.credential
            expired = credential.expires_at is not None and time.time() >= credential.expires_at
            if force_refresh or expired:
                try:
                    credential = refresh_xai_credential(credential, client=client)
                except _InvalidGrantError:
                    delete_oauth_credential(XAI_PROVIDER_ID)
                    return None
                save_oauth_credential(
                    XAI_PROVIDER_ID,
                    credential,
                    source=current.source,
                )
    return credential.access_token


def get_xai_token_status() -> dict[str, object]:
    stored = load_oauth_credential(XAI_PROVIDER_ID)
    if stored is None:
        return {"present": False, "source": None, "expires_at": None, "expired": False}
    expires_at = stored.credential.expires_at
    return {
        "present": True,
        "source": stored.source,
        "expires_at": expires_at,
        "expired": expires_at is not None and time.time() >= expires_at,
    }


def clear_xai_tokens() -> bool:
    return delete_oauth_credential(XAI_PROVIDER_ID)
