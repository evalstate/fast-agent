"""OAuth 2.0 Device Authorization Grant (RFC 8628) for MCP servers.

A no-browser alternative to the authorization-code + PKCE flow in ``oauth_client``.
Discovery, client registration, resource selection and token storage are shared with
the MCP OAuth provider so that tokens obtained here are picked up (and refreshed) by
normal MCP connections without any extra wiring.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Final

import httpx2 as httpx
from mcp.client.auth import OAuthFlowError
from mcp.client.auth.oauth2 import check_registration_usable
from mcp.client.auth.utils import (
    create_client_info_from_metadata_url,
    create_client_registration_request,
    credentials_match_issuer,
    get_client_metadata_scopes,
    handle_registration_response,
    should_use_client_metadata_url,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError

from fast_agent.mcp.oauth_client import build_oauth_provider, compute_server_identity

if TYPE_CHECKING:
    from mcp.client.auth import TokenStorage
    from mcp.shared.auth import OAuthMetadata

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.oauth_client import OAuthClientProvider

DEVICE_CODE_GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:device_code"
_SLOW_DOWN_INCREMENT_SECONDS: Final = 5.0
_HTTP_TIMEOUT_SECONDS: Final = 30.0

_PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
_NonemptyString = Annotated[str, Field(min_length=1)]


class MCPDeviceAuthorizationError(RuntimeError):
    """Device authorization could not be completed."""


class MCPDeviceAuthorizationUnsupportedError(MCPDeviceAuthorizationError):
    """The server's authorization server does not offer a usable device flow."""


class MCPDeviceAuthorizationDeniedError(MCPDeviceAuthorizationError):
    """The user denied the device authorization request."""


class MCPDeviceAuthorizationExpiredError(MCPDeviceAuthorizationError):
    """The device code expired before the user approved it."""


@dataclass(frozen=True, slots=True)
class MCPDeviceCode:
    """What a surface must show the user to complete device authorization."""

    server_name: str
    user_code: str = field(repr=False)
    verification_uri: str
    # Embeds the user code; hidden from reprs along with it.
    verification_uri_complete: str | None = field(repr=False)
    expires_in: float
    interval: float


@dataclass(frozen=True, slots=True)
class MCPDeviceLoginResult:
    server_name: str
    # Token-storage identity (the same key `fast-agent auth mcp` reports).
    resource: str
    scope: str | None
    expires_in: int | None


type DeviceCodeHandler = Callable[[MCPDeviceCode], Awaitable[None]]
type SleepFn = Callable[[float], Awaitable[None]]


class _ResponseModel(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)


class _DeviceMetadata(_ResponseModel):
    device_authorization_endpoint: AnyHttpUrl | None = None


class _DeviceAuthorizationResponse(_ResponseModel):
    device_code: _NonemptyString = Field(repr=False)
    user_code: _NonemptyString = Field(repr=False)
    verification_uri: AnyHttpUrl
    verification_uri_complete: AnyHttpUrl | None = Field(default=None, repr=False)
    expires_in: _PositiveSeconds
    interval: _PositiveSeconds = 5.0


class _OAuthErrorResponse(_ResponseModel):
    error: _NonemptyString
    error_description: str | None = None


@dataclass(frozen=True, slots=True)
class DeviceTokenRequest:
    """A pending device authorization: where and how to poll for its token."""

    token_endpoint: str
    # Form fields/headers already carry client authentication and the RFC 8707 resource.
    data: dict[str, str] = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    interval: float
    # Monotonic deadline; includes time spent requesting and displaying the code.
    deadline: float


def _describe_oauth_error(payload: _OAuthErrorResponse) -> str:
    if payload.error_description:
        return f"{payload.error}: {payload.error_description}"
    return payload.error


def _oauth_error_or_none(response: httpx.Response) -> _OAuthErrorResponse | None:
    try:
        return _OAuthErrorResponse.model_validate_json(response.content)
    except ValidationError:
        return None


async def poll_device_token(
    client: httpx.AsyncClient,
    request: DeviceTokenRequest,
    *,
    sleep: SleepFn = asyncio.sleep,
) -> OAuthToken:
    """Poll the token endpoint until the user approves, denies, or the code expires.

    Honors ``interval`` and ``slow_down`` (RFC 8628 §3.5). Cancellation propagates
    immediately; nothing is persisted by this function.
    """
    interval = request.interval
    remaining = request.deadline - time.monotonic()
    if remaining <= 0:
        raise MCPDeviceAuthorizationExpiredError("Device code expired before approval.")
    try:
        # One deadline bounds both the sleeps and any in-flight HTTP request.
        async with asyncio.timeout(remaining):
            while True:
                await sleep(interval)
                try:
                    response = await client.post(
                        request.token_endpoint,
                        data=request.data,
                        headers=request.headers,
                    )
                except httpx.TimeoutException:
                    # RFC 8628 §3.5: back off on connection timeouts.
                    interval *= 2
                    continue
                except httpx.RequestError as exc:
                    raise MCPDeviceAuthorizationError(
                        f"Unable to contact the token endpoint: {type(exc).__name__}"
                    ) from None

                if response.status_code == 200:
                    try:
                        return OAuthToken.model_validate_json(response.content)
                    except ValidationError:
                        raise MCPDeviceAuthorizationError(
                            "Invalid token response from authorization server."
                        ) from None

                error = _oauth_error_or_none(response)
                if error is None:
                    raise MCPDeviceAuthorizationError(
                        f"Token endpoint returned HTTP {response.status_code}."
                    )
                match error.error:
                    case "authorization_pending":
                        continue
                    case "slow_down":
                        interval += _SLOW_DOWN_INCREMENT_SECONDS
                        continue
                    case "access_denied":
                        raise MCPDeviceAuthorizationDeniedError("Device authorization was denied.")
                    case "expired_token":
                        raise MCPDeviceAuthorizationExpiredError(
                            "Device code expired before approval."
                        )
                    case _:
                        raise MCPDeviceAuthorizationError(
                            f"Device authorization failed ({_describe_oauth_error(error)})."
                        )
    except TimeoutError:
        raise MCPDeviceAuthorizationExpiredError("Device code expired before approval.") from None


async def _discover_device_endpoint(
    provider: OAuthClientProvider,
    client: httpx.AsyncClient,
    server_name: str,
) -> tuple[OAuthMetadata, str]:
    raw_metadata = await provider.discover_authorization_server(client)
    metadata = provider.context.oauth_metadata
    if raw_metadata is None or metadata is None:
        raise MCPDeviceAuthorizationUnsupportedError(
            f"Could not discover OAuth authorization server metadata for '{server_name}'."
        )
    device_endpoint = _DeviceMetadata.model_validate_json(
        raw_metadata
    ).device_authorization_endpoint
    if device_endpoint is None:
        raise MCPDeviceAuthorizationUnsupportedError(
            f"The authorization server for '{server_name}' ({metadata.issuer}) does not "
            "advertise a device_authorization_endpoint; use browser login instead."
        )
    return metadata, str(device_endpoint)


def _reusable_client(
    client_info: OAuthClientInformationFull | None,
    *,
    issuer: str,
    client_metadata_url: str | None,
) -> bool:
    return (
        client_info is not None
        and client_info.client_id != client_metadata_url
        and DEVICE_CODE_GRANT_TYPE in (client_info.grant_types or [])
        and credentials_match_issuer(client_info, issuer, client_metadata_url)
    )


async def _resolve_client(
    provider: OAuthClientProvider,
    client: httpx.AsyncClient,
    *,
    storage: TokenStorage,
    metadata: OAuthMetadata,
    issuer: str,
    server_name: str,
) -> OAuthClientInformationFull:
    """Pick a client for the device grant without writing to storage.

    Prefers a stored dynamically-registered client that declared the device grant, then
    registers a new one (declaring the device grant) and finally falls back to a Client
    ID Metadata Document. The result is only persisted once a token is issued, so an
    abandoned login never orphans stored tokens from their client.
    """
    context = provider.context
    stored = await storage.get_client_info()
    if _reusable_client(stored, issuer=issuer, client_metadata_url=context.client_metadata_url):
        assert stored is not None
        return stored

    if metadata.registration_endpoint is not None:
        # RFC 8628 §5.6: a device-flow client cannot keep a secret, so register as a
        # public client; servers otherwise default to a secret-based auth method.
        client_metadata = context.client_metadata.model_copy(
            update={
                "grant_types": [*context.client_metadata.grant_types, DEVICE_CODE_GRANT_TYPE],
                "token_endpoint_auth_method": "none",
            }
        )
        registration = create_client_registration_request(
            metadata,
            client_metadata,
            context.get_authorization_base_url(context.server_url),
        )
        client_info = await handle_registration_response(await client.send(registration))
        check_registration_usable(client_info)
        client_info.issuer = issuer
        return client_info

    if (
        should_use_client_metadata_url(metadata, context.client_metadata_url)
        and context.client_metadata_url is not None
    ):
        client_info = create_client_info_from_metadata_url(
            context.client_metadata_url,
            redirect_uris=context.client_metadata.redirect_uris,
        )
        client_info.issuer = issuer
        return client_info

    raise MCPDeviceAuthorizationUnsupportedError(
        f"The authorization server for '{server_name}' supports neither dynamic client "
        "registration nor client ID metadata documents, so no client is available for "
        "device authorization."
    )


async def _request_device_code(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    data: dict[str, str],
    headers: dict[str, str],
) -> _DeviceAuthorizationResponse:
    try:
        response = await client.post(endpoint, data=data, headers=headers)
    except httpx.RequestError as exc:
        raise MCPDeviceAuthorizationError(
            f"Unable to contact the device authorization endpoint: {type(exc).__name__}"
        ) from None
    if response.status_code != 200:
        error = _oauth_error_or_none(response)
        detail = (
            _describe_oauth_error(error) if error is not None else f"HTTP {response.status_code}"
        )
        raise MCPDeviceAuthorizationError(f"Device authorization request failed ({detail}).")
    try:
        return _DeviceAuthorizationResponse.model_validate_json(response.content)
    except ValidationError:
        raise MCPDeviceAuthorizationError(
            "Invalid device authorization response from authorization server."
        ) from None


async def _authorize(
    provider: OAuthClientProvider,
    client: httpx.AsyncClient,
    *,
    storage: TokenStorage,
    server_config: MCPServerSettings,
    server_name: str,
    on_user_code: DeviceCodeHandler,
    sleep: SleepFn,
) -> tuple[OAuthClientInformationFull, OAuthToken]:
    context = provider.context
    metadata, device_endpoint = await _discover_device_endpoint(provider, client, server_name)
    if (
        metadata.grant_types_supported is not None
        and DEVICE_CODE_GRANT_TYPE not in metadata.grant_types_supported
    ):
        raise MCPDeviceAuthorizationUnsupportedError(
            f"The authorization server for '{server_name}' does not support the "
            "device_code grant type; use browser login instead."
        )
    issuer = context.auth_server_url or str(metadata.issuer)
    client_info = await _resolve_client(
        provider,
        client,
        storage=storage,
        metadata=metadata,
        issuer=issuer,
        server_name=server_name,
    )
    # The SDK context applies token-endpoint client authentication from client_info.
    context.client_info = client_info

    configured_scope = server_config.auth.scope if server_config.auth else None
    scope = (
        " ".join(configured_scope) if isinstance(configured_scope, list) else configured_scope
    ) or get_client_metadata_scopes(
        None,
        context.protected_resource_metadata,
        metadata,
        client_info.grant_types,
    )
    common: dict[str, str] = {"client_id": client_info.client_id}
    # Same RFC 8707 resource selection as the authorization-code flow.
    if context.should_include_resource_param(None):
        common["resource"] = context.get_resource_url()

    device_data = {**common, "scope": scope} if scope else dict(common)
    device_data, device_headers = context.prepare_token_auth(
        device_data, {"Accept": "application/json"}
    )
    started = time.monotonic()
    device = await _request_device_code(
        client, device_endpoint, data=device_data, headers=device_headers
    )
    await on_user_code(
        MCPDeviceCode(
            server_name=server_name,
            user_code=device.user_code,
            verification_uri=str(device.verification_uri),
            verification_uri_complete=(
                str(device.verification_uri_complete)
                if device.verification_uri_complete is not None
                else None
            ),
            expires_in=device.expires_in,
            interval=device.interval,
        )
    )

    token_data, token_headers = context.prepare_token_auth(
        {**common, "grant_type": DEVICE_CODE_GRANT_TYPE, "device_code": device.device_code},
        {"Accept": "application/json"},
    )
    token = await poll_device_token(
        client,
        DeviceTokenRequest(
            token_endpoint=str(metadata.token_endpoint),
            data=token_data,
            headers=token_headers,
            interval=device.interval,
            deadline=started + device.expires_in,
        ),
        sleep=sleep,
    )
    # RFC 6749 §5.1: an omitted scope equals the requested scope. Record it so the stored
    # token is self-describing, matching the authorization-code flow.
    if token.scope is None:
        token.scope = scope
    return client_info, token


async def login_mcp_server_with_device_code(
    server_config: MCPServerSettings,
    *,
    on_user_code: DeviceCodeHandler,
    storage: TokenStorage | None = None,
    http_client: httpx.AsyncClient | None = None,
    sleep: SleepFn = asyncio.sleep,
) -> MCPDeviceLoginResult:
    """Authenticate an MCP server with the OAuth device authorization grant.

    ``on_user_code`` is awaited once with the code and verification URI to show the user.
    Tokens and the client registration are written to the same storage the MCP OAuth
    provider reads (keyring or memory per ``auth.persist``), unless ``storage`` is given
    (e.g. to share an in-memory store with a live connection). Cancel the awaiting task
    to abort; nothing is stored unless a token is issued. A supplied ``http_client``
    must not follow redirects and is not closed.

    Raises:
        MCPDeviceAuthorizationError: For every non-cancellation failure; subclasses
            distinguish unsupported servers, denial and expiry.
    """
    provider = build_oauth_provider(server_config, emit_console_output=False)
    server_name = server_config.name or "default"
    if provider is None:
        raise MCPDeviceAuthorizationError(
            f"OAuth is not available for '{server_name}' (requires an HTTP/SSE URL "
            "with OAuth enabled)."
        )
    token_storage = storage or provider.context.storage

    try:
        async with (
            httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS, follow_redirects=False)
            if http_client is None
            else nullcontext(http_client)
        ) as client:
            client_info, token = await _authorize(
                provider,
                client,
                storage=token_storage,
                server_config=server_config,
                server_name=server_name,
                on_user_code=on_user_code,
                sleep=sleep,
            )
    except OAuthFlowError as exc:
        # Discovery issuer mismatches and dynamic client registration failures.
        raise MCPDeviceAuthorizationError(str(exc)) from exc
    except httpx.RequestError as exc:
        raise MCPDeviceAuthorizationError(
            f"Unable to contact the authorization server: {type(exc).__name__}"
        ) from None

    await token_storage.set_client_info(client_info)
    await token_storage.set_tokens(token)
    return MCPDeviceLoginResult(
        server_name=server_name,
        resource=compute_server_identity(server_config),
        scope=token.scope,
        expires_in=token.expires_in,
    )
