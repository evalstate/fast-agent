from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from fast_agent.auth.credentials import OAuthCredential, export_oauth_credential
from fast_agent.core.exceptions import ProviderKeyError

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProviderAuthStatus:
    provider: str
    display_name: str
    present: bool
    source: str | None
    expires_at: float | None
    expired: bool


@dataclass(frozen=True, slots=True)
class OAuthProvider:
    id: str
    display_name: str
    login: Callable[[], OAuthCredential]
    credential: Callable[[], OAuthCredential | None]
    access_token: Callable[[], str | None]
    status: Callable[[], dict[str, object]]
    logout: Callable[[], bool]


def _xai_provider() -> OAuthProvider:
    from fast_agent.auth.credentials import load_oauth_credential
    from fast_agent.llm.provider.openai.xai_oauth import (
        clear_xai_tokens,
        get_xai_access_token,
        get_xai_token_status,
        login_xai_oauth,
    )

    def credential() -> OAuthCredential | None:
        stored = load_oauth_credential("xai")
        return stored.credential if stored else None

    return OAuthProvider(
        id="xai",
        display_name="xAI",
        login=login_xai_oauth,
        credential=credential,
        access_token=get_xai_access_token,
        status=get_xai_token_status,
        logout=clear_xai_tokens,
    )


def _codex_provider() -> OAuthProvider:
    from fast_agent.llm.provider.openai.codex_oauth import (
        CodexOAuthTokens,
        clear_codex_tokens,
        get_codex_access_token,
        get_codex_token_status,
        load_codex_tokens,
        login_codex_oauth,
    )

    def login() -> OAuthCredential:
        return _codex_credential(login_codex_oauth())

    def credential() -> OAuthCredential | None:
        tokens = load_codex_tokens()
        return _codex_credential(tokens) if tokens else None

    def _codex_credential(tokens: CodexOAuthTokens) -> OAuthCredential:
        return OAuthCredential(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            expires_at=tokens.expires_at,
            scope=tokens.scope,
            token_type=tokens.token_type,
        )

    return OAuthProvider(
        id="codex",
        display_name="Codex",
        login=login,
        credential=credential,
        access_token=get_codex_access_token,
        status=get_codex_token_status,
        logout=clear_codex_tokens,
    )


def provider_ids() -> tuple[str, ...]:
    return ("xai", "codex")


def get_oauth_provider(provider: str) -> OAuthProvider:
    normalized = provider.strip().casefold()
    if normalized == "xai":
        return _xai_provider()
    if normalized in {"codex", "codexplan", "codexresponses"}:
        return _codex_provider()
    raise ProviderKeyError(
        "Unsupported OAuth provider",
        f"Choose one of: {', '.join(provider_ids())}.",
    )


def provider_status(provider: str) -> ProviderAuthStatus:
    handler = get_oauth_provider(provider)
    status = handler.status()
    source_value = status.get("source")
    expires_value = status.get("expires_at")
    return ProviderAuthStatus(
        provider=handler.id,
        display_name=handler.display_name,
        present=bool(status.get("present")),
        source=source_value if isinstance(source_value, str) else None,
        expires_at=(
            float(expires_value)
            if isinstance(expires_value, (int, float))
            and not isinstance(expires_value, bool)
            else None
        ),
        expired=bool(status.get("expired")),
    )


def export_provider_credential(provider: str, path: Path) -> None:
    handler = get_oauth_provider(provider)
    token = handler.access_token()
    credential = handler.credential()
    if token is None or credential is None:
        raise ProviderKeyError(
            f"{handler.display_name} OAuth token not configured",
            f"Run `fast-agent auth login {handler.id}` first.",
        )
    export_oauth_credential(handler.id, credential, path)
