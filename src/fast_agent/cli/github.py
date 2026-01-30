"""GitHub helper routines for CLI actions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Final
from urllib import error, request

DEFAULT_REPO: Final[str] = "evalstate/fast-agent"
DEFAULT_REPO_URL: Final[str] = f"https://github.com/{DEFAULT_REPO}"
DEFAULT_STAR_API_URL: Final[str] = f"https://api.github.com/user/starred/{DEFAULT_REPO}"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
TOKEN_ENV_VARS: Final[tuple[str, ...]] = (
    "FAST_AGENT_GITHUB_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)


@dataclass(frozen=True)
class StarResult:
    ok: bool
    status_code: int | None
    message: str
    already_starred: bool = False
    token_source: str | None = None


def _resolve_github_token(explicit_token: str | None) -> tuple[str | None, str | None]:
    if explicit_token is not None:
        stripped = explicit_token.strip()
        return (stripped, "explicit") if stripped else (None, None)

    for env_name in TOKEN_ENV_VARS:
        token = os.getenv(env_name)
        if token:
            stripped = token.strip()
            if stripped:
                return stripped, env_name
    return None, None


def _extract_api_message(payload: str) -> str | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def star_repository(
    repo_api_url: str = DEFAULT_STAR_API_URL,
    repo_url: str = DEFAULT_REPO_URL,
    *,
    token: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> StarResult:
    resolved_token, token_source = _resolve_github_token(token)
    if resolved_token is None:
        return StarResult(
            ok=False,
            status_code=None,
            message=(
                "No GitHub token found. Set FAST_AGENT_GITHUB_TOKEN, GITHUB_TOKEN, or "
                "GH_TOKEN with the 'public_repo' scope (classic) or Starring permission "
                f"(fine-grained), or visit {repo_url}."
            ),
        )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {resolved_token}",
        "User-Agent": "fast-agent",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    request_obj = request.Request(repo_api_url, method="PUT", headers=headers)

    try:
        with request.urlopen(request_obj, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            if status_code == 204:
                return StarResult(
                    ok=True,
                    status_code=status_code,
                    message=f"Starred {repo_url}. Thanks for the support!",
                    token_source=token_source,
                )
            if status_code == 304:
                return StarResult(
                    ok=True,
                    status_code=status_code,
                    message=f"{repo_url} is already starred.",
                    already_starred=True,
                    token_source=token_source,
                )
            return StarResult(
                ok=False,
                status_code=status_code,
                message=f"GitHub API returned unexpected status {status_code}.",
                token_source=token_source,
            )
    except error.HTTPError as exc:
        status_code = exc.code
        payload = exc.read().decode("utf-8", errors="replace")
        api_message = _extract_api_message(payload)
        if status_code in (401, 403):
            message = (
                api_message
                or "GitHub rejected the token. Ensure it has the 'public_repo' scope "
                "(classic) or Starring permission (fine-grained)."
            )
        elif status_code == 404:
            message = api_message or "Repository not found or token lacks access."
        else:
            message = api_message or f"GitHub API error {status_code}."
        return StarResult(
            ok=False,
            status_code=status_code,
            message=message,
            token_source=token_source,
        )
    except error.URLError as exc:
        return StarResult(
            ok=False,
            status_code=None,
            message=f"Network error contacting GitHub: {exc.reason}",
            token_source=token_source,
        )
