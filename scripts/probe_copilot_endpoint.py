# /// script
# requires-python = ">=3.12"
# dependencies = ["github-copilot-sdk==1.0.14", "httpx==0.28.1"]
# ///
"""Discover Copilot's experimental direct-inference endpoint without printing secrets.

Run with uv run scripts/probe_copilot_endpoint.py --help.
No inference is performed unless --infer is supplied. See docs/copilot-probe.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import httpx

TOKEN_ENV_NAMES = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


class ProbeError(Exception):
    """A diagnostic safe to show without credentials or raw server payloads."""


@dataclass(frozen=True)
class Endpoint:
    provider: str
    wire_api: str
    transport: str
    base_url: str = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    api_key: str | None = field(default=None, repr=False)
    session_token: str | None = field(default=None, repr=False)
    session_token_header: str | None = None
    session_token_model: str | None = None

    def summary(self) -> dict[str, object]:
        # Never print a credential-bearing URL, header value, or SDK object repr.
        url = urlsplit(self.base_url)
        return {
            "provider": self.provider,
            "wire_api": self.wire_api,
            "transport": self.transport,
            "host": url.hostname,
            "has_api_key": bool(self.api_key),
            "header_names": sorted(self.headers),
            "has_session_token": bool(self.session_token),
        }


def runtime_environment(use_env_token: bool) -> tuple[dict[str, str], str | None, str]:
    env = dict(os.environ)
    token = None
    source = "Copilot CLI stored login"
    if use_env_token:
        for name in TOKEN_ENV_NAMES:
            if env.get(name):
                token, source = env[name], name
                break
        if not token:
            raise ProbeError(
                "--use-env-token requires COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN."
            )
    # Avoid silently testing a different identity, a BYOK token, or a proxy override.
    for name in (*TOKEN_ENV_NAMES, "GITHUB_COPILOT_API_TOKEN", "COPILOT_API_URL"):
        env.pop(name, None)
    env["COPILOT_ALLOW_GET_PROVIDER_ENDPOINT"] = "true"
    return env, token, source


def inference_request(
    endpoint: Endpoint, model: str, max_tokens: int
) -> tuple[str, dict[str, str], dict[str, object]]:
    url = urlsplit(endpoint.base_url)
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ProbeError(
            "Refusing inference: expected an HTTPS base URL without credentials, query, or fragment."
        )
    if endpoint.transport != "http":
        raise ProbeError(
            "Endpoint discovery succeeded, but this probe only implements HTTP inference."
        )
    if endpoint.session_token_model and endpoint.session_token_model != model:
        raise ProbeError("Refusing inference: the session token is bound to a different model.")
    headers = httpx.Headers(endpoint.headers)
    if (
        endpoint.api_key
        and "authorization" not in headers
        and "x-api-key" not in headers
        and "api-key" not in headers
    ):
        if endpoint.provider == "anthropic":
            headers["x-api-key"] = endpoint.api_key
        elif endpoint.provider == "azure":
            headers["api-key"] = endpoint.api_key
        else:
            headers["authorization"] = f"Bearer {endpoint.api_key}"
    if endpoint.session_token:
        if not endpoint.session_token_header:
            raise ProbeError("Refusing inference: session credential has no header name.")
        headers[endpoint.session_token_header] = endpoint.session_token
    headers["content-type"] = "application/json"
    prompt = "Reply with exactly the word OK."
    body: dict[str, object] = {"model": model, "stream": False}
    if endpoint.provider == "anthropic":
        # Like the Anthropic SDK, append /v1/messages to its supplied base URL.
        route = "v1/messages"
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        body.update(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    elif endpoint.provider in {"openai", "azure"} and endpoint.wire_api == "responses":
        route = "responses"
        body.update(input=prompt, max_output_tokens=max_tokens, store=False)
    elif endpoint.provider in {"openai", "azure"} and endpoint.wire_api == "completions":
        route = "chat/completions"
        body.update(
            messages=[{"role": "user", "content": prompt}], max_completion_tokens=max_tokens
        )
    else:
        raise ProbeError(
            "Endpoint discovery succeeded, but its wire API is not supported by this probe."
        )
    return f"{endpoint.base_url.rstrip('/')}/{route}", dict(headers), body


async def infer(endpoint: Endpoint, model: str, max_tokens: int) -> None:
    url, headers, body = inference_request(endpoint, model, max_tokens)
    # One HTTP request; no redirects, automatic retries, or environment proxy routing.
    async with httpx.AsyncClient(timeout=60, follow_redirects=False, trust_env=False) as client:
        response = await client.post(url, headers=headers, json=body)
    print(json.dumps({"inference_http_status": response.status_code}))
    if not response.is_success:
        raise ProbeError(
            "Direct inference was rejected; response body withheld. No retry was attempted."
        )
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("error") is not None:
        raise ProbeError("Direct inference returned an unexpected response; body withheld.")
    expected = (
        "content"
        if endpoint.provider == "anthropic"
        else ("output" if endpoint.wire_api == "responses" else "choices")
    )
    if not isinstance(payload.get(expected), list):
        raise ProbeError("Direct inference returned no recognizable output list; body withheld.")
    # Do not echo arbitrary server content, which can include request diagnostics.
    print(json.dumps({"direct_inference": "accepted", "output_items": len(payload[expected])}))


async def probe(args: argparse.Namespace) -> None:
    # Optional SDK imports stay inside the executable path; no repository dependency change.
    from copilot.client import CopilotClient, RuntimeConnection
    from copilot.generated.rpc import (
        PermissionDecisionUserNotAvailable,
        SessionProviderGetEndpointRequest,
    )

    cli_path = shutil.which(args.cli)
    if cli_path is None:
        raise ProbeError("Copilot CLI not found. Install it, then run: copilot login --device-code")
    env, token, source = runtime_environment(args.use_env_token)
    print(json.dumps({"credential_source": source}))
    # No repository context, hooks, tools, skills, or prompts are sent to the agent runtime.
    with TemporaryDirectory(prefix="fast-agent-copilot-probe-") as workdir:
        client = CopilotClient(
            connection=RuntimeConnection.for_stdio(path=cli_path),
            working_directory=workdir,
            env=env,
            github_token=token,
            use_logged_in_user=token is None,
            log_level="error",
            # Keep normal runtime mode: SDK "empty" mode disables system keychain access.
        )
        try:
            await client.start()
            auth = await client.get_auth_status()
            print(json.dumps({"authenticated": auth.isAuthenticated}))
            if not auth.isAuthenticated:
                raise ProbeError("Sign in locally first: copilot login --device-code")
            if not args.model:
                models = await client.list_models()
                print(json.dumps({"models": [model.id for model in models]}))
                print(
                    "Choose a model and rerun with --model MODEL; add --infer only to test inference."
                )
                return
            session = await client.create_session(
                model=args.model,
                available_tools=[],
                on_permission_request=lambda _request, _invocation: (
                    PermissionDecisionUserNotAvailable()
                ),
                enable_config_discovery=False,
                enable_file_hooks=False,
                enable_host_git_operations=False,
                enable_session_store=False,
                enable_skills=False,
                skip_custom_instructions=True,
                mcp_servers={},
            )
            try:
                resolved = await session.rpc.provider.get_endpoint(
                    SessionProviderGetEndpointRequest(model_id=args.model)
                )
                session_token = resolved.session_token
                endpoint = Endpoint(
                    provider=resolved.type.value,
                    wire_api=(
                        "messages"
                        if resolved.type.value == "anthropic"
                        else resolved.wire_api.value
                        if resolved.wire_api
                        else "completions"
                    ),
                    transport=resolved.transport.value if resolved.transport else "http",
                    base_url=resolved.base_url,
                    headers=resolved.headers,
                    api_key=resolved.api_key,
                    session_token=session_token.token if session_token else None,
                    session_token_header=session_token.header if session_token else None,
                    session_token_model=session_token.model if session_token else None,
                )
                print(json.dumps({"endpoint": endpoint.summary()}))
                if args.infer:
                    await infer(endpoint, args.model, args.max_tokens)
                else:
                    print(
                        "Discovery only: no inference request sent. Use --infer to send one request."
                    )
            finally:
                await session.disconnect()
        finally:
            await client.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="copilot", help="Copilot CLI executable (default: PATH).")
    parser.add_argument("--model", help="Account model ID; omit to list models without inference.")
    parser.add_argument(
        "--infer", action="store_true", help="Send ONE direct model request (uses plan quota)."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Output token limit for --infer."
    )
    parser.add_argument(
        "--use-env-token",
        action="store_true",
        help="Opt in to a GitHub token from the environment instead of stored login.",
    )
    args = parser.parse_args()
    if args.infer and not args.model:
        parser.error("--infer requires --model")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    # Do not enable SDK/HTTP wire logs or print exception strings containing credentials.
    logging.disable(logging.CRITICAL)
    try:
        asyncio.run(probe(args))
    except ProbeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"Probe failed ({type(exc).__name__}); raw error withheld to protect credentials. "
            "Check CLI login/version and experimental endpoint API availability.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
