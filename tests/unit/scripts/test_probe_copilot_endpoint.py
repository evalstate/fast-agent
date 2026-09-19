"""Offline contracts for the standalone Copilot endpoint probe (no SDK required)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_probe() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "probe_copilot_endpoint.py"
    spec = importlib.util.spec_from_file_location("probe_copilot_endpoint_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves annotations through this entry.
    spec.loader.exec_module(module)
    return module


probe = _load_probe()
ENDPOINT = probe.Endpoint(
    provider="openai",
    wire_api="completions",
    transport="http",
    base_url="https://inference.example.test/v1/",
    headers={},
)
TOKEN_NAMES = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


@pytest.mark.parametrize("selected", [None, *TOKEN_NAMES])
def test_credential_selection_is_explicit_and_child_env_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], selected: str | None
) -> None:
    for index, name in enumerate(TOKEN_NAMES):
        monkeypatch.delenv(name, raising=False)
        if selected is None or index >= TOKEN_NAMES.index(selected):
            monkeypatch.setenv(name, f"secret-{name}")
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "secret-byok")
    monkeypatch.setenv("COPILOT_API_URL", "https://override.example.test/secret")
    monkeypatch.setenv("COPILOT_ALLOW_GET_PROVIDER_ENDPOINT", "false")
    monkeypatch.setenv("PROBE_UNRELATED", "preserved")
    before = dict(os.environ)

    env, token, source = probe.runtime_environment(selected is not None)

    assert token == (f"secret-{selected}" if selected else None)
    assert source == (selected or "Copilot CLI stored login")
    assert not set((*TOKEN_NAMES, "GITHUB_COPILOT_API_TOKEN", "COPILOT_API_URL")) & env.keys()
    assert env["COPILOT_ALLOW_GET_PROVIDER_ENDPOINT"] == "true"
    assert env["PROBE_UNRELATED"] == "preserved"
    assert dict(os.environ) == before
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert "secret" not in source


def test_env_token_opt_in_requires_nonempty_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in TOKEN_NAMES:
        monkeypatch.setenv(name, "")
    with pytest.raises(probe.ProbeError, match="--use-env-token requires"):
        probe.runtime_environment(True)


@pytest.mark.parametrize("authorization", ["Authorization", "authorization", "AUTHORIZATION"])
def test_returned_authorization_and_rotating_session_token(authorization: str) -> None:
    endpoint = replace(
        ENDPOINT,
        headers={authorization: "Bearer returned-secret", "X-Session": "stale-secret"},
        api_key="fallback-secret",
        session_token="rotated-secret",
        session_token_header="x-session",
        session_token_model="test-model",
    )
    _, headers, _ = probe.inference_request(endpoint, "test-model", 17)
    normalized = httpx.Headers(headers)
    assert normalized["authorization"] == "Bearer returned-secret"
    assert normalized["x-session"] == "rotated-secret"
    assert "fallback-secret" not in headers.values()
    assert endpoint.headers["X-Session"] == "stale-secret"
    _, refreshed, _ = probe.inference_request(
        replace(endpoint, session_token="next-secret"), "test-model", 17
    )
    assert httpx.Headers(refreshed)["x-session"] == "next-secret"


@pytest.mark.parametrize(
    ("provider", "wire_api", "route", "limit_key", "auth_header"),
    [
        ("openai", "completions", "chat/completions", "max_completion_tokens", "authorization"),
        ("openai", "responses", "responses", "max_output_tokens", "authorization"),
        ("azure", "responses", "responses", "max_output_tokens", "api-key"),
        ("anthropic", "messages", "v1/messages", "max_tokens", "x-api-key"),
    ],
)
def test_request_routes_and_standard_invariants(
    provider: str, wire_api: str, route: str, limit_key: str, auth_header: str
) -> None:
    base_url = "https://inference.example.test/" if provider == "anthropic" else ENDPOINT.base_url
    endpoint = replace(
        ENDPOINT, provider=provider, wire_api=wire_api, api_key="api-secret", base_url=base_url
    )
    url, headers, body = probe.inference_request(endpoint, "test-model", 17)
    headers = httpx.Headers(headers)
    assert url == f"{base_url}{route}"
    assert headers["content-type"] == "application/json"
    assert headers[auth_header] == ("Bearer api-secret" if provider == "openai" else "api-secret")
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body[limit_key] == 17
    assert "tools" not in body
    if route == "responses":
        assert isinstance(body["input"], str) and body["input"]
        assert body["store"] is False
    else:
        messages = body["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"]
    if provider == "anthropic":
        assert headers["anthropic-version"]


def test_returned_anthropic_headers_are_not_overwritten() -> None:
    endpoint = replace(
        ENDPOINT,
        provider="anthropic",
        api_key="fallback-secret",
        headers={"X-API-Key": "returned-secret", "Anthropic-Version": "custom-version"},
    )
    _, headers, _ = probe.inference_request(endpoint, "test-model", 17)
    headers = httpx.Headers(headers)
    assert headers["x-api-key"] == "returned-secret"
    assert headers["anthropic-version"] == "custom-version"


def test_endpoint_repr_and_summary_do_not_disclose_credentials() -> None:
    endpoint = replace(
        ENDPOINT,
        base_url="https://user-secret:password-secret@inference.example.test/path-secret?q=query-secret#fragment-secret",
        headers={"Authorization": "header-secret"},
        api_key="api-secret",
        session_token="session-secret",
        session_token_header="X-Session",
    )
    summary = endpoint.summary()
    assert summary["host"] == "inference.example.test"
    assert summary["has_api_key"] is True
    assert summary["has_session_token"] is True
    assert summary["header_names"] == ["Authorization"]
    assert "secret" not in repr(endpoint) + json.dumps(summary)


@pytest.mark.parametrize(
    "url",
    [
        "http://inference.example.test/v1",
        "wss://inference.example.test/v1",
        "https:///v1",
        "https://user-secret@inference.example.test/v1",
        "https://user:password-secret@inference.example.test/v1",
        "https://inference.example.test/v1?token=query-secret",
        "https://inference.example.test/v1#fragment-secret",
    ],
)
def test_rejects_unsafe_base_urls(url: str) -> None:
    with pytest.raises(probe.ProbeError, match="HTTPS base URL") as exc:
        probe.inference_request(replace(ENDPOINT, base_url=url), "test-model", 17)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"transport": "websocket"},
        {"session_token_model": "other-model", "session_token": "session-secret"},
        {"session_token": "session-secret"},
        {"wire_api": "unsupported"},
        {"provider": "unsupported"},
    ],
)
def test_rejects_unsupported_or_unusable_endpoint(changes: dict[str, str]) -> None:
    with pytest.raises(probe.ProbeError) as exc:
        probe.inference_request(replace(ENDPOINT, **changes), "test-model", 17)
    assert "secret" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "wire_api", "status", "payload", "accepted"),
    [
        ("openai", "completions", 200, {"choices": ["server-secret"]}, True),
        ("openai", "responses", 200, {"output": [], "error": None}, True),
        ("anthropic", "completions", 200, {"content": ["server-secret"]}, True),
        ("openai", "completions", 302, {"detail": "server-secret"}, False),
        ("openai", "completions", 429, {"detail": "server-secret"}, False),
        ("openai", "completions", 503, {"detail": "server-secret"}, False),
        ("openai", "completions", 200, ["server-secret"], False),
        ("openai", "completions", 200, {"error": "server-secret", "choices": []}, False),
        ("openai", "completions", 200, {"choices": "server-secret"}, False),
        ("openai", "responses", 200, {"choices": []}, False),
        ("anthropic", "completions", 200, {"output": []}, False),
    ],
)
async def test_infer_single_offline_request_and_safe_response_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider: str,
    wire_api: str,
    status: int,
    payload: object,
    accepted: bool,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status, json=payload, headers={"Location": "https://redirect.example.test/secret"}
        )

    real_client = httpx.AsyncClient

    def client_factory(
        *, timeout: float, follow_redirects: bool, trust_env: bool
    ) -> httpx.AsyncClient:
        assert timeout > 0
        assert follow_redirects is False
        assert trust_env is False
        return real_client(
            timeout=timeout,
            follow_redirects=follow_redirects,
            trust_env=trust_env,
            transport=httpx.MockTransport(handle),
        )

    monkeypatch.setattr(probe.httpx, "AsyncClient", client_factory)
    endpoint = replace(ENDPOINT, provider=provider, wire_api=wire_api, api_key="request-secret")
    if accepted:
        await probe.infer(endpoint, "test-model", 17)
    else:
        with pytest.raises(probe.ProbeError) as exc:
            await probe.infer(endpoint, "test-model", 17)
        assert "secret" not in str(exc.value)
    assert len(requests) == 1  # Includes redirect and retryable HTTP failures.
    request = requests[0]
    assert request.method == "POST"
    expected_url, expected_headers, expected_body = probe.inference_request(
        endpoint, "test-model", 17
    )
    assert str(request.url) == expected_url
    assert all(request.headers[key] == value for key, value in expected_headers.items())
    assert json.loads(request.content) == expected_body
    captured = capsys.readouterr()
    assert "secret" not in captured.out + captured.err
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert lines[0] == {"inference_http_status": status}
    if accepted:
        assert lines[1] == {
            "direct_inference": "accepted",
            "output_items": 0 if wire_api == "responses" else 1,
        }
    else:
        assert len(lines) == 1
