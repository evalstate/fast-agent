from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_MAX_RESULTS = 20
DEFAULT_TIMEOUT_SEC = 30


def _load_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token_value = token_path.read_text(encoding="utf-8").strip()
        return token_value or None

    return None


def _max_results_from_env() -> int:
    raw = os.getenv("HF_MAX_RESULTS")
    if not raw:
        return DEFAULT_MAX_RESULTS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_RESULTS
    return value if value > 0 else DEFAULT_MAX_RESULTS


def _normalize_endpoint(endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        raise ValueError("Endpoint must be a path relative to /api, not a full URL.")
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("Endpoint must be a non-empty string.")
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return endpoint


def _normalize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            normalized[key] = [str(item) for item in value]
        else:
            normalized[key] = str(value)
    return normalized


def _build_url(endpoint: str, params: dict[str, Any] | None) -> str:
    base = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{base}/api{_normalize_endpoint(endpoint)}"
    normalized_params = _normalize_params(params)
    if normalized_params:
        url = f"{url}?{urlencode(normalized_params, doseq=True)}"
    return url


def hf_api_request(
    endpoint: str,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    """
    Call the Hugging Face Hub API (GET/POST only).

    Args:
        endpoint: API endpoint relative to /api (e.g. "/whoami-v2").
        method: HTTP method (GET or POST).
        params: Optional query parameters.
        json_body: Optional JSON payload for POST requests.
        max_results: Max results when response is a list (defaults to HF_MAX_RESULTS).

    Returns:
        A dict with the response data and request metadata.
    """
    method_upper = method.upper()
    if method_upper not in {"GET", "POST"}:
        raise ValueError("Only GET and POST are allowed for hf_api_request.")

    if method_upper == "GET" and json_body is not None:
        raise ValueError("GET requests do not accept json_body.")

    url = _build_url(endpoint, params)

    headers = {
        "Accept": "application/json",
    }
    token = _load_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = None
    if method_upper == "POST":
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_body or {}).encode("utf-8")

    request = Request(url, headers=headers, data=data, method=method_upper)

    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SEC) as response:
            raw = response.read()
            status_code = response.status
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HF API error {exc.code} for {url}: {error_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"HF API request failed for {url}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw.decode("utf-8", errors="replace")

    if isinstance(payload, list):
        limit = max_results if max_results is not None else _max_results_from_env()
        payload = payload[: max(limit, 0)]

    return {
        "url": url,
        "status": status_code,
        "data": payload,
    }
