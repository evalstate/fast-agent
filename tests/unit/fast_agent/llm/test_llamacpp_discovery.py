from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from fast_agent.llm.llamacpp_discovery import (
    LlamaCppDiscoveredModel,
    LlamaCppModelListing,
    build_llamacpp_overlay_manifest,
    discover_llamacpp_models,
    interrogate_llamacpp_model,
    normalize_llamacpp_url,
)


def test_normalize_llamacpp_url_accepts_root_and_v1_urls() -> None:
    root = normalize_llamacpp_url("http://localhost:8080")
    assert root.server_url == "http://localhost:8080"
    assert root.request_base_url == "http://localhost:8080/v1"
    assert root.models_urls()[0] == "http://localhost:8080/v1/models"

    v1 = normalize_llamacpp_url("http://localhost:8080/v1")
    assert v1.server_url == "http://localhost:8080"
    assert v1.request_base_url == "http://localhost:8080/v1"
    assert v1.models_urls()[0] == "http://localhost:8080/v1/models"


def test_build_llamacpp_overlay_manifest_omits_sampling_defaults_by_default() -> None:
    manifest = build_llamacpp_overlay_manifest(
        overlay_name="qwen-local",
        discovered_model=LlamaCppDiscoveredModel(
            listing=LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=262144,
            ),
            props_url="http://localhost:8080/props?model=unsloth%2FQwen3.5-9B-GGUF",
            runtime_context_window=75264,
            max_output_tokens=2048,
            temperature=0.800000011920929,
            top_k=40,
            top_p=0.949999988079071,
            min_p=0.05000000074505806,
            tokenizes=("text/plain", "image/jpeg", "image/png", "image/webp"),
            model_alias="Qwen local",
        ),
        base_url="http://localhost:8080/v1",
        auth="none",
        api_key_env=None,
        secret_ref=None,
        current=True,
    )

    payload = manifest.model_dump(mode="json", exclude_none=True)
    assert payload["provider"] == "openresponses"
    assert payload["connection"]["base_url"] == "http://localhost:8080/v1"
    assert payload["defaults"]["max_tokens"] == 2048
    assert "temperature" not in payload["defaults"]
    assert "top_k" not in payload["defaults"]
    assert "top_p" not in payload["defaults"]
    assert "min_p" not in payload["defaults"]
    assert payload["metadata"]["context_window"] == 75264
    assert payload["metadata"]["max_output_tokens"] == 2048
    assert payload["metadata"]["tokenizes"] == [
        "text/plain",
        "image/jpeg",
        "image/png",
        "image/webp",
    ]
    assert payload["picker"]["label"] == "Qwen local"
    assert payload["picker"]["description"] == "Imported from llama.cpp"


def test_build_llamacpp_overlay_manifest_can_include_sampling_defaults() -> None:
    manifest = build_llamacpp_overlay_manifest(
        overlay_name="qwen-local",
        discovered_model=LlamaCppDiscoveredModel(
            listing=LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=262144,
            ),
            props_url="http://localhost:8080/props?model=unsloth%2FQwen3.5-9B-GGUF",
            runtime_context_window=75264,
            max_output_tokens=2048,
            temperature=0.800000011920929,
            top_k=40,
            top_p=0.949999988079071,
            min_p=0.05000000074505806,
            tokenizes=("text/plain", "image/jpeg", "image/png", "image/webp"),
            model_alias="Qwen local",
        ),
        base_url="http://localhost:8080/v1",
        auth="none",
        api_key_env=None,
        secret_ref=None,
        current=True,
        include_sampling_defaults=True,
    )

    payload = manifest.model_dump(mode="json", exclude_none=True)
    assert payload["defaults"]["temperature"] == 0.8
    assert payload["defaults"]["top_k"] == 40
    assert payload["defaults"]["top_p"] == 0.95
    assert payload["defaults"]["min_p"] == 0.05


@dataclass
class _RouterState:
    child_port: int
    loaded: bool = True
    request_paths: list[str] = field(default_factory=list)


@dataclass
class _RouterServer:
    server: ThreadingHTTPServer
    thread: threading.Thread
    state: _RouterState

    @property
    def base_url(self) -> str:
        host = str(self.server.server_address[0])
        port = int(self.server.server_address[1])
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _start_child_server() -> _RouterServer:
    state = _RouterState(child_port=0)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/v1/models", "/models"}:
                payload = {
                    "data": [
                        {
                            "id": "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
                            "owned_by": "llamacpp",
                            "meta": {"n_ctx_train": 262144},
                        }
                    ]
                }
                self._write_json(payload)
                return

            if self.path == "/props":
                payload = {
                    "default_generation_settings": {
                        "n_ctx": 77056,
                        "params": {"n_predict": 2048},
                    },
                    "model_alias": "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
                    "modalities": {"vision": True, "audio": False},
                }
                self._write_json(payload)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _write_json(self, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    state.child_port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return _RouterServer(server=server, thread=thread, state=state)


def _start_router_server(
    *,
    child_port: int,
    initially_loaded: bool,
) -> _RouterServer:
    state = _RouterState(child_port=child_port, loaded=initially_loaded)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            state.request_paths.append(self.path)

            if parsed.path in {"/v1/models", "/models"}:
                status = {
                    "value": "loaded" if state.loaded else "unloaded",
                    "args": [
                        "llama-server",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(state.child_port if state.loaded else 0),
                    ],
                }
                payload = {
                    "data": [
                        {
                            "id": "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
                            "owned_by": "llamacpp",
                            "status": status,
                        }
                    ]
                }
                self._write_json(payload)
                return

            if parsed.path == "/props":
                selected_model = parse_qs(parsed.query).get("model", [""])[0]
                if selected_model == "unsloth/Qwen3.5-9B-GGUF:Q4_K_M":
                    state.loaded = True
                    payload = {
                        "default_generation_settings": {
                            "n_ctx": 77056,
                            "params": {"n_predict": 2048},
                        },
                        "model_alias": "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
                        "modalities": {"vision": True, "audio": False},
                    }
                else:
                    payload = {
                        "role": "router",
                        "default_generation_settings": {"n_ctx": 0, "params": None},
                    }
                self._write_json(payload)
                return

            if parsed.path == "/slots":
                self._write_json([])
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _write_json(self, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return _RouterServer(server=server, thread=thread, state=state)


@pytest.mark.asyncio
async def test_discover_llamacpp_models_enriches_router_loaded_model_metadata() -> None:
    child = _start_child_server()
    router = _start_router_server(child_port=child.state.child_port, initially_loaded=True)

    try:
        catalog = await discover_llamacpp_models(url=router.base_url)
    finally:
        router.close()
        child.close()

    assert catalog.router_mode is True
    assert catalog.models[0].training_context_window == 262144


@pytest.mark.asyncio
async def test_interrogate_llamacpp_model_refreshes_router_metadata_after_autoload() -> None:
    child = _start_child_server()
    router = _start_router_server(child_port=child.state.child_port, initially_loaded=False)

    try:
        catalog = await discover_llamacpp_models(url=router.base_url)
        assert catalog.models[0].training_context_window is None

        discovered = await interrogate_llamacpp_model(
            catalog=catalog,
            model_id="unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
        )
    finally:
        router.close()
        child.close()

    assert discovered.listing.training_context_window == 262144
    assert discovered.runtime_context_window == 77056
