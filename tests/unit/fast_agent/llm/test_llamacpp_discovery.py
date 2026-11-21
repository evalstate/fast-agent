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
    default_overlay_name_for_model,
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


def test_default_overlay_name_for_model_normalizes_model_leaf() -> None:
    assert default_overlay_name_for_model(" owner/ Qwen 3.5:Q4_K_M ") == (
        "llamacpp-qwen-3-5-q4-k-m"
    )


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
        def do_GET(self) -> None:
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
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    return _RouterServer(server=server, thread=thread, state=state)


def _start_router_server(
    *,
    child_port: int,
    initially_loaded: bool,
) -> _RouterServer:
    state = _RouterState(child_port=child_port, loaded=initially_loaded)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
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
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
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


def test_candidate_local_llamacpp_urls_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_agent.llm.llamacpp_discovery import (
        DEFAULT_LLAMA_CPP_URL,
        LEGACY_LLAMA_CPP_URL,
        candidate_local_llamacpp_urls,
    )

    monkeypatch.delenv("LLAMA_ARG_HOST", raising=False)
    monkeypatch.delenv("LLAMA_ARG_PORT", raising=False)

    assert DEFAULT_LLAMA_CPP_URL == "http://localhost:9931/v1"
    assert candidate_local_llamacpp_urls() == (
        "http://localhost:9931/v1",
        LEGACY_LLAMA_CPP_URL,
    )
    assert candidate_local_llamacpp_urls("http://127.0.0.1:7777") == ("http://127.0.0.1:7777/v1",)

    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    monkeypatch.setenv("LLAMA_ARG_PORT", "11435")
    assert candidate_local_llamacpp_urls() == (
        "http://localhost:11435/v1",
        "http://localhost:9931/v1",
        "http://localhost:8080/v1",
    )


def test_props_urls_supports_autoload_false() -> None:
    from fast_agent.llm.llamacpp_discovery import normalize_llamacpp_url

    endpoints = normalize_llamacpp_url("http://localhost:9931/v1")
    assert endpoints.props_urls(model_id="org/model:Q4_K_M", autoload=False) == (
        "http://localhost:9931/props?model=org%2Fmodel%3AQ4_K_M&autoload=false",
    )


@pytest.mark.parametrize(
    ("raw_name", "expected_key"),
    [
        ("Inkling-2-Mini-IQ4_NL", "thinkingmachines/inkling-2-mini"),
        ("ggml-org/Inkling-2-Mini-GGUF:IQ4_NL", "thinkingmachines/inkling-2-mini"),
        ("unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL", "google/gemma-4-26b-a4b-it"),
        ("unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL", "qwen/qwen3.6-35b-a3b"),
        ("Muse-Glimmer-30B-Q4_K_M.gguf", "meta-models/muse-glimmer-30b"),
        ("Qwen3.6-27B-Q8_0.gguf", "qwen/qwen3.6-27b"),
    ],
)
def test_match_llamacpp_model_database_entry(raw_name: str, expected_key: str) -> None:
    from fast_agent.llm.llamacpp_discovery import match_llamacpp_model_database_entry

    matched = match_llamacpp_model_database_entry(raw_name)
    assert matched is not None
    assert matched[0] == expected_key


def test_build_llamacpp_runtime_model_parameters_merges_db_and_live_server() -> None:
    from fast_agent.llm.llamacpp_discovery import (
        LlamaCppChatTemplateCaps,
        LlamaCppDiscoveredModel,
        LlamaCppModelListing,
        build_ephemeral_llamacpp_overlay,
        build_llamacpp_runtime_model_parameters,
    )
    from fast_agent.llm.model_overlays import (
        clear_ephemeral_model_overlays,
        load_model_overlay_registry,
        register_ephemeral_model_overlay,
    )

    discovered = LlamaCppDiscoveredModel(
        listing=LlamaCppModelListing(
            model_id="Inkling-2-Mini-IQ4_NL",
            owned_by="llamacpp",
            training_context_window=1048576,
            status="loaded",
        ),
        props_url="http://localhost:9931/props",
        runtime_context_window=32768,
        max_output_tokens=None,
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        min_p=0.05,
        tokenizes=("text/plain", "image/jpeg", "image/png", "image/webp"),
        model_alias="Inkling-2-Mini-IQ4_NL",
        chat_template_caps=LlamaCppChatTemplateCaps(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            supports_reasoning_effort=True,
            reasoning_effort_levels=("none", "low", "medium", "high"),
        ),
        matched_database_model="thinkingmachines/inkling-2-mini",
    )

    params = build_llamacpp_runtime_model_parameters(discovered)
    assert params.context_window == 32768
    assert params.json_mode == "schema"
    assert params.structured_tool_policy == "defer"
    assert params.reasoning == "reasoning_content"
    assert "image/png" in params.tokenizes
    assert params.reasoning_effort_spec is not None
    assert params.reasoning_effort_spec.allowed_efforts == ["none", "low", "medium", "high"]

    overlay = build_ephemeral_llamacpp_overlay(
        discovered_model=discovered,
        base_url="http://localhost:9931/v1",
    )
    try:
        register_ephemeral_model_overlay(overlay)
        registry = load_model_overlay_registry()
        loaded = registry.by_name().get(overlay.name)
        assert loaded is not None
        assert loaded.build_model_parameters() == params
    finally:
        clear_ephemeral_model_overlays()


@pytest.mark.asyncio
async def test_openresponses_inlines_local_file_image_as_data_uri(tmp_path) -> None:
    import base64

    from fast_agent.llm.provider.openai.openresponses import OpenResponsesLLM

    img_path = tmp_path / "sample.png"
    raw_png = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    img_path.write_bytes(raw_png)

    llm = OpenResponsesLLM.__new__(OpenResponsesLLM)
    part = {"type": "input_image", "image_url": img_path.as_uri()}
    from typing import Any, cast

    normalized, uploaded = await llm._normalize_input_image_part(cast("Any", None), part)

    assert uploaded is False
    expected_b64 = base64.b64encode(raw_png).decode("ascii")
    assert normalized["image_url"] == f"data:image/png;base64,{expected_b64}"
