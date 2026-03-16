from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import yaml
from typer.testing import CliRunner

from fast_agent.cli.commands import model as model_command
from fast_agent.llm.model_overlays import load_model_overlay_registry


@dataclass
class _ServerState:
    request_paths: list[str] = field(default_factory=list)
    auth_headers: list[str | None] = field(default_factory=list)


@dataclass
class _LlamaCppServer:
    server: ThreadingHTTPServer
    state: _ServerState
    thread: threading.Thread

    @property
    def base_url(self) -> str:
        host = str(self.server.server_address[0])
        port = int(self.server.server_address[1])
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _start_llamacpp_server() -> _LlamaCppServer:
    state = _ServerState()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            state.request_paths.append(parsed.path)
            state.auth_headers.append(self.headers.get("Authorization"))

            if parsed.path == "/v1/models":
                payload = {
                    "data": [
                        {
                            "id": "unsloth/Qwen3.5-9B-GGUF",
                            "owned_by": "llamacpp",
                            "meta": {"n_ctx_train": 262144},
                        },
                        {
                            "id": "meta-llama/Llama-3.2-3B-Instruct",
                            "owned_by": "llamacpp",
                            "meta": {"n_ctx_train": 131072},
                        },
                    ]
                }
                self._write_json(payload)
                return

            if parsed.path == "/slots":
                payload = [
                    {"id": 0, "n_ctx": 75264, "speculative": False, "is_processing": False},
                    {
                        "id": 1,
                        "n_ctx": 75264,
                        "speculative": False,
                        "is_processing": True,
                        "params": {
                            "temperature": 0.8,
                            "top_k": 40,
                            "top_p": 0.95,
                            "min_p": 0.05,
                            "max_tokens": 2048,
                            "n_predict": 2048,
                        },
                    },
                ]
                self._write_json(payload)
                return

            if parsed.path == "/props":
                selected_model = parse_qs(parsed.query).get("model", [""])[0]
                if selected_model == "meta-llama/Llama-3.2-3B-Instruct":
                    payload = {
                        "default_generation_settings": {
                            "n_ctx": 32768,
                            "params": {
                                "temperature": 0.7,
                                "top_k": 30,
                                "top_p": 0.9,
                                "min_p": 0.02,
                                "n_predict": 1024,
                            },
                        },
                        "model_alias": "Llama local",
                        "modalities": {"vision": False, "audio": False},
                    }
                else:
                    payload = {
                        "default_generation_settings": {
                            "n_ctx": 75264,
                            "params": {
                                "temperature": 0.8,
                                "top_k": 40,
                                "top_p": 0.95,
                                "min_p": 0.05,
                                "max_tokens": -1,
                                "n_predict": -1,
                            },
                        },
                        "model_alias": "Qwen local",
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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return _LlamaCppServer(server=server, state=state, thread=thread)


def test_model_llamacpp_command_imports_overlay_from_models_endpoint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    runner = CliRunner()
    server = _start_llamacpp_server()

    previous_cwd = Path.cwd()
    previous_token = os.environ.get("LLAMA_CPP_TOKEN")
    os.environ["LLAMA_CPP_TOKEN"] = "test-token"
    try:
        os.chdir(workspace)
        result = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "--model",
                "unsloth/Qwen3.5-9B-GGUF",
                "--name",
                "qwen-local",
                "--auth",
                "env",
                "--api-key-env",
                "LLAMA_CPP_TOKEN",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        if previous_token is None:
            os.environ.pop("LLAMA_CPP_TOKEN", None)
        else:
            os.environ["LLAMA_CPP_TOKEN"] = previous_token
        server.close()

    assert result.exit_code == 0, result.stdout
    assert server.state.request_paths[:3] == ["/v1/models", "/props", "/slots"]
    assert server.state.auth_headers[:3] == [
        "Bearer test-token",
        "Bearer test-token",
        "Bearer test-token",
    ]

    overlay_path = env_dir / "model-overlays" / "qwen-local.yaml"
    assert overlay_path.exists()

    payload = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "openresponses"
    assert payload["model"] == "unsloth/Qwen3.5-9B-GGUF"
    assert payload["connection"]["base_url"] == f"{server.base_url}/v1"
    assert payload["connection"]["auth"] == "env"
    assert payload["connection"]["api_key_env"] == "LLAMA_CPP_TOKEN"
    assert payload["defaults"]["temperature"] == 0.8
    assert payload["defaults"]["top_k"] == 40
    assert payload["defaults"]["top_p"] == 0.95
    assert payload["defaults"]["min_p"] == 0.05
    assert payload["defaults"]["max_tokens"] == 2048
    assert payload["metadata"]["context_window"] == 75264
    assert payload["metadata"]["max_output_tokens"] == 2048
    assert payload["metadata"]["tokenizes"] == [
        "text/plain",
        "image/jpeg",
        "image/png",
        "image/webp",
    ]
    assert payload["picker"]["description"] == "Imported from llama.cpp"
    assert "Overlay token: qwen-local" in result.stdout

    registry = load_model_overlay_registry(start_path=workspace, env_dir=env_dir)
    loaded = registry.resolve_model_string("qwen-local")
    assert loaded is not None
    assert loaded.manifest.connection.base_url == f"{server.base_url}/v1"


def test_model_llamacpp_command_generate_overlay_dry_run_prints_yaml(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    runner = CliRunner()
    server = _start_llamacpp_server()

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        result = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "--url",
                f"{server.base_url}/v1",
                "--env",
                str(env_dir),
                "--model",
                "meta-llama/Llama-3.2-3B-Instruct",
                "--name",
                "llama-local",
                "--dry-run",
                "--generate-overlay",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert result.exit_code == 0, result.stdout
    assert not (env_dir / "model-overlays" / "llama-local.yaml").exists()
    assert "Dry run only; no overlay files were written." in result.stdout
    assert "name: llama-local" in result.stdout
    assert "provider: openresponses" in result.stdout
    assert "model: meta-llama/Llama-3.2-3B-Instruct" in result.stdout


def test_model_llamacpp_command_json_lists_discovered_models(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    runner = CliRunner()
    server = _start_llamacpp_server()

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        result = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "--url",
                server.base_url,
                "--json",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["request_base_url"] == f"{server.base_url}/v1"
    assert payload["models_url"] == f"{server.base_url}/v1/models"
    assert payload["models"] == [
        {
            "id": "unsloth/Qwen3.5-9B-GGUF",
            "owned_by": "llamacpp",
            "training_context_window": 262144,
        },
        {
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "owned_by": "llamacpp",
            "training_context_window": 131072,
        },
    ]
