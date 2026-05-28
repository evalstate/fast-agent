from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
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


def _start_llamacpp_server(
    *,
    null_props_params: bool = False,
    zero_runtime_context: bool = False,
) -> _LlamaCppServer:
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
                            "n_ctx": 0 if zero_runtime_context else 75264,
                            "params": (
                                None
                                if null_props_params
                                else {
                                    "temperature": 0.800000011920929,
                                    "top_k": 40,
                                    "top_p": 0.949999988079071,
                                    "min_p": 0.05000000074505806,
                                    "max_tokens": -1,
                                    "n_predict": -1,
                                }
                            ),
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
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
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
    assert "default_temperature" not in payload["metadata"]
    assert payload["picker"]["description"] == "Imported from llama.cpp"
    assert "Overlay token: qwen-local" in result.stdout
    assert "Context window: 75264 (runtime /props)" in result.stdout
    assert "copied the server's current sampling defaults" not in result.stdout


def test_model_llamacpp_import_tolerates_null_props_params_and_uses_slots_fallback(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    runner = CliRunner()
    server = _start_llamacpp_server(null_props_params=True, zero_runtime_context=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        result = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "unsloth/Qwen3.5-9B-GGUF",
                "--name",
                "qwen-local",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert result.exit_code == 0, result.stdout
    overlay_path = env_dir / "model-overlays" / "qwen-local.yaml"
    payload = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    assert payload["defaults"]["max_tokens"] == 2048
    assert "temperature" not in payload["defaults"]
    assert payload["metadata"]["context_window"] == 262144
    assert "Context window: 262144 (catalog fallback; /props reported none)" in result.stdout

    registry = load_model_overlay_registry(start_path=workspace, env_dir=env_dir)
    loaded = registry.resolve_model_string("qwen-local")
    assert loaded is not None
    assert loaded.manifest.connection.base_url == f"{server.base_url}/v1"


def test_resolve_llamacpp_picker_import_defaults_ignores_openresponses_settings(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    previous_base_url = os.environ.get("OPENRESPONSES_BASE_URL")
    previous_api_key = os.environ.get("OPENRESPONSES_API_KEY")
    try:
        os.environ["OPENRESPONSES_BASE_URL"] = "http://example.invalid:9999/v1"
        os.environ["OPENRESPONSES_API_KEY"] = "test-token"
        monkeypatch.setattr(
            model_command,
            "_load_cli_settings",
            lambda **kwargs: SimpleNamespace(
                openresponses=SimpleNamespace(base_url="http://ignored.example/v1")
            ),
        )

        defaults = model_command._resolve_llamacpp_picker_import_defaults(
            start_path=workspace,
            env_dir=None,
        )
    finally:
        if previous_base_url is None:
            os.environ.pop("OPENRESPONSES_BASE_URL", None)
        else:
            os.environ["OPENRESPONSES_BASE_URL"] = previous_base_url
        if previous_api_key is None:
            os.environ.pop("OPENRESPONSES_API_KEY", None)
        else:
            os.environ["OPENRESPONSES_API_KEY"] = previous_api_key

    assert defaults.url == model_command.DEFAULT_LLAMA_CPP_URL
    assert defaults.auth == "none"
    assert defaults.interrogation_api_key is None


def test_model_llamacpp_group_options_apply_before_subcommand(tmp_path: Path) -> None:
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
                server.base_url,
                "--env",
                str(env_dir),
                "import",
                "unsloth/Qwen3.5-9B-GGUF",
                "--name",
                "group-first",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert result.exit_code == 0, result.stdout
    assert server.state.request_paths[:3] == ["/v1/models", "/props", "/slots"]

    overlay_path = env_dir / "model-overlays" / "group-first.yaml"
    assert overlay_path.exists()

    payload = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    assert payload["connection"]["base_url"] == f"{server.base_url}/v1"


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
                "preview",
                "--url",
                f"{server.base_url}/v1",
                "--env",
                str(env_dir),
                "meta-llama/Llama-3.2-3B-Instruct",
                "--name",
                "llama-local",
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
    assert "default_temperature:" not in result.stdout
    assert "temperature:" not in result.stdout


def test_model_llamacpp_import_can_include_sampling_defaults(tmp_path: Path) -> None:
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
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "--include-sampling-defaults",
                "unsloth/Qwen3.5-9B-GGUF",
                "--name",
                "qwen-sampling",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert result.exit_code == 0, result.stdout
    payload = yaml.safe_load(
        (env_dir / "model-overlays" / "qwen-sampling.yaml").read_text(encoding="utf-8")
    )
    assert payload["defaults"]["temperature"] == 0.8
    assert payload["defaults"]["top_k"] == 40
    assert payload["defaults"]["top_p"] == 0.95
    assert payload["defaults"]["min_p"] == 0.05
    assert "copied the server's current sampling defaults" in result.stdout


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
                "list",
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


def test_model_llamacpp_import_json_start_now_still_launches(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    runner = CliRunner()
    server = _start_llamacpp_server()
    launched: dict[str, object] = {}

    def _fake_launch(
        *,
        overlay_name: str,
        env_dir: Path | None,
        with_shell: bool = False,
        smart: bool = False,
        announce: bool = True,
        execvpe_fn=...,
    ) -> None:
        launched["overlay_name"] = overlay_name
        launched["env_dir"] = env_dir
        launched["with_shell"] = with_shell
        launched["smart"] = smart
        launched["announce"] = announce

    monkeypatch.setattr(model_command, "_launch_llamacpp_overlay_now", _fake_launch)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        result = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "--json",
                "--start-now",
                "unsloth/Qwen3.5-9B-GGUF",
                "--name",
                "json-start-now",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["overlay_name"] == "json-start-now"
    assert launched == {
        "overlay_name": "json-start-now",
        "env_dir": env_dir,
        "with_shell": False,
        "smart": False,
        "announce": False,
    }


def test_model_llamacpp_help_clarifies_picker_and_overlay_flags() -> None:
    runner = CliRunner()

    result = runner.invoke(
        model_command.app,
        [
            "llamacpp",
            "--help",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Discover llama.cpp models, preview overlays" in result.stdout
    assert "list" in result.stdout
    assert "preview" in result.stdout
    assert "import" in result.stdout


def test_model_llamacpp_list_command_has_human_readable_output(tmp_path: Path) -> None:
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
                "list",
                "--url",
                server.base_url,
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert result.exit_code == 0, result.stdout
    assert "Discovered 2 llama.cpp model(s):" in result.stdout
    assert "unsloth/Qwen3.5-9B-GGUF (ctx 262144)" in result.stdout


def test_build_llamacpp_start_now_argv_includes_env_override(tmp_path: Path) -> None:
    env_dir = tmp_path / ".custom-env"

    argv = model_command._build_llamacpp_start_now_argv(
        overlay_name="llamacpp-qwen",
        env_dir=env_dir,
        with_shell=False,
        smart=False,
    )

    assert argv == [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "go",
        "--model",
        "llamacpp-qwen",
        "--env",
        str(env_dir),
    ]


def test_launch_llamacpp_overlay_now_execs_go_with_current_python(tmp_path: Path) -> None:
    env_dir = tmp_path / ".custom-env"
    captured: dict[str, object] = {}

    def _fake_execvpe(executable: str, argv: list[str], env: dict[str, str]) -> None:
        captured["executable"] = executable
        captured["argv"] = list(argv)
        captured["env_dir"] = env.get("ENVIRONMENT_DIR")

    model_command._launch_llamacpp_overlay_now(
        overlay_name="llamacpp-qwen",
        env_dir=env_dir,
        execvpe_fn=_fake_execvpe,
    )

    assert captured["executable"] == sys.executable
    assert captured["argv"] == [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "go",
        "--model",
        "llamacpp-qwen",
        "--env",
        str(env_dir),
    ]


def test_build_llamacpp_start_now_argv_with_shell_forces_x(tmp_path: Path) -> None:
    env_dir = tmp_path / ".custom-env"

    argv = model_command._build_llamacpp_start_now_argv(
        overlay_name="llamacpp-qwen",
        env_dir=env_dir,
        with_shell=True,
        smart=False,
    )

    assert argv == [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "go",
        "--model",
        "llamacpp-qwen",
        "-x",
        "--env",
        str(env_dir),
    ]


def test_launch_llamacpp_overlay_now_with_shell_execs_go_x(tmp_path: Path) -> None:
    env_dir = tmp_path / ".custom-env"
    captured: dict[str, object] = {}

    def _fake_execvpe(executable: str, argv: list[str], env: dict[str, str]) -> None:
        captured["executable"] = executable
        captured["argv"] = list(argv)
        captured["env_dir"] = env.get("ENVIRONMENT_DIR")

    model_command._launch_llamacpp_overlay_now(
        overlay_name="llamacpp-qwen",
        env_dir=env_dir,
        with_shell=True,
        execvpe_fn=_fake_execvpe,
    )

    assert captured["executable"] == sys.executable
    assert captured["argv"] == [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "go",
        "--model",
        "llamacpp-qwen",
        "-x",
        "--env",
        str(env_dir),
    ]


def test_build_llamacpp_start_now_argv_smart_uses_smart_and_shell(tmp_path: Path) -> None:
    env_dir = tmp_path / ".custom-env"

    argv = model_command._build_llamacpp_start_now_argv(
        overlay_name="llamacpp-qwen",
        env_dir=env_dir,
        with_shell=True,
        smart=True,
    )

    assert argv == [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "go",
        "--model",
        "llamacpp-qwen",
        "--smart",
        "-x",
        "--env",
        str(env_dir),
    ]


def test_model_llamacpp_import_start_now_smart_launches_smart_shell(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    runner = CliRunner()
    server = _start_llamacpp_server()
    launched: dict[str, object] = {}

    def _fake_launch(
        *,
        overlay_name: str,
        env_dir: Path | None,
        with_shell: bool = False,
        smart: bool = False,
        announce: bool = True,
        execvpe_fn=...,
    ) -> None:
        launched["overlay_name"] = overlay_name
        launched["env_dir"] = env_dir
        launched["with_shell"] = with_shell
        launched["smart"] = smart
        launched["announce"] = announce

    monkeypatch.setattr(model_command, "_launch_llamacpp_overlay_now", _fake_launch)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        result = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "--start-now",
                "--smart",
                "unsloth/Qwen3.5-9B-GGUF",
                "--name",
                "smart-start-now",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert result.exit_code == 0, result.stdout
    assert launched == {
        "overlay_name": "smart-start-now",
        "env_dir": env_dir,
        "with_shell": True,
        "smart": True,
        "announce": True,
    }


def test_model_llamacpp_reuses_existing_generated_overlay_for_unnamed_repeat_import(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    runner = CliRunner()
    server = _start_llamacpp_server()

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        first = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "unsloth/Qwen3.5-9B-GGUF",
            ],
        )
        second = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "unsloth/Qwen3.5-9B-GGUF",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    overlays_dir = env_dir / "model-overlays"
    overlay_files = sorted(path.name for path in overlays_dir.glob("*.yaml"))
    assert overlay_files == ["llamacpp-qwen3-5-9b-gguf.yaml"]
    assert "Overlay token: llamacpp-qwen3-5-9b-gguf" in second.stdout


def test_model_llamacpp_unnamed_import_does_not_reuse_named_overlay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".model-env"
    workspace.mkdir(parents=True)
    runner = CliRunner()
    server = _start_llamacpp_server()

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        named = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "unsloth/Qwen3.5-9B-GGUF",
                "--name",
                "qwen-dev",
            ],
        )
        unnamed = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "unsloth/Qwen3.5-9B-GGUF",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        server.close()

    assert named.exit_code == 0, named.stdout
    assert unnamed.exit_code == 0, unnamed.stdout
    overlays_dir = env_dir / "model-overlays"
    overlay_files = sorted(path.name for path in overlays_dir.glob("*.yaml"))
    assert overlay_files == ["llamacpp-qwen3-5-9b-gguf.yaml", "qwen-dev.yaml"]
    assert "Overlay token: llamacpp-qwen3-5-9b-gguf" in unnamed.stdout


def test_model_llamacpp_reused_generated_overlay_preserves_existing_auth(tmp_path: Path) -> None:
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
        first = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "unsloth/Qwen3.5-9B-GGUF",
                "--auth",
                "env",
                "--api-key-env",
                "LLAMA_CPP_TOKEN",
            ],
        )
        second = runner.invoke(
            model_command.app,
            [
                "llamacpp",
                "import",
                "--url",
                server.base_url,
                "--env",
                str(env_dir),
                "unsloth/Qwen3.5-9B-GGUF",
            ],
        )
    finally:
        os.chdir(previous_cwd)
        if previous_token is None:
            os.environ.pop("LLAMA_CPP_TOKEN", None)
        else:
            os.environ["LLAMA_CPP_TOKEN"] = previous_token
        server.close()

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    overlay_path = env_dir / "model-overlays" / "llamacpp-qwen3-5-9b-gguf.yaml"
    payload = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    assert payload["connection"]["auth"] == "env"
    assert payload["connection"]["api_key_env"] == "LLAMA_CPP_TOKEN"
