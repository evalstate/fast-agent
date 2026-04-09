"""
Testing notes:

- This module owns overlay loading/resolution behavior, overlay-scoped metadata,
  and overlay precedence over ambient presets.
- Prefer real overlay files and real factory/selection flows rather than
  rebuilding overlay-derived objects by hand.
- Avoid depending on incidental picker/provider ordering; assert against the
  overlay group by identity instead.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from fast_agent.config import Settings, update_global_settings

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.openai.openresponses import OpenResponsesLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker_common import build_snapshot


def _write_overlay(env_dir: Path, filename: str, content: str) -> None:
    overlays_dir = env_dir / "model-overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    (overlays_dir / filename).write_text(content, encoding="utf-8")


def _cleanup_overlay_runtime_state(base_dir: Path) -> None:
    empty_env_dir = base_dir / "empty-fast-agent"
    empty_env_dir.mkdir(parents=True, exist_ok=True)
    load_model_overlay_registry(start_path=base_dir, env_dir=empty_env_dir)


def _overlay_group(snapshot):
    return next(option for option in snapshot.providers if option.overlay_group)


def test_same_provider_overlays_create_distinct_openresponses_clients(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "qwen-local.yaml",
        """
name: qwen-local
provider: openresponses
model: overlay-tests/Qwen-Local
connection:
  base_url: http://localhost:8080/v1
  auth: none
defaults:
  temperature: 0.7
  max_tokens: 4096
metadata:
  context_window: 131072
  max_output_tokens: 4096
""".strip(),
    )
    _write_overlay(
        env_dir,
        "qwen-remote.yaml",
        """
name: qwen-remote
provider: openresponses
model: overlay-tests/Qwen-Local
connection:
  base_url: https://remote.example/v1
  auth: env
  api_key_env: REMOTE_QWEN_KEY
defaults:
  temperature: 0.5
""".strip(),
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    previous_remote_key = os.environ.get("REMOTE_QWEN_KEY")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)
    os.environ["REMOTE_QWEN_KEY"] = "remote-key"

    try:
        local_llm = ModelFactory.create_factory("qwen-local")(LlmAgent(AgentConfig(name="local")))
        remote_llm = ModelFactory.create_factory("qwen-remote")(
            LlmAgent(AgentConfig(name="remote"))
        )

        assert isinstance(local_llm, OpenResponsesLLM)
        assert isinstance(remote_llm, OpenResponsesLLM)
        assert local_llm._base_url() == "http://localhost:8080/v1"
        assert remote_llm._base_url() == "https://remote.example/v1"
        assert local_llm._api_key() == ""
        assert remote_llm._api_key() == "remote-key"
        assert local_llm.default_request_params.maxTokens == 4096
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

        if previous_remote_key is None:
            os.environ.pop("REMOTE_QWEN_KEY", None)
        else:
            os.environ["REMOTE_QWEN_KEY"] = previous_remote_key


def test_overlay_presets_resolve_overlay_metadata_and_picker_entries(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "picker-overlay.yaml",
        """
name: picker-local
provider: openresponses
model: overlay-tests/Qwen-Picker
connection:
  base_url: http://localhost:8081/v1
  auth: none
defaults:
  temperature: 0.65
  top_p: 0.95
picker:
  label: Picker local
  description: Local picker entry
  current: true
metadata:
  context_window: 65536
  max_output_tokens: 2048
  fast: true
""".strip(),
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)

    try:
        presets = ModelFactory.get_runtime_presets()
        assert presets["picker-local"] == (
            "openresponses.overlay-tests/Qwen-Picker?temperature=0.65&top_p=0.95"
        )

        parsed = ModelFactory.parse_model_string("picker-local")
        assert parsed.provider == Provider.OPENRESPONSES
        assert parsed.model_name == "overlay-tests/Qwen-Picker"
        assert parsed.temperature == 0.65
        assert parsed.top_p == 0.95

        resolved = ModelFactory.resolve_model_spec("picker-local")
        params = resolved.model_params
        assert params is not None
        assert resolved.source == "overlay"
        assert resolved.selected_model_name == "picker-local"
        assert resolved.wire_model_name == "overlay-tests/Qwen-Picker"
        assert params.context_window == 65536
        assert params.max_output_tokens == 2048
        assert params.fast is True

        assert ModelDatabase.get_model_params("overlay-tests/Qwen-Picker") is None

        assert "picker-local" in ModelSelectionCatalog.list_current_aliases(Provider.OPENRESPONSES)
        snapshot = build_snapshot(config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.option_display_name == "Overlays"
        picker_entry = next(
            entry for entry in overlay_group.curated_entries if entry.alias == "picker-local"
        )
        assert picker_entry.local is True
        assert picker_entry.description == "Local picker entry"
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_same_wire_model_overlays_keep_distinct_resolved_metadata(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "tiny-local.yaml",
        """
name: tiny-local
provider: openresponses
model: overlay-tests/Llama-Local
connection:
  base_url: http://localhost:8080/v1
  auth: none
defaults:
  max_tokens: 1024
metadata:
  context_window: 8192
  max_output_tokens: 1024
""".strip(),
    )
    _write_overlay(
        env_dir,
        "big-local.yaml",
        """
name: big-local
provider: openresponses
model: overlay-tests/Llama-Local
connection:
  base_url: http://localhost:8081/v1
  auth: none
defaults:
  max_tokens: 8192
metadata:
  context_window: 131072
  max_output_tokens: 8192
""".strip(),
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)

    try:
        tiny_resolved = ModelFactory.resolve_model_spec("tiny-local")
        big_resolved = ModelFactory.resolve_model_spec("big-local")

        assert tiny_resolved.source == "overlay"
        assert big_resolved.source == "overlay"
        assert tiny_resolved.wire_model_name == "overlay-tests/Llama-Local"
        assert big_resolved.wire_model_name == "overlay-tests/Llama-Local"
        assert tiny_resolved.context_window == 8192
        assert big_resolved.context_window == 131072
        assert tiny_resolved.max_output_tokens == 1024
        assert big_resolved.max_output_tokens == 8192
        assert ModelDatabase.get_model_params("overlay-tests/Llama-Local") is None

        tiny_llm = ModelFactory.create_factory("tiny-local")(LlmAgent(AgentConfig(name="tiny")))
        big_llm = ModelFactory.create_factory("big-local")(LlmAgent(AgentConfig(name="big")))

        assert tiny_llm.model_info is not None
        assert big_llm.model_info is not None
        assert tiny_llm.model_info.context_window == 8192
        assert big_llm.model_info.context_window == 131072
        assert tiny_llm.model_info.max_output_tokens == 1024
        assert big_llm.model_info.max_output_tokens == 8192
        assert tiny_llm.usage_accumulator is not None
        assert big_llm.usage_accumulator is not None
        assert tiny_llm.usage_accumulator.context_window_size == 8192
        assert big_llm.usage_accumulator.context_window_size == 131072
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_overlay_resolution_precedence_beats_custom_preset(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "picker-local.yaml",
        """
name: picker-local
provider: openresponses
model: overlay-tests/Qwen-Picker
connection:
  base_url: http://localhost:8081/v1
  auth: none
metadata:
  context_window: 65536
  max_output_tokens: 2048
""".strip(),
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)

    try:
        resolved = ModelFactory.resolve_model_spec(
            "picker-local",
            presets={"picker-local": "responses.gpt-5.2"},
        )

        assert resolved.source == "overlay"
        assert resolved.provider == Provider.OPENRESPONSES
        assert resolved.wire_model_name == "overlay-tests/Qwen-Picker"
        assert resolved.context_window == 65536
        assert resolved.max_output_tokens == 2048
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_overlay_known_model_metadata_applies_to_llm_model_info(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "haikutiny.yaml",
        """
name: haikutiny
provider: anthropic
model: claude-haiku-4-5
defaults:
  temperature: 0.5
  max_tokens: 1024
metadata:
  context_window: 8192
  max_output_tokens: 1024
""".strip(),
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)

    try:
        llm = ModelFactory.create_factory("haikutiny")(LlmAgent(AgentConfig(name="haikutiny")))

        assert llm.resolved_model is not None
        assert llm.resolved_model.selected_model_name == "haikutiny"
        assert llm.resolved_model.overlay_name == "haikutiny"
        assert llm.resolved_model.display_name == "haikutiny"
        assert llm.resolved_model.wire_model_name == "claude-haiku-4-5"
        assert llm.model_info is not None
        assert llm.model_info.context_window == 8192
        assert llm.model_info.max_output_tokens == 1024
        assert llm.usage_accumulator is not None
        assert llm.usage_accumulator.context_window_size == 8192
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_overlay_resolution_uses_config_relative_environment_dir_when_cwd_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    env_dir = project_dir / ".fast-agent"
    _write_overlay(
        env_dir,
        "picker-overlay.yaml",
        """
name: picker-local
provider: openresponses
model: overlay-tests/Qwen-Picker
connection:
  base_url: http://localhost:8081/v1
  auth: none
metadata:
  context_window: 65536
  max_output_tokens: 2048
""".strip(),
    )

    settings = Settings(environment_dir=".fast-agent")
    settings._config_file = str(project_dir / "fastagent.config.yaml")
    update_global_settings(settings)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)

    resolved = ModelFactory.resolve_model_spec("picker-local")

    assert resolved.source == "overlay"
    assert resolved.overlay_name == "picker-local"
    assert resolved.wire_model_name == "overlay-tests/Qwen-Picker"


@pytest.mark.asyncio
async def test_overlay_model_switch_reapplies_overlay_max_tokens_defaults(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "tiny-local.yaml",
        """
name: tiny-local
provider: openresponses
model: overlay-tests/Llama-Local
connection:
  base_url: http://localhost:8080/v1
  auth: none
defaults:
  max_tokens: 1024
metadata:
  context_window: 8192
  max_output_tokens: 1024
""".strip(),
    )
    _write_overlay(
        env_dir,
        "big-local.yaml",
        """
name: big-local
provider: openresponses
model: overlay-tests/Llama-Local
connection:
  base_url: http://localhost:8081/v1
  auth: none
defaults:
  max_tokens: 8192
metadata:
  context_window: 131072
  max_output_tokens: 8192
""".strip(),
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)

    try:
        agent = LlmAgent(AgentConfig(name="switcher"))
        await agent.attach_llm(ModelFactory.create_factory("big-local"))

        assert agent.llm is not None
        assert agent.llm.default_request_params.maxTokens == 8192

        await agent.set_model("tiny-local")

        assert agent.llm is not None
        assert agent.llm.default_request_params.maxTokens == 1024
        assert agent.llm.resolved_model is not None
        assert agent.llm.resolved_model.overlay_name == "tiny-local"
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_overlay_secret_ref_resolves_api_key_from_companion_file(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "secret-overlay.yaml",
        """
name: qwen-secret
provider: openresponses
model: overlay-tests/Qwen-Secret
connection:
  base_url: https://secret.example/v1
  auth: secret_ref
  secret_ref: remote-qwen
""".strip(),
    )
    (env_dir / "model-overlays.secrets.yaml").write_text(
        """
remote-qwen:
  api_key: secret-token
""".strip(),
        encoding="utf-8",
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)

    try:
        llm = ModelFactory.create_factory("qwen-secret")(LlmAgent(AgentConfig(name="secret")))
        assert isinstance(llm, OpenResponsesLLM)
        assert llm._base_url() == "https://secret.example/v1"
        assert llm._api_key() == "secret-token"
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_overlay_context_window_survives_missing_max_output_tokens(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "llamacpp-qwen.yaml",
        """
name: llamacpp-qwen
provider: openresponses
model: unsloth/Qwen3.5-9B-GGUF
connection:
  base_url: http://localhost:8080/v1
  auth: none
defaults:
  temperature: 0.8
metadata:
  context_window: 75264
  tokenizes:
    - text/plain
    - image/jpeg
    - image/png
    - image/webp
""".strip(),
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)

    try:
        resolved = ModelFactory.resolve_model_spec("llamacpp-qwen")
        assert resolved.context_window == 75264
        assert resolved.max_output_tokens is None

        llm = ModelFactory.create_factory("llamacpp-qwen")(LlmAgent(AgentConfig(name="local")))
        assert llm.model_info is not None
        assert llm.model_info.context_window == 75264
        assert llm.model_info.max_output_tokens is None
        assert llm.usage_accumulator is not None
        assert llm.usage_accumulator.context_window_size == 75264
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_overlay_legacy_metadata_default_temperature_is_still_used(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    _write_overlay(
        env_dir,
        "legacy-temp.yaml",
        """
name: legacy-temp
provider: openresponses
model: unsloth/Qwen3.5-9B-GGUF
connection:
  base_url: http://localhost:8080/v1
  auth: none
metadata:
  context_window: 75264
  max_output_tokens: 2048
  default_temperature: 0.7
""".strip(),
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)

    try:
        resolved = ModelFactory.resolve_model_spec("legacy-temp")
        assert resolved.model_params is not None
        assert resolved.model_params.default_temperature == 0.7
    finally:
        _cleanup_overlay_runtime_state(tmp_path)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
