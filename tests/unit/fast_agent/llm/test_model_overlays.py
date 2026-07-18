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
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from fast_agent.config import Settings, get_settings, update_global_settings
from fast_agent.constants import FAST_AGENT_RUNTIME_HOME

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import (
    build_model_overlay_manifest_from_database,
    load_model_overlay_registry,
)
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.openai.openresponses import OpenResponsesLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker_common import build_snapshot


def _write_overlay(home: Path, filename: str, content: str) -> None:
    overlays_dir = home / "model-overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    (overlays_dir / filename).write_text(content, encoding="utf-8")


def _cleanup_overlay_runtime_state(base_dir: Path) -> None:
    empty_home = base_dir / "empty-fast-agent"
    empty_home.mkdir(parents=True, exist_ok=True)
    load_model_overlay_registry(start_path=base_dir, home=empty_home)


@contextmanager
def _isolated_overlay_environment(
    home: Path | None,
    *,
    cleanup_base: Path,
) -> Iterator[None]:
    env_keys = ("FAST_AGENT_HOME", "FAST_AGENT_HOME", FAST_AGENT_RUNTIME_HOME)
    previous = {key: os.environ.get(key) for key in env_keys}

    os.environ.pop("FAST_AGENT_HOME", None)
    os.environ.pop(FAST_AGENT_RUNTIME_HOME, None)
    if home is None:
        os.environ.pop("FAST_AGENT_HOME", None)
    else:
        os.environ["FAST_AGENT_HOME"] = str(home)

    try:
        yield
    finally:
        _cleanup_overlay_runtime_state(cleanup_base)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _overlay_group(snapshot):
    return next(option for option in snapshot.providers if option.overlay_group)


def test_export_preserves_explicit_provider_for_namespaced_model() -> None:
    manifest = build_model_overlay_manifest_from_database("openrouter.moonshotai/kimi-k2")

    assert manifest.provider == Provider.OPENROUTER
    assert manifest.model == "moonshotai/kimi-k2"


def test_export_preserves_managed_process_poll_folding_policy() -> None:
    manifest = build_model_overlay_manifest_from_database("xai.grok-4.5")

    assert manifest.metadata.managed_process_poll_folding is True


def test_export_preserves_explicit_provider_over_catalog_default() -> None:
    manifest = build_model_overlay_manifest_from_database("openrouter.gpt-4o")

    assert manifest.provider == Provider.OPENROUTER
    assert manifest.model == "gpt-4o"


def test_export_preserves_hf_namespace_before_database_lookup() -> None:
    manifest = build_model_overlay_manifest_from_database(
        "HuggingFace.OpenAI/GPT-OSS-120B:CEREBRAS"
    )

    assert manifest.name == "hf-OpenAI-GPT-OSS-120B-CEREBRAS"
    assert manifest.provider == Provider.HUGGINGFACE
    assert manifest.model == "OpenAI/GPT-OSS-120B:CEREBRAS"


def test_export_accepts_slash_prefixed_provider_model() -> None:
    manifest = build_model_overlay_manifest_from_database("anthropic/claude-sonnet-4-5")

    assert manifest.provider == Provider.ANTHROPIC
    assert manifest.model == "claude-sonnet-4-5"


def test_export_default_name_strips_model_query_params() -> None:
    manifest = build_model_overlay_manifest_from_database(
        "hf.Qwen/Qwen3.5-397B-A17B:novita?"
        "temperature=0.6&top_p=0.95&top_k=20&min_p=0.0&presence_penalty=0.0"
        "&repetition_penalty=1.0&reasoning=on"
    )

    assert manifest.name == "hf-Qwen-Qwen3.5-397B-A17B-novita"
    assert manifest.model == (
        "Qwen/Qwen3.5-397B-A17B:novita?"
        "temperature=0.6&top_p=0.95&top_k=20&min_p=0.0&presence_penalty=0.0"
        "&repetition_penalty=1.0&reasoning=on"
    )


def test_export_preserves_bare_hf_namespace_that_matches_provider() -> None:
    manifest = build_model_overlay_manifest_from_database("openai/gpt-oss-120b")

    assert manifest.provider == Provider.HUGGINGFACE
    assert manifest.model == "openai/gpt-oss-120b"


def test_overlay_configures_process_poll_default_wait(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
        "poll-wait.yaml",
        """
name: poll-wait
provider: openresponses
model: overlay-tests/Poll-Wait
connection:
  base_url: http://localhost:8080/v1
  auth: none
metadata:
  context_window: 65536
  max_output_tokens: 4096
  process_poll_default_wait_seconds: 30
""".strip(),
    )

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        resolved = ModelFactory.resolve_model_spec("poll-wait")

    assert resolved.model_params is not None
    assert resolved.model_params.process_poll_default_wait_seconds == 30


def test_same_provider_overlays_create_distinct_openresponses_clients(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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
        home,
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

    previous_remote_key = os.environ.get("REMOTE_QWEN_KEY")
    os.environ["REMOTE_QWEN_KEY"] = "remote-key"

    try:
        with _isolated_overlay_environment(home, cleanup_base=tmp_path):
            local_llm = ModelFactory.create_factory("qwen-local")(
                LlmAgent(AgentConfig(name="local"))
            )
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
        if previous_remote_key is None:
            os.environ.pop("REMOTE_QWEN_KEY", None)
        else:
            os.environ["REMOTE_QWEN_KEY"] = previous_remote_key


def test_overlay_presets_resolve_overlay_metadata_and_picker_entries(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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
  json_mode: NONE
  structured_tool_policy: defer
  model_specific: Overlay-specific instructions.
  fast: true
""".strip(),
    )

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
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
        assert params.json_mode is None
        assert params.structured_tool_policy == "defer"
        assert params.model_specific == "Overlay-specific instructions."
        assert params.fast is True

        assert ModelDatabase.get_model_params("overlay-tests/Qwen-Picker") is None

        assert "picker-local" in ModelSelectionCatalog.list_current_aliases(Provider.OPENRESPONSES)
        snapshot = build_snapshot(config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.display_name == "Overlays"
        picker_entry = next(
            entry for entry in overlay_group.curated_entries if entry.alias == "picker-local"
        )
        assert picker_entry.local is True
        assert picker_entry.description == "Local picker entry"


def test_same_wire_model_overlays_keep_distinct_resolved_metadata(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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
        home,
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

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
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


def test_overlay_resolution_precedence_beats_custom_preset(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        resolved = ModelFactory.resolve_model_spec(
            "picker-local",
            presets={"picker-local": "responses.gpt-5.2"},
        )

        assert resolved.source == "overlay"
        assert resolved.provider == Provider.OPENRESPONSES
        assert resolved.wire_model_name == "overlay-tests/Qwen-Picker"
        assert resolved.context_window == 65536
        assert resolved.max_output_tokens == 2048


def test_overlay_registry_loads_uppercase_yaml_suffix(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
        "upper-suffix.YAML",
        """
name: upper-suffix
provider: openresponses
model: overlay-tests/Upper-Suffix
connection:
  auth: none
metadata:
  context_window: 8192
  max_output_tokens: 1024
""".strip(),
    )

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        registry = load_model_overlay_registry(start_path=tmp_path, home=home)

    assert "upper-suffix" in registry.by_name()


def test_new_overlay_model_defaults_to_schema_json_mode(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
        "schema-default.yaml",
        """
name: schema-default
provider: openresponses
model: overlay-tests/Schema-Default
connection:
  base_url: http://localhost:8081/v1
  auth: none
metadata:
  context_window: 65536
  max_output_tokens: 2048
""".strip(),
    )

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        resolved = ModelFactory.resolve_model_spec("schema-default")

        assert resolved.model_params is not None
        assert resolved.model_params.json_mode == "schema"


def test_overlay_known_model_metadata_applies_to_llm_model_info(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
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


def test_overlay_resolution_uses_config_relative_home_when_cwd_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    home = project_dir / ".fast-agent"
    _write_overlay(
        home,
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

    previous_settings = get_settings()
    settings = Settings(home=".fast-agent")
    settings._config_file = str(project_dir / "fastagent.config.yaml")
    update_global_settings(settings)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    with _isolated_overlay_environment(None, cleanup_base=tmp_path):
        try:
            resolved = ModelFactory.resolve_model_spec("picker-local")
        finally:
            update_global_settings(previous_settings)

    assert resolved.source == "overlay"
    assert resolved.overlay_name == "picker-local"
    assert resolved.wire_model_name == "overlay-tests/Qwen-Picker"


def test_overlay_resolution_uses_config_relative_default_home_when_cwd_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    home = project_dir / ".fast-agent"
    _write_overlay(
        home,
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

    previous_settings = get_settings()
    settings = Settings(home=None)
    settings._config_file = str(project_dir / "fastagent.config.yaml")
    update_global_settings(settings)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    with _isolated_overlay_environment(None, cleanup_base=tmp_path):
        try:
            resolved = ModelFactory.resolve_model_spec("picker-local")
        finally:
            update_global_settings(previous_settings)

    assert resolved.source == "overlay"
    assert resolved.overlay_name == "picker-local"
    assert resolved.wire_model_name == "overlay-tests/Qwen-Picker"


@pytest.mark.asyncio
async def test_overlay_model_switch_reapplies_overlay_max_tokens_defaults(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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
        home,
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

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        agent = LlmAgent(AgentConfig(name="switcher"))
        await agent.attach_llm(ModelFactory.create_factory("big-local"))

        assert agent.llm is not None
        assert agent.llm.default_request_params.maxTokens == 8192

        await agent.set_model("tiny-local")

        assert agent.llm is not None
        assert agent.llm.default_request_params.maxTokens == 1024
        assert agent.llm.resolved_model is not None
        assert agent.llm.resolved_model.overlay_name == "tiny-local"


def test_overlay_secret_ref_resolves_api_key_from_companion_file(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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
    (home / "model-overlays.secrets.yaml").write_text(
        """
remote-qwen:
  api_key: secret-token
""".strip(),
        encoding="utf-8",
    )

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        llm = ModelFactory.create_factory("qwen-secret")(LlmAgent(AgentConfig(name="secret")))
        assert isinstance(llm, OpenResponsesLLM)
        assert llm._base_url() == "https://secret.example/v1"
        assert llm._api_key() == "secret-token"


def test_overlay_context_window_survives_missing_max_output_tokens(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        resolved = ModelFactory.resolve_model_spec("llamacpp-qwen")
        assert resolved.context_window == 75264
        assert resolved.max_output_tokens is None

        llm = ModelFactory.create_factory("llamacpp-qwen")(LlmAgent(AgentConfig(name="local")))
        assert llm.model_info is not None
        assert llm.model_info.context_window == 75264
        assert llm.model_info.max_output_tokens is None
        assert llm.usage_accumulator is not None
        assert llm.usage_accumulator.context_window_size == 75264


def test_overlay_legacy_metadata_default_temperature_is_still_used(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
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

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        resolved = ModelFactory.resolve_model_spec("legacy-temp")
        assert resolved.model_params is not None
        assert resolved.model_params.default_temperature == 0.7


def test_overlay_numeric_fields_reject_yaml_booleans(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    _write_overlay(
        home,
        "valid-numeric-strings.yaml",
        """
name: valid-numeric-strings
provider: openresponses
model: overlay-tests/Valid-Numeric
connection:
  auth: none
defaults:
  temperature: "0.4"
  top_k: "20"
  max_tokens: "1024"
metadata:
  context_window: "8192"
  max_output_tokens: "1024"
""".strip(),
    )
    _write_overlay(
        home,
        "invalid-boolean-numbers.yaml",
        """
name: invalid-boolean-numbers
provider: openresponses
model: overlay-tests/Invalid-Boolean
connection:
  auth: none
defaults:
  temperature: true
metadata:
  context_window: true
  max_output_tokens: false
""".strip(),
    )

    with _isolated_overlay_environment(home, cleanup_base=tmp_path):
        registry = load_model_overlay_registry(start_path=tmp_path, home=home)
        by_name = registry.by_name()

    assert "valid-numeric-strings" in by_name
    assert "invalid-boolean-numbers" not in by_name
    assert by_name["valid-numeric-strings"].manifest.defaults.temperature == 0.4
    assert by_name["valid-numeric-strings"].manifest.defaults.top_k == 20
    assert by_name["valid-numeric-strings"].manifest.metadata.context_window == 8192
