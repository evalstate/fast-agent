"""
Testing notes:

- This module owns picker snapshot/model-option contracts and provider-group
  availability behavior.
- Prefer asserting the presence and properties of provider groups/options rather
  than depending on incidental list ordering.
- Curated catalog membership rules belong in llm/test_model_selection_catalog.py;
  low-level display rendering belongs in ui/test_model_picker.py.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker_common import (
    ANTHROPIC_VERTEX_PROVIDER_KEY,
    GENERIC_CUSTOM_MODEL_SENTINEL,
    LLAMACPP_IMPORT_SENTINEL,
    LLAMACPP_PROVIDER_KEY,
    ModelOption,
    ModelPickerSnapshot,
    build_snapshot,
    infer_initial_picker_provider,
    model_capabilities,
    model_options_for_option,
    model_options_for_provider,
)

if TYPE_CHECKING:
    from pathlib import Path


def _overlay_group(snapshot: ModelPickerSnapshot):
    return next(option for option in snapshot.providers if option.overlay_group)


def test_generic_provider_uses_custom_local_model_option() -> None:
    snapshot = build_snapshot()

    options = model_options_for_provider(snapshot, Provider.GENERIC, source="curated")

    assert options == [
        ModelOption(
            spec=GENERIC_CUSTOM_MODEL_SENTINEL,
            label="Enter local model string (e.g. llama3.2)",
        )
    ]


def test_openresponses_models_do_not_report_web_search_support() -> None:
    capabilities = model_capabilities("openresponses.gpt-5-mini")

    assert capabilities.provider == Provider.OPENRESPONSES
    assert capabilities.web_search_supported is False


def test_46_models_do_not_report_optional_long_context() -> None:
    capabilities = model_capabilities("claude-opus-4-6?context=1m")

    assert capabilities.provider == Provider.ANTHROPIC
    assert capabilities.supports_long_context is False
    assert capabilities.current_long_context is False
    assert capabilities.long_context_window is None


def test_infer_initial_picker_provider_uses_vertex_group_for_anthropic_vertex() -> None:
    assert infer_initial_picker_provider("anthropic-vertex.claude-sonnet-4-6") == ANTHROPIC_VERTEX_PROVIDER_KEY


def test_build_snapshot_surfaces_overlays_as_a_separate_group(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    overlays_dir = env_dir / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "haikutiny.yaml").write_text(
        "\n".join(
            [
                "name: haikutiny",
                "provider: anthropic",
                "model: claude-haiku-4-5",
                "defaults:",
                "  temperature: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)
    try:
        snapshot = build_snapshot(config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.option_display_name == "Overlays"
        assert overlay_group.overlay_group is True
        assert all(option.option_key != "openresponses" for option in snapshot.providers)
        assert any(option.option_key == "openrouter" for option in snapshot.providers)
        assert any(option.option_key == "azure" for option in snapshot.providers)
        assert any(option.option_key == "bedrock" for option in snapshot.providers)
    finally:
        empty_env_dir = tmp_path / ".empty-fast-agent"
        empty_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=empty_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_build_snapshot_shows_empty_overlay_group_even_without_overlays(tmp_path: Path) -> None:
    empty_env_dir = tmp_path / ".fast-agent"
    empty_env_dir.mkdir(parents=True, exist_ok=True)
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(empty_env_dir)
    try:
        snapshot = build_snapshot(config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.overlay_group is True
        assert overlay_group.curated_entries == ()
        assert any(option.option_key == "fast-agent" for option in snapshot.providers)
    finally:
        reset_env_dir = tmp_path / ".empty-fast-agent-reset"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_build_snapshot_includes_llamacpp_import_flow(tmp_path: Path) -> None:
    empty_env_dir = tmp_path / ".fast-agent"
    empty_env_dir.mkdir(parents=True, exist_ok=True)
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(empty_env_dir)
    try:
        snapshot = build_snapshot(config_payload={})
    finally:
        reset_env_dir = tmp_path / ".empty-fast-agent-llamacpp"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    option = next(provider for provider in snapshot.providers if provider.option_key == LLAMACPP_PROVIDER_KEY)

    assert option.option_display_name == "llama.cpp"
    assert option.active is False
    assert model_options_for_option(snapshot, option, source="curated") == [
        ModelOption(
            spec=LLAMACPP_IMPORT_SENTINEL,
            label="Discover local llama.cpp models and write overlay",
        )
    ]

    option_keys = [provider.option_key for provider in snapshot.providers]
    assert option_keys.index(LLAMACPP_PROVIDER_KEY) == option_keys.index(Provider.GENERIC.config_name) + 1


def test_build_snapshot_loads_overlays_relative_to_config_path(tmp_path: Path) -> None:
    from pathlib import Path

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    env_dir = workspace / ".fast-agent"
    overlays_dir = env_dir / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "haikutiny.yaml").write_text(
        "\n".join(
            [
                "name: haikutiny",
                "provider: anthropic",
                "model: claude-haiku-4-5",
            ]
        ),
        encoding="utf-8",
    )
    config_path = workspace / "fastagent.config.yaml"
    config_path.write_text("default_model: haiku\n", encoding="utf-8")

    cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    nested_cwd = workspace / "nested" / "deeper"
    nested_cwd.mkdir(parents=True)
    try:
        os.chdir(nested_cwd)
        snapshot = build_snapshot(config_path=config_path, config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert any(entry.alias == "haikutiny" for entry in overlay_group.curated_entries)
    finally:
        os.chdir(cwd)
        reset_env_dir = tmp_path / ".empty-fast-agent-config-path"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_build_snapshot_loads_overlays_relative_to_explicit_start_path(tmp_path: Path) -> None:
    from pathlib import Path

    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    overlays_dir = project_root / ".fast-agent" / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "haikutiny.yaml").write_text(
        "\n".join(
            [
                "name: haikutiny",
                "provider: anthropic",
                "model: claude-haiku-4-5",
            ]
        ),
        encoding="utf-8",
    )

    outside_cwd = tmp_path / "elsewhere"
    outside_cwd.mkdir(parents=True)

    cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(outside_cwd)
        snapshot = build_snapshot(
            config_payload={"environment_dir": ".fast-agent"},
            start_path=project_root,
        )
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert any(entry.alias == "haikutiny" for entry in overlay_group.curated_entries)
    finally:
        os.chdir(cwd)
        reset_env_dir = tmp_path / ".empty-fast-agent-start-path"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_build_snapshot_with_explicit_config_stays_scoped_to_config_project(tmp_path: Path) -> None:
    from pathlib import Path

    config_workspace = tmp_path / "config-workspace"
    config_workspace.mkdir(parents=True)
    config_path = config_workspace / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")

    cwd_workspace = tmp_path / "cwd-workspace"
    cwd_env_dir = cwd_workspace / ".fast-agent" / "model-overlays"
    cwd_env_dir.mkdir(parents=True)
    (cwd_env_dir / "sonnet.yaml").write_text(
        "\n".join(
            [
                "name: sonnet",
                "provider: anthropic",
                "model: claude-haiku-4-5",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        os.chdir(cwd_workspace)
        snapshot = build_snapshot(config_path=config_path, config_payload={})

        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.curated_entries == ()

        anthropic_option = next(
            option for option in snapshot.providers if option.option_key == Provider.ANTHROPIC.config_name
        )
        assert any(entry.alias == "sonnet" for entry in anthropic_option.curated_entries)
        assert all(not entry.local for entry in anthropic_option.curated_entries)
    finally:
        os.chdir(cwd)
        reset_env_dir = tmp_path / ".empty-fast-agent-config-scope"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_build_snapshot_with_explicit_project_config_ignores_parent_overlays(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True)
    config_path = project_root / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")

    parent_overlays = tmp_path / ".fast-agent" / "model-overlays"
    parent_overlays.mkdir(parents=True)
    (parent_overlays / "haikutiny.yaml").write_text(
        "\n".join(
            [
                "name: haikutiny",
                "provider: anthropic",
                "model: claude-haiku-4-5",
            ]
        ),
        encoding="utf-8",
    )

    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        snapshot = build_snapshot(config_path=config_path, config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.curated_entries == ()
    finally:
        reset_env_dir = tmp_path / ".empty-fast-agent-parent-scope"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_build_snapshot_loads_overlays_for_explicit_env_config_path(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    overlays_dir = env_dir / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "haikutiny.yaml").write_text(
        "\n".join(
            [
                "name: haikutiny",
                "provider: anthropic",
                "model: claude-haiku-4-5",
            ]
        ),
        encoding="utf-8",
    )
    config_path = env_dir / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")

    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        snapshot = build_snapshot(config_path=config_path, config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert any(entry.alias == "haikutiny" for entry in overlay_group.curated_entries)
    finally:
        reset_env_dir = tmp_path / ".empty-fast-agent-explicit-env-config"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
