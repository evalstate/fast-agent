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
from fast_agent.llm.model_selection import CatalogModelEntry
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker_common import (
    ANTHROPIC_VERTEX_PROVIDER_KEY,
    DEFAULT_VALUE,
    GENERIC_CUSTOM_MODEL_SENTINEL,
    LLAMACPP_IMPORT_SENTINEL,
    LLAMACPP_PROVIDER_KEY,
    ModelOption,
    ModelPickerSnapshot,
    ProviderOption,
    _provider_is_active,
    apply_option_overrides,
    build_snapshot,
    format_catalog_model_entry_label,
    has_explicit_provider_prefix,
    infer_initial_picker_provider,
    is_model_availability,
    model_capabilities,
    model_options_for_option,
    model_options_for_provider,
    provider_option_count_label,
    provider_option_status_label,
    web_search_display,
)

if TYPE_CHECKING:
    from pathlib import Path


def _overlay_group(snapshot: ModelPickerSnapshot):
    return next(option for option in snapshot.providers if option.overlay_group)


def test_is_model_availability_narrows_known_values() -> None:
    assert is_model_availability("attention")
    assert not is_model_availability("disabled")


def test_generic_provider_uses_custom_local_model_option() -> None:
    snapshot = build_snapshot()

    options = model_options_for_provider(snapshot, Provider.GENERIC, source="curated")

    assert options == [
        ModelOption(
            spec=GENERIC_CUSTOM_MODEL_SENTINEL,
            label="Enter local model string (e.g. llama3.2)",
        )
    ]


def test_refer_to_docs_provider_uses_synthetic_setup_option() -> None:
    options = model_options_for_provider(build_snapshot(), Provider.OPENROUTER, source="curated")

    assert options == [
        ModelOption(
            spec="openrouter.refer-to-docs",
            label="Refer to docs (provider-specific setup)",
        )
    ]


def test_format_catalog_model_entry_label_includes_tags_and_description() -> None:
    label = format_catalog_model_entry_label(
        CatalogModelEntry(
            alias="local-fast",
            model="anthropic.local-fast",
            current=False,
            fast=True,
            local=True,
            display_label="Local Fast",
            description="test overlay",
        ),
        spec="anthropic.local-fast?reasoning=low",
    )

    assert label == (
        "Local Fast          → anthropic.local-fast?reasoning=low "
        "(local, fast, legacy) — test overlay"
    )


def test_curated_scope_hides_non_current_catalog_entries(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir(parents=True, exist_ok=True)
    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)
    try:
        snapshot = build_snapshot(config_payload={})
    finally:
        reset_env_dir = tmp_path / ".empty-fast-agent-curated-scope"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    hf_option = next(option for option in snapshot.providers if option.option_key == "hf")

    curated_options = model_options_for_option(hf_option, source="curated")
    all_options = model_options_for_option(hf_option, source="all")

    curated_tokens = {option.preset_token for option in curated_options}
    all_tokens = {option.preset_token for option in all_options}

    assert "glm47" not in curated_tokens
    assert "glm47" in all_tokens


def test_openresponses_models_do_not_report_web_search_support() -> None:
    capabilities = model_capabilities("openresponses.gpt-5-mini")

    assert capabilities.provider == Provider.OPENRESPONSES
    assert capabilities.web_search_supported is False


def test_provider_option_count_label_names_curated_models() -> None:
    option = ProviderOption(
        provider=Provider.ANTHROPIC,
        active=True,
        curated_entries=(CatalogModelEntry(alias="sonnet", model="anthropic.sonnet"),),
    )

    assert provider_option_count_label(option) == "1 curated model"

    plural_option = ProviderOption(
        provider=Provider.ANTHROPIC,
        active=True,
        curated_entries=(
            CatalogModelEntry(alias="sonnet", model="anthropic.sonnet"),
            CatalogModelEntry(alias="haiku", model="anthropic.haiku"),
        ),
    )

    assert provider_option_count_label(plural_option) == "2 curated models"


def test_provider_option_count_label_names_overlay_groups() -> None:
    option = ProviderOption(
        provider=None,
        active=True,
        curated_entries=(CatalogModelEntry(alias="one", model="responses.one", local=True),),
        key="overlays",
        display_name="Overlays",
        overlay_group=True,
    )

    assert provider_option_count_label(option) == "1 overlay"

    plural_option = ProviderOption(
        provider=None,
        active=True,
        curated_entries=(
            CatalogModelEntry(alias="one", model="responses.one", local=True),
            CatalogModelEntry(alias="two", model="responses.two", local=True),
        ),
        key="overlays",
        display_name="Overlays",
        overlay_group=True,
    )

    assert provider_option_count_label(plural_option) == "2 overlays"


def test_provider_option_count_label_names_llamacpp_import_flow() -> None:
    option = ProviderOption(
        provider=None,
        active=False,
        curated_entries=(),
        key=LLAMACPP_PROVIDER_KEY,
        display_name="llama.cpp",
    )

    assert provider_option_count_label(option) == "import flow"


def test_provider_is_active_accepts_provider_specific_fallbacks() -> None:
    assert _provider_is_active(
        Provider.GOOGLE,
        {"google": {"vertex_ai": {"enabled": True}}},
    )
    assert _provider_is_active(
        Provider.AZURE,
        {
            "azure": {
                "use_default_azure_credential": True,
                "base_url": "https://example.openai.azure.com",
            }
        },
    )
    assert _provider_is_active(
        Provider.AZURE,
        {
            "azure": {
                "use_default_azure_credential": True,
                "base_url": " https://example.openai.azure.com ",
            }
        },
    )
    assert not _provider_is_active(
        Provider.AZURE,
        {
            "azure": {
                "use_default_azure_credential": True,
                "base_url": "   ",
            }
        },
    )


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


def test_overlay_group_curated_scope_includes_non_current_local_overlays(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / ".fast-agent"
    overlays_dir = env_dir / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "legacylocal.yaml").write_text(
        "\n".join(
            [
                "name: legacylocal",
                "provider: anthropic",
                "model: claude-haiku-4-5",
                "picker:",
                "  current: false",
            ]
        ),
        encoding="utf-8",
    )

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)
    try:
        snapshot = build_snapshot(config_payload={})
        overlay_group = _overlay_group(snapshot)
        overlay_options = model_options_for_option(overlay_group, source="curated")
        assert any(option.preset_token == "legacylocal" for option in overlay_options)
    finally:
        empty_env_dir = tmp_path / ".empty-fast-agent-legacy-overlays"
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
    assert model_options_for_option(option, source="curated") == [
        ModelOption(
            spec=LLAMACPP_IMPORT_SENTINEL,
            label="Discover local llama.cpp models and write overlay",
        )
    ]

    option_keys = [provider.option_key for provider in snapshot.providers]
    assert option_keys.index(LLAMACPP_PROVIDER_KEY) == option_keys.index(Provider.DEEPSEEK.config_name) - 1

    generic_option = next(provider for provider in snapshot.providers if provider.option_key == Provider.GENERIC.config_name)
    assert generic_option.option_display_name == "Generic (ollama)"


def test_build_snapshot_places_deepseek_under_llamacpp() -> None:
    snapshot = build_snapshot(config_payload={})
    option_keys = [provider.option_key for provider in snapshot.providers]

    assert option_keys.index(Provider.DEEPSEEK.config_name) == option_keys.index(LLAMACPP_PROVIDER_KEY) + 1


def test_build_snapshot_uses_xai_brand_casing() -> None:
    snapshot = build_snapshot(config_payload={})
    option = next(provider for provider in snapshot.providers if provider.option_key == Provider.XAI.config_name)

    assert option.option_display_name == "xAI"


def test_provider_option_status_label_is_explicit() -> None:
    active = ProviderOption(provider=Provider.ANTHROPIC, active=True, curated_entries=())
    disabled = ProviderOption(
        provider=Provider.GOOGLE,
        active=False,
        curated_entries=(),
        disabled_reason="missing ADC",
    )
    inactive = ProviderOption(provider=Provider.GENERIC, active=False, curated_entries=())
    import_flow = ProviderOption(
        provider=None,
        active=False,
        curated_entries=(),
        key=LLAMACPP_PROVIDER_KEY,
        display_name="llama.cpp",
    )

    assert provider_option_status_label(active) == "active"
    assert provider_option_status_label(disabled) == "disabled"
    assert provider_option_status_label(inactive) == "inactive"
    assert provider_option_status_label(import_flow) == "active"


def test_apply_option_overrides_sets_and_clears_query_values() -> None:
    result = apply_option_overrides(
        "responses.gpt-5?reasoning=high&web_search=on&context=1m",
        reasoning_value="low",
        web_search_value=DEFAULT_VALUE,
        context_value="2m",
    )

    assert result == "responses.gpt-5?reasoning=low&context=2m"


def test_web_search_display_formats_optional_boolean_state() -> None:
    assert web_search_display(None) == "default"
    assert web_search_display(True) == "on"
    assert web_search_display(False) == "off"


def test_has_explicit_provider_prefix_handles_supported_delimiters() -> None:
    assert has_explicit_provider_prefix("openai.gpt-4.1") is True
    assert has_explicit_provider_prefix("openai/gpt-4.1") is True
    assert has_explicit_provider_prefix("gpt-4.1") is False
    assert has_explicit_provider_prefix("openai") is False
    assert has_explicit_provider_prefix("unknown/model") is False


def test_refer_to_docs_providers_show_docs_option() -> None:
    snapshot = build_snapshot(config_payload={})
    option = next(provider for provider in snapshot.providers if provider.provider == Provider.AZURE)

    assert model_options_for_option(option, source="curated") == [
        ModelOption(
            spec="azure.refer-to-docs",
            label="Refer to docs (provider-specific setup)",
        )
    ]


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


def test_build_snapshot_loads_overlays_when_settings_config_lives_in_env_dir(
    tmp_path: Path,
) -> None:
    from fast_agent.config import Settings
    from fast_agent.llm.model_reference_config import resolve_model_reference_start_path

    workspace = tmp_path / "workspace"
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
    config_path = env_dir / "fast-agent.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")
    settings = Settings(environment_dir=None)
    settings._config_file = str(config_path)
    start_path = resolve_model_reference_start_path(settings=settings)

    previous_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    try:
        snapshot = build_snapshot(
            config_payload={"environment_dir": None},
            start_path=start_path,
        )
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert any(entry.alias == "haikutiny" for entry in overlay_group.curated_entries)
    finally:
        reset_env_dir = tmp_path / ".empty-fast-agent-settings-env-config"
        reset_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=reset_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir
