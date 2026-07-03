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
    GENERIC_CUSTOM_MODEL_SENTINEL,
    LLAMACPP_IMPORT_SENTINEL,
    LLAMACPP_PROVIDER_KEY,
    ModelOption,
    ModelPickerSnapshot,
    ProviderOption,
    _provider_is_active,
    build_snapshot,
    format_catalog_model_entry_label,
    has_explicit_provider_prefix,
    infer_initial_picker_provider,
    model_capabilities,
    model_options_for_option,
    model_options_for_provider,
    provider_option_count_label,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _overlay_group(snapshot: ModelPickerSnapshot):
    return next(option for option in snapshot.providers if option.overlay_group)


def _provider_option(snapshot: ModelPickerSnapshot, provider: Provider) -> ProviderOption:
    return next(option for option in snapshot.providers if option.provider == provider)


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
    home = tmp_path / ".fast-agent"
    home.mkdir(parents=True, exist_ok=True)
    previous_home = os.environ.get("FAST_AGENT_HOME")
    os.environ["FAST_AGENT_HOME"] = str(home)
    try:
        snapshot = build_snapshot(config_payload={})
    finally:
        reset_home = tmp_path / ".empty-fast-agent-curated-scope"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home

    hf_option = next(option for option in snapshot.providers if option.option_key == "hf")

    curated_options = model_options_for_option(hf_option, source="curated")
    all_options = model_options_for_option(hf_option, source="all")

    curated_tokens = {option.preset_token for option in curated_options}
    all_tokens = {option.preset_token for option in all_options}

    assert "glm47" not in curated_tokens
    assert "glm47" in all_tokens


def test_huggingface_provider_is_active_when_hub_login_is_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fast_agent.ui.model_picker_common.is_huggingface_hub_logged_in",
        lambda: True,
    )

    snapshot = build_snapshot(config_payload={})

    assert _provider_option(snapshot, Provider.HUGGINGFACE).active is True


def test_huggingface_provider_is_inactive_without_hub_login_or_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "fast_agent.ui.model_picker_common.is_huggingface_hub_logged_in",
        lambda: False,
    )

    snapshot = build_snapshot(config_payload={})

    assert _provider_option(snapshot, Provider.HUGGINGFACE).active is False


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
    assert capabilities.long_context_window is None


def test_infer_initial_picker_provider_uses_vertex_group_for_anthropic_vertex() -> None:
    assert (
        infer_initial_picker_provider("anthropic-vertex.claude-sonnet-4-6")
        == Provider.ANTHROPIC_VERTEX.config_name
    )


def test_build_snapshot_surfaces_overlays_as_a_separate_group(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    overlays_dir = home / "model-overlays"
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

    previous_home = os.environ.get("FAST_AGENT_HOME")
    os.environ["FAST_AGENT_HOME"] = str(home)
    try:
        snapshot = build_snapshot(config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.display_name == "Overlays"
        assert overlay_group.overlay_group is True
        assert all(option.option_key != "openresponses" for option in snapshot.providers)
        assert any(option.option_key == "openrouter" for option in snapshot.providers)
        assert any(option.option_key == "azure" for option in snapshot.providers)
        assert any(option.option_key == "bedrock" for option in snapshot.providers)
    finally:
        empty_home = tmp_path / ".empty-fast-agent"
        empty_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=empty_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


def test_overlay_group_curated_scope_includes_non_current_local_overlays(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".fast-agent"
    overlays_dir = home / "model-overlays"
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

    previous_home = os.environ.get("FAST_AGENT_HOME")
    os.environ["FAST_AGENT_HOME"] = str(home)
    try:
        snapshot = build_snapshot(config_payload={})
        overlay_group = _overlay_group(snapshot)
        overlay_options = model_options_for_option(overlay_group, source="curated")
        assert any(option.preset_token == "legacylocal" for option in overlay_options)
    finally:
        empty_home = tmp_path / ".empty-fast-agent-legacy-overlays"
        empty_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=empty_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


def test_build_snapshot_shows_empty_overlay_group_even_without_overlays(tmp_path: Path) -> None:
    empty_home = tmp_path / ".fast-agent"
    empty_home.mkdir(parents=True, exist_ok=True)
    previous_home = os.environ.get("FAST_AGENT_HOME")
    os.environ["FAST_AGENT_HOME"] = str(empty_home)
    try:
        snapshot = build_snapshot(config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.overlay_group is True
        assert overlay_group.curated_entries == ()
        assert any(option.option_key == "fast-agent" for option in snapshot.providers)
    finally:
        reset_home = tmp_path / ".empty-fast-agent-reset"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


def test_build_snapshot_includes_llamacpp_import_flow(tmp_path: Path) -> None:
    empty_home = tmp_path / ".fast-agent"
    empty_home.mkdir(parents=True, exist_ok=True)
    previous_home = os.environ.get("FAST_AGENT_HOME")
    os.environ["FAST_AGENT_HOME"] = str(empty_home)
    try:
        snapshot = build_snapshot(config_payload={})
    finally:
        reset_home = tmp_path / ".empty-fast-agent-llamacpp"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home

    option = next(
        provider for provider in snapshot.providers if provider.option_key == LLAMACPP_PROVIDER_KEY
    )

    assert option.display_name == "llama.cpp"
    assert option.active is False
    assert model_options_for_option(option, source="curated") == [
        ModelOption(
            spec=LLAMACPP_IMPORT_SENTINEL,
            label="Discover local llama.cpp models and write overlay",
        )
    ]

    option_keys = [provider.option_key for provider in snapshot.providers]
    assert (
        option_keys.index(LLAMACPP_PROVIDER_KEY)
        == option_keys.index(Provider.DEEPSEEK.config_name) - 1
    )

    generic_option = next(
        provider
        for provider in snapshot.providers
        if provider.option_key == Provider.GENERIC.config_name
    )
    assert generic_option.display_name == "Generic (ollama)"


def test_build_snapshot_places_deepseek_under_llamacpp() -> None:
    snapshot = build_snapshot(config_payload={})
    option_keys = [provider.option_key for provider in snapshot.providers]

    assert (
        option_keys.index(Provider.DEEPSEEK.config_name)
        == option_keys.index(LLAMACPP_PROVIDER_KEY) + 1
    )


def test_build_snapshot_uses_xai_brand_casing() -> None:
    snapshot = build_snapshot(config_payload={})
    option = next(
        provider
        for provider in snapshot.providers
        if provider.option_key == Provider.XAI.config_name
    )

    assert option.provider is Provider.XAI
    assert option.display_name is None
    assert Provider.XAI.display_name == "xAI"


def test_has_explicit_provider_prefix_handles_supported_delimiters() -> None:
    assert has_explicit_provider_prefix("openai.gpt-4.1") is True
    assert has_explicit_provider_prefix("openai/gpt-4.1") is True
    assert has_explicit_provider_prefix("gpt-4.1") is False
    assert has_explicit_provider_prefix("openai") is False
    assert has_explicit_provider_prefix("unknown/model") is False


def test_refer_to_docs_providers_show_docs_option() -> None:
    snapshot = build_snapshot(config_payload={})
    option = next(
        provider for provider in snapshot.providers if provider.provider == Provider.AZURE
    )

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
    home = workspace / ".fast-agent"
    overlays_dir = home / "model-overlays"
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
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
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
        reset_home = tmp_path / ".empty-fast-agent-config-path"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


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
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        os.chdir(outside_cwd)
        snapshot = build_snapshot(
            config_payload={"home": ".fast-agent"},
            start_path=project_root,
        )
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert any(entry.alias == "haikutiny" for entry in overlay_group.curated_entries)
    finally:
        os.chdir(cwd)
        reset_home = tmp_path / ".empty-fast-agent-start-path"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


def test_build_snapshot_with_explicit_config_stays_scoped_to_config_project(tmp_path: Path) -> None:
    from pathlib import Path

    config_workspace = tmp_path / "config-workspace"
    config_workspace.mkdir(parents=True)
    config_path = config_workspace / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")

    cwd_workspace = tmp_path / "cwd-workspace"
    cwd_home = cwd_workspace / ".fast-agent" / "model-overlays"
    cwd_home.mkdir(parents=True)
    (cwd_home / "sonnet.yaml").write_text(
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
    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        os.chdir(cwd_workspace)
        snapshot = build_snapshot(config_path=config_path, config_payload={})

        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.curated_entries == ()

        anthropic_option = next(
            option
            for option in snapshot.providers
            if option.option_key == Provider.ANTHROPIC.config_name
        )
        assert any(entry.alias == "sonnet" for entry in anthropic_option.curated_entries)
        assert all(not entry.local for entry in anthropic_option.curated_entries)
    finally:
        os.chdir(cwd)
        reset_home = tmp_path / ".empty-fast-agent-config-scope"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


def test_build_snapshot_with_explicit_project_config_ignores_parent_overlays(
    tmp_path: Path,
) -> None:
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

    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        snapshot = build_snapshot(config_path=config_path, config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert overlay_group.curated_entries == ()
    finally:
        reset_home = tmp_path / ".empty-fast-agent-parent-scope"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


def test_build_snapshot_loads_overlays_for_explicit_env_config_path(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    overlays_dir = home / "model-overlays"
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
    config_path = home / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")

    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        snapshot = build_snapshot(config_path=config_path, config_payload={})
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert any(entry.alias == "haikutiny" for entry in overlay_group.curated_entries)
    finally:
        reset_home = tmp_path / ".empty-fast-agent-explicit-env-config"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


def test_build_snapshot_loads_overlays_when_settings_config_lives_in_home(
    tmp_path: Path,
) -> None:
    from fast_agent.config import Settings
    from fast_agent.llm.model_reference_config import resolve_model_reference_start_path

    workspace = tmp_path / "workspace"
    home = workspace / ".fast-agent"
    overlays_dir = home / "model-overlays"
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
    config_path = home / "fast-agent.yaml"
    config_path.write_text("default_model: sonnet\n", encoding="utf-8")
    settings = Settings(home=None)
    settings._config_file = str(config_path)
    start_path = resolve_model_reference_start_path(settings=settings)

    previous_home = os.environ.pop("FAST_AGENT_HOME", None)
    try:
        snapshot = build_snapshot(
            config_payload={"home": None},
            start_path=start_path,
        )
        overlay_group = _overlay_group(snapshot)
        assert overlay_group.option_key == "overlays"
        assert any(entry.alias == "haikutiny" for entry in overlay_group.curated_entries)
    finally:
        reset_home = tmp_path / ".empty-fast-agent-settings-env-config"
        reset_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=reset_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home
