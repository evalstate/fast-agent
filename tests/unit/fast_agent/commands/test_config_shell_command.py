from pathlib import Path

from fast_agent.cli.commands.config import (
    _build_model_form,
    _build_shell_form,
    _normalize_shell_updates,
)
from fast_agent.config import ShellSettings
from fast_agent.human_input.form_fields import IntegerField, StringField
from fast_agent.llm.provider_types import Provider


def test_build_shell_form_uses_minus_one_sentinel_for_show_all_lines() -> None:
    current = ShellSettings(output_display_lines=None)
    schema = _build_shell_form(current)

    field = schema.fields["output_display_lines"]
    assert isinstance(field, IntegerField)
    assert field.default == -1
    assert field.minimum == -1
    assert field.description is not None
    assert "-1 = show all" in field.description
    assert "0 = show none" in field.description


def test_build_shell_form_includes_write_text_file_mode_field() -> None:
    current = ShellSettings(write_text_file_mode="off")
    schema = _build_shell_form(current)

    mode_field = schema.fields["write_text_file_mode"]
    assert isinstance(mode_field, StringField)
    assert mode_field.default == "off"
    assert mode_field.description is not None
    assert "auto|on|off|apply_patch" in mode_field.description


def test_normalize_shell_updates_supports_none_zero_and_positive_line_modes() -> None:
    updates_show_all = _normalize_shell_updates(
        {
            "timeout_seconds": 90,
            "warning_interval_seconds": 30,
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "show_bash": True,
        }
    )
    assert updates_show_all["output_display_lines"] is None

    updates_show_none = _normalize_shell_updates(
        {
            "output_display_lines": 0,
            "output_byte_limit": 0,
            "show_bash": True,
        }
    )
    assert updates_show_none["output_display_lines"] == 0

    updates_show_some = _normalize_shell_updates(
        {
            "output_display_lines": 12,
            "output_byte_limit": 0,
            "show_bash": True,
        }
    )
    assert updates_show_some["output_display_lines"] == 12


def test_normalize_shell_updates_persists_filesystem_toggles() -> None:
    updates = _normalize_shell_updates(
        {
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "show_bash": True,
            "enable_read_text_file": False,
            "write_text_file_mode": "off",
        }
    )

    assert updates["enable_read_text_file"] is False
    assert updates["write_text_file_mode"] == "off"


def test_normalize_shell_updates_uses_write_text_file_mode() -> None:
    updates = _normalize_shell_updates(
        {
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "show_bash": True,
            "enable_read_text_file": True,
            "write_text_file_mode": "ON",
        }
    )

    assert updates["write_text_file_mode"] == "on"


def test_shell_settings_write_text_file_mode_accepts_yaml_boolean_values() -> None:
    assert ShellSettings.model_validate({"write_text_file_mode": False}).write_text_file_mode == "off"
    assert ShellSettings.model_validate({"write_text_file_mode": True}).write_text_file_mode == "on"


def test_normalize_shell_updates_accepts_apply_patch_mode() -> None:
    updates = _normalize_shell_updates(
        {
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "show_bash": True,
            "enable_read_text_file": True,
            "write_text_file_mode": "apply_patch",
        }
    )

    assert updates["write_text_file_mode"] == "apply_patch"


def test_shell_settings_write_text_file_mode_accepts_apply_patch_string() -> None:
    settings = ShellSettings.model_validate({"write_text_file_mode": "apply_patch"})
    assert settings.write_text_file_mode == "apply_patch"


def test_build_model_form_scopes_overlay_suggestions_to_config_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    env_dir = project_dir / ".fast-agent"
    ambient_env_dir = tmp_path / "ambient-env"
    config_path = project_dir / "fastagent.config.yaml"
    (env_dir / "model-overlays").mkdir(parents=True)
    (ambient_env_dir / "model-overlays").mkdir(parents=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("environment_dir: .fast-agent\n", encoding="utf-8")
    (env_dir / "model-overlays" / "projectoverlay.yaml").write_text(
        "name: projectoverlay\nprovider: openresponses\nmodel: overlay-tests/project\n",
        encoding="utf-8",
    )
    (ambient_env_dir / "model-overlays" / "ambientoverlay.yaml").write_text(
        "name: ambientoverlay\nprovider: openresponses\nmodel: overlay-tests/ambient\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENVIRONMENT_DIR", str(ambient_env_dir))
    monkeypatch.setattr(
        "fast_agent.cli.commands.config.ModelSelectionCatalog.configured_providers",
        lambda *args, **kwargs: [Provider.OPENRESPONSES],
    )

    schema = _build_model_form(
        None,
        {
            "environment_dir": ".fast-agent",
            "openresponses": {"api_key": "test-key"},
        },
        config_path=config_path,
    )

    field = schema.fields["default_model"]
    assert field.description is not None
    assert "openresponses.overlay-tests/project" in field.description
    assert "openresponses.overlay-tests/ambient" not in field.description
