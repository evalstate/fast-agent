from pathlib import Path

import pytest

from fast_agent.cli.commands import config as config_command
from fast_agent.cli.commands.config import (
    _build_shell_form,
    _normalize_shell_updates,
)
from fast_agent.config import ShellSettings
from fast_agent.constants import (
    MAX_FOREGROUND_AUTO_AWAIT_SECONDS,
    MAX_PROCESS_POLL_WAIT_SECONDS,
)
from fast_agent.human_input.form_fields import FormSchema, IntegerField, StringField


def test_build_shell_form_uses_minus_one_sentinel_for_show_all_lines() -> None:
    current = ShellSettings(output_display_lines=None)
    schema = _build_shell_form(current)

    field = schema.fields["output_display_lines"]
    assert isinstance(field, IntegerField)
    assert field.title == "Output Display Lines"
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
    assert mode_field.title == "Write Text File Mode"
    assert mode_field.default == "off"
    assert mode_field.description is not None
    assert "auto|on|off|apply_patch" in mode_field.description


@pytest.mark.parametrize(
    ("field_name", "expected_default"),
    [
        ("retained_output_max_bytes", 2 * 1024 * 1024),
        ("durable_output_max_bytes", 2 * 1024 * 1024),
    ],
)
def test_build_shell_form_allows_default_retained_output_quota(
    field_name: str,
    expected_default: int,
) -> None:
    current = ShellSettings()
    schema = _build_shell_form(current)

    field = schema.fields[field_name]
    assert isinstance(field, IntegerField)
    assert field.default == expected_default
    assert field.maximum is not None
    assert field.maximum >= expected_default


def test_build_shell_form_uses_managed_process_wait_ceiling() -> None:
    current = ShellSettings()
    schema = _build_shell_form(current)

    field = schema.fields["process_poll_max_wait_seconds"]
    assert isinstance(field, IntegerField)
    assert field.default == MAX_PROCESS_POLL_WAIT_SECONDS
    assert field.maximum == MAX_PROCESS_POLL_WAIT_SECONDS


def test_foreground_auto_await_setting_has_bounded_zero_opt_out() -> None:
    current = ShellSettings()
    schema = _build_shell_form(current)

    field = schema.fields["foreground_auto_await_max_seconds"]
    assert isinstance(field, IntegerField)
    assert field.default == 240
    assert field.minimum == 0
    assert field.maximum == MAX_FOREGROUND_AUTO_AWAIT_SECONDS

    assert ShellSettings(foreground_auto_await_max_seconds=0).foreground_auto_await_max_seconds == 0
    assert (
        ShellSettings.model_validate(
            {"foreground_auto_await_max_seconds": "4m"}
        ).foreground_auto_await_max_seconds
        == 240
    )
    assert (
        ShellSettings(
            foreground_auto_await_max_seconds=MAX_FOREGROUND_AUTO_AWAIT_SECONDS
        ).foreground_auto_await_max_seconds
        == MAX_FOREGROUND_AUTO_AWAIT_SECONDS
    )
    with pytest.raises(ValueError):
        ShellSettings(foreground_auto_await_max_seconds=MAX_FOREGROUND_AUTO_AWAIT_SECONDS + 1)
    with pytest.raises(
        TypeError,
        match="foreground_auto_await_max_seconds must be an integer",
    ):
        ShellSettings.model_validate({"foreground_auto_await_max_seconds": True})
    for value in (-0.1, 0.9, 1.1):
        with pytest.raises(
            TypeError,
            match="foreground_auto_await_max_seconds must be an integer",
        ):
            ShellSettings.model_validate({"foreground_auto_await_max_seconds": value})


def test_shell_settings_rejects_managed_process_wait_above_ceiling() -> None:
    assert (
        ShellSettings(
            process_poll_max_wait_seconds=MAX_PROCESS_POLL_WAIT_SECONDS
        ).process_poll_max_wait_seconds
        == MAX_PROCESS_POLL_WAIT_SECONDS
    )
    with pytest.raises(ValueError):
        ShellSettings(process_poll_max_wait_seconds=MAX_PROCESS_POLL_WAIT_SECONDS + 1)


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


def test_normalize_shell_updates_rejects_boolean_timeout_values() -> None:
    updates = _normalize_shell_updates(
        {
            "timeout_seconds": True,
            "warning_interval_seconds": False,
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "show_bash": True,
        }
    )

    assert "timeout_seconds" not in updates
    assert "warning_interval_seconds" not in updates


def test_normalize_shell_updates_preserves_foreground_auto_await_opt_out() -> None:
    updates = _normalize_shell_updates(
        {
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "foreground_auto_await_max_seconds": 0,
        }
    )

    assert updates["foreground_auto_await_max_seconds"] == 0


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


def test_normalize_shell_updates_persists_retained_output_settings() -> None:
    updates = _normalize_shell_updates(
        {
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "retain_truncated_output": False,
            "retained_output_max_bytes": 4 * 1024 * 1024,
            "durable_output_max_bytes": 8 * 1024 * 1024,
            "prefer_local_shell": True,
        }
    )

    assert updates["retain_truncated_output"] is False
    assert updates["retained_output_max_bytes"] == 4 * 1024 * 1024
    assert updates["durable_output_max_bytes"] == 8 * 1024 * 1024
    assert updates["prefer_local_shell"] is True


def test_shell_config_save_preserves_fields_omitted_from_form(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        """
shell_execution:
  tool_profile: native
  retained_output_temp_directory: /private/spools
  show_bash: false
""".lstrip(),
        encoding="utf-8",
    )

    def submit_defaults(schema: FormSchema, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {name: field.default for name, field in schema.fields.items()}

    monkeypatch.setattr(config_command, "form_sync", submit_defaults)

    config_command.config_shell(config_path)

    saved, _ = config_command._load_config(config_path)
    shell = saved["shell_execution"]
    assert shell["tool_profile"] == "native"
    assert shell["retained_output_temp_directory"] == "/private/spools"
    assert shell["show_bash"] is False


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
    assert (
        ShellSettings.model_validate({"write_text_file_mode": False}).write_text_file_mode == "off"
    )
    assert ShellSettings.model_validate({"write_text_file_mode": True}).write_text_file_mode == "on"
    assert (
        ShellSettings.model_validate({"write_text_file_mode": "enable"}).write_text_file_mode
        == "on"
    )
    assert (
        ShellSettings.model_validate({"write_text_file_mode": "disable"}).write_text_file_mode
        == "off"
    )


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


def test_normalize_shell_updates_uses_shared_write_text_file_mode_aliases() -> None:
    updates = _normalize_shell_updates(
        {
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "show_bash": True,
            "enable_read_text_file": True,
            "write_text_file_mode": "yes",
        }
    )

    assert updates["write_text_file_mode"] == "on"


def test_normalize_shell_updates_defaults_invalid_write_text_file_mode() -> None:
    updates = _normalize_shell_updates(
        {
            "output_display_lines": -1,
            "output_byte_limit": 0,
            "show_bash": True,
            "enable_read_text_file": True,
            "write_text_file_mode": "sometimes",
        }
    )

    assert updates["write_text_file_mode"] == "auto"


def test_shell_settings_write_text_file_mode_accepts_apply_patch_string() -> None:
    settings = ShellSettings.model_validate({"write_text_file_mode": "apply_patch"})
    assert settings.write_text_file_mode == "apply_patch"


def test_shell_settings_write_text_file_mode_accepts_edit_file_string() -> None:
    settings = ShellSettings.model_validate({"write_text_file_mode": "edit_file"})
    assert settings.write_text_file_mode == "edit_file"
