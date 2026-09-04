"""The CLI must read and write config files as UTF-8, not in the platform codec.

`Settings` loads `fastagent.config.yaml` with an explicit `encoding="utf-8"`
(`config.py`), so the CLI commands that read and rewrite the same file have to
agree. Where they don't, a config with any non-ASCII in it either fails to load
under a non-UTF-8 default codec, or - worse - gets rewritten in that codec and
becomes a file the runtime can no longer read.

The tests force a default codec rather than relying on the host's, so they
assert the same thing on a UTF-8 CI runner as on a cp936 Windows box.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from typer.testing import CliRunner

from fast_agent.cli.commands import check_config as check_config_command
from fast_agent.cli.commands import config as config_command

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# Deliberately mixes a character that no legacy Chinese codec can represent
# (the coffee cup) with ones that cp936 can, so both failure modes are in reach.
NON_ASCII = "你好，世界 — café ☕"
CP936_SAFE = "团队默认横幅"


@contextlib.contextmanager
def _default_codec(codec: str) -> Iterator[None]:
    """Makes implicit text IO use ``codec``, as a non-UTF-8 locale would.

    `pathlib.Path.open` resolves the default encoding inside `io.open`, and
    patching `locale.getpreferredencoding` does not reach it, so the wrapper
    goes on `io.open` itself. Only calls that passed no encoding are affected -
    which is exactly the set of calls under test.
    """
    real_open = io.open

    def patched_open(
        file: Any,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "b" not in mode and encoding in (None, "locale"):
            encoding = codec
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io, "open", patched_open)
        yield


def _write_config(path: Path, banner: str) -> None:
    path.write_text(
        f'default_model: haiku\nshell_execution:\n  enabled: true\n  banner: "{banner}"\n',
        encoding="utf-8",
    )


def test_the_forced_codec_actually_bites(tmp_path: Path) -> None:
    """Guards the tests below from passing because the wrapper does nothing."""
    path = tmp_path / "probe.txt"
    path.write_text(NON_ASCII, encoding="utf-8")

    with _default_codec("ascii"), pytest.raises(UnicodeDecodeError):
        path.open().read()


def test_load_config_reads_utf8_under_a_non_utf8_default_codec(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path, NON_ASCII)

    with _default_codec("ascii"):
        config, resolved = config_command._load_config(config_path)

    assert resolved == config_path.resolve()
    assert config["shell_execution"]["banner"] == NON_ASCII


def test_load_effective_config_reads_utf8_under_a_non_utf8_default_codec(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path, NON_ASCII)

    with _default_codec("ascii"):
        config = config_command._load_effective_config(config_path)

    assert config["shell_execution"]["banner"] == NON_ASCII


def test_load_config_does_not_silently_mojibake_a_cp936_safe_value(
    tmp_path: Path,
) -> None:
    """The quiet half of the bug: cp936 decodes these UTF-8 bytes without error.

    Nothing raises, so the value simply arrives as different characters and is
    shown that way in the interactive form.
    """
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path, CP936_SAFE)

    with _default_codec("cp936"):
        config, _ = config_command._load_config(config_path)

    assert config["shell_execution"]["banner"] == CP936_SAFE


def test_saved_config_stays_loadable_by_the_runtime(tmp_path: Path) -> None:
    """A value entered in the form must survive back through the UTF-8 loader.

    This is the damaging case: written in the platform codec, the file is not
    valid UTF-8 any more, so `Settings` fails on a config the CLI just wrote.
    """
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path, "plain")

    with _default_codec("cp936"):
        config, _ = config_command._load_config(config_path)
        config["shell_execution"]["banner"] = CP936_SAFE
        config_command._save_config(config, config_path)

    # Read the way Settings does, and from the bytes on disk rather than through
    # any default codec.
    reloaded = yaml.safe_load(config_path.read_bytes().decode("utf-8"))
    assert reloaded["shell_execution"]["banner"] == CP936_SAFE


def test_get_config_summary_reads_utf8_under_a_non_utf8_default_codec(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path, NON_ASCII)

    with _default_codec("ascii"):
        summary = check_config_command.get_config_summary(config_path)

    assert summary["status"] == "parsed"
    assert summary["config"]["shell_execution"]["banner"] == NON_ASCII


def test_get_secrets_summary_reads_utf8_under_a_non_utf8_default_codec(
    tmp_path: Path,
) -> None:
    secrets_path = tmp_path / "fastagent.secrets.yaml"
    secrets_path.write_text(f'openai:\n  api_key: "{NON_ASCII}"\n', encoding="utf-8")

    with _default_codec("ascii"):
        summary = check_config_command.get_secrets_summary(secrets_path)

    assert summary["status"] == "parsed"
    assert summary["secrets"]["openai"]["api_key"] == NON_ASCII


def test_check_show_reads_utf8_under_a_non_utf8_default_codec(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path, NON_ASCII)

    with _default_codec("ascii"):
        result = CliRunner().invoke(check_config_command.app, ["show", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "YAML syntax is valid" in result.output
    assert "Error parsing" not in result.output
