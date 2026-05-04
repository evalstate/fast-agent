from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.config import get_settings

if TYPE_CHECKING:
    from pathlib import Path


def test_settings_parses_global_plugin_commands(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "commands:",
                "  draft-next:",
                "    description: Draft the next user message",
                "    input_hint: \"[format]\"",
                "    handler: \"commands.py:draft_next\"",
                "    key: \"c-x d\"",
            ]
        ),
        encoding="utf-8",
    )

    settings = get_settings(config_path)

    assert settings.commands is not None
    assert settings.commands["draft-next"].description == "Draft the next user message"
    assert settings.commands["draft-next"].handler == "commands.py:draft_next"
    assert settings.commands["draft-next"].input_hint == "[format]"
    assert settings.commands["draft-next"].key == "c-x d"
