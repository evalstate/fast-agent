from __future__ import annotations

from pathlib import Path

import pytest

from fast_agent.core.agent_card_paths import (
    agent_card_suffix,
    is_agent_card_path,
    is_markdown_agent_card_path,
    is_yaml_agent_card_path,
)


@pytest.mark.parametrize(
    ("path", "suffix"),
    [
        (Path("agent.MD"), ".md"),
        (Path("agent.MARKDOWN"), ".markdown"),
        (Path("agent.YAML"), ".yaml"),
        (Path("agent.YML"), ".yml"),
    ],
)
def test_agent_card_suffix_normalizes_case(path: Path, suffix: str) -> None:
    assert agent_card_suffix(path) == suffix


def test_agent_card_path_classifiers_use_normalized_suffixes() -> None:
    assert is_agent_card_path(Path("agent.YAML"))
    assert is_yaml_agent_card_path(Path("agent.YML"))
    assert is_markdown_agent_card_path(Path("agent.MARKDOWN"))
    assert not is_agent_card_path(Path("agent.JSON"))
