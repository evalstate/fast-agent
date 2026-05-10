from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _card_text(name: str, *, model: str | None = None) -> str:
    model_lines = [f"model: {model}"] if model else []
    return "\n".join(
        [
            "---",
            "type: agent",
            f"name: {name}",
            *model_lines,
            "---",
            "Return ok.",
            "",
        ]
    )


def test_load_agents_supports_file_uri_agent_card(tmp_path: Path) -> None:
    card_path = tmp_path / "file_agent.md"
    card_path.write_text(_card_text("file_agent"), encoding="utf-8")
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    loaded_names = fast.load_agents(card_path.as_uri())

    assert loaded_names == ["file_agent"]
    assert "file_agent" in fast.agents


def test_load_agents_supports_hf_uri_agent_card(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_agent.io import source_resolver

    def fake_read_text_source(source: str, *, label: str) -> str:
        assert source == "hf://buckets/evalstate/home/remote_agent.md"
        assert label == "AgentCard URL"
        return _card_text("remote_agent")

    monkeypatch.setattr(source_resolver, "read_text_source", fake_read_text_source)
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    loaded_names = fast.load_agents("hf://buckets/evalstate/home/remote_agent.md")

    assert loaded_names == ["remote_agent"]
    assert "remote_agent" in fast.agents


def test_load_agents_defaults_extensionless_remote_agent_card_to_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.io import source_resolver

    def fake_read_text_source(source: str, *, label: str) -> str:
        assert source == "hf://buckets/evalstate/home/remote_agent"
        assert label == "AgentCard URL"
        return _card_text("remote_agent", model="passthrough")

    monkeypatch.setattr(source_resolver, "read_text_source", fake_read_text_source)
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    loaded_names = fast.load_agents("hf://buckets/evalstate/home/remote_agent")

    assert loaded_names == ["remote_agent"]
    config = fast.agents["remote_agent"]["config"]
    assert isinstance(config, AgentConfig)
    assert config.model == "passthrough"
