import pytest

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.batch.agent_card import (
    force_loaded_card_history_off,
    load_batch_agent_card,
    override_selected_agent_model,
)


def _fast(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    return FastAgent(
        name="test",
        parse_cli_args=False,
        ignore_unknown_args=True,
        quiet=True,
        environment_dir=env_dir,
    )


def _card(path, *, name, extra="", model="passthrough"):
    path.write_text(
        f"---\nname: {name}\nmodel: {model}\n{extra}---\n\nTest agent.\n",
        encoding="utf-8",
    )


def test_single_runnable_agent_is_selected(tmp_path):
    card = tmp_path / "extractor.md"
    _card(card, name="extractor")

    selection = load_batch_agent_card(_fast(tmp_path), str(card), None)

    assert selection.target_name == "extractor"
    assert selection.loaded_names == ["extractor"]


def test_directory_default_agent_is_selected(tmp_path):
    cards = tmp_path / "cards"
    cards.mkdir()
    _card(cards / "extractor.md", name="extractor")
    _card(cards / "verifier.md", name="verifier", extra="default: true\n")

    selection = load_batch_agent_card(_fast(tmp_path), str(cards), None)

    assert selection.target_name == "verifier"


def test_ambiguous_directory_requires_agent(tmp_path):
    cards = tmp_path / "cards"
    cards.mkdir()
    _card(cards / "extractor.md", name="extractor")
    _card(cards / "verifier.md", name="verifier")

    with pytest.raises(ValueError, match=r"multiple runnable agents: extractor, verifier"):
        load_batch_agent_card(_fast(tmp_path), str(cards), None)


def test_requested_tool_only_agent_fails(tmp_path):
    card = tmp_path / "helper.md"
    _card(card, name="helper", extra="tool_only: true\n")

    with pytest.raises(ValueError, match="tool_only"):
        load_batch_agent_card(_fast(tmp_path), str(card), "helper")


def test_human_input_agent_fails(tmp_path):
    card = tmp_path / "interactive.md"
    _card(card, name="interactive", extra="human_input: true\n")

    with pytest.raises(ValueError, match="human_input agents: interactive"):
        load_batch_agent_card(_fast(tmp_path), str(card), None)


def test_history_and_model_mutators_update_loaded_config(tmp_path):
    card = tmp_path / "extractor.md"
    _card(
        card,
        name="extractor",
        model="sonnet",
        extra="use_history: true\nrequest_params:\n  use_history: true\n",
    )
    fast = _fast(tmp_path)
    selection = load_batch_agent_card(fast, str(card), None)

    force_loaded_card_history_off(fast, selection.loaded_names)
    override_selected_agent_model(fast, selection.target_name, "passthrough")

    config = fast.agents["extractor"]["config"]
    assert isinstance(config, AgentConfig)
    assert config.use_history is False
    assert config.default_request_params is not None
    assert config.default_request_params.use_history is False
    assert config.model == "passthrough"
