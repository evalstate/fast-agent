from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.llm.model_reference_diagnostics import (
    ModelReferenceSetupItem,
    collect_model_reference_setup_diagnostics,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_collect_model_reference_setup_diagnostics_reports_pack_and_card_references(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    env_dir = workspace / ".fast-agent"
    workspace.mkdir(parents=True)

    (workspace / "fastagent.config.yaml").write_text(
        'default_model: "$system.default"\n'
        "model_references:\n"
        "  system:\n"
        '    default: ""\n',
        encoding="utf-8",
    )

    pack_dir = env_dir / "card-packs" / "smart"
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "card-pack.yaml").write_text(
        "schema_version: 1\n"
        "name: smart\n"
        "kind: card\n"
        "model_references_required:\n"
        "  - $system.default\n"
        "model_references_recommended:\n"
        "  - $system.fast\n"
        "install:\n"
        "  agent_cards: []\n"
        "  tool_cards: []\n"
        "  files: []\n",
        encoding="utf-8",
    )

    agent_cards_dir = env_dir / "agent-cards"
    agent_cards_dir.mkdir(parents=True, exist_ok=True)
    (agent_cards_dir / "helper.yaml").write_text(
        'name: helper\n'
        "type: agent\n"
        'instruction: "Be helpful."\n'
        'model: "$system.fast"\n',
        encoding="utf-8",
    )

    diagnostics = collect_model_reference_setup_diagnostics(
        cwd=workspace,
        env_dir=env_dir,
    )

    assert diagnostics.valid_references == {}
    assert diagnostics.items == (
        ModelReferenceSetupItem(
            token="$system.default",
            priority="required",
            status="invalid",
            current_value="",
            summary="Configured reference value is empty.",
            references=("card pack smart", "default_model"),
        ),
        ModelReferenceSetupItem(
            token="$system.fast",
            priority="required",
            status="missing",
            current_value=None,
            summary="Referenced model reference is not configured.",
            references=("agent card helper", "card pack smart"),
        ),
    )
