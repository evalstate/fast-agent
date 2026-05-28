from pathlib import Path

import fast_agent.core.instruction_source as instruction_source
import fast_agent.io.source_resolver as source_resolver
from fast_agent.core.instruction_source import resolve_instruction_source


def test_resolve_instruction_source_reads_hf_uri(monkeypatch):
    def fake_read_text_source(source: str, *, label: str) -> str:
        assert source == "hf://datasets/evalstate/batch-demo/instructions.md"
        assert label == "instruction"
        return "remote instruction"

    def fake_resolve_instruction(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    monkeypatch.setattr(source_resolver, "read_text_source", fake_read_text_source)
    monkeypatch.setattr(instruction_source, "_resolve_instruction", fake_resolve_instruction)

    assert (
        resolve_instruction_source("hf://datasets/evalstate/batch-demo/instructions.md")
        == "remote instruction"
    )
