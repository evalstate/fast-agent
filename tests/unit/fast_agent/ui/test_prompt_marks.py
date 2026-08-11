from __future__ import annotations

from fast_agent.ui import prompt_marks


def test_herdr_reporting_precedes_osc_suppression(monkeypatch) -> None:
    reported: list[str] = []
    monkeypatch.setattr(prompt_marks, "report_prompt_mark", reported.append)
    monkeypatch.setattr(prompt_marks, "prompt_mark_sequence", lambda *args, **kwargs: "")

    prompt_marks.emit_prompt_mark("C")

    assert reported == ["C"]
