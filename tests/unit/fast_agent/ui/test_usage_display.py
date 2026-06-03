from dataclasses import dataclass
from typing import Any

from fast_agent.ui.usage_display import (
    _format_usage_row,
    _UsageDisplayRow,
    collect_agents_from_provider,
    display_usage_report,
)


@dataclass
class _Agent:
    name: str


class _RegisteredProvider:
    def __init__(self, agents: dict[str, object]) -> None:
        self._agents = agents

    def registered_agents(self) -> dict[str, object]:
        return self._agents


@dataclass
class _SingleAgentProvider:
    agent: object


@dataclass
class _UsageAccumulator:
    context_usage_percentage: Any = None
    summary: dict[str, object] | None = None

    def get_summary(self) -> dict[str, object]:
        if self.summary is not None:
            return self.summary
        return {
            "turn_count": 2,
            "cumulative_input_tokens": 1000,
            "cumulative_output_tokens": 250,
            "cumulative_billing_tokens": 1250,
            "cumulative_tool_calls": 3,
        }


@dataclass
class _UsageAgent:
    usage_accumulator: _UsageAccumulator
    llm: object | None = None


def test_collect_agents_from_registered_provider() -> None:
    agent = _Agent(name="alpha")

    assert collect_agents_from_provider(_RegisteredProvider({"alpha": agent})) == {
        "alpha": agent
    }


def test_collect_agents_from_single_agent_provider() -> None:
    agent = _Agent(name="solo")

    assert collect_agents_from_provider(_SingleAgentProvider(agent=agent)) == {"solo": agent}


def test_collect_agents_ignores_single_agent_without_name() -> None:
    assert collect_agents_from_provider(_SingleAgentProvider(agent=object())) == {}


def test_display_usage_report_renders_usage_rows(capsys) -> None:
    display_usage_report(
        {"alpha": _UsageAgent(usage_accumulator=_UsageAccumulator(42.4))},
        show_if_progress_disabled=True,
    )

    output = capsys.readouterr().out
    assert "Usage Summary" in output
    assert "alpha" in output
    assert "1,000" in output
    assert "1,250" in output
    assert "42.4%" in output


def test_display_usage_report_omits_invalid_context_percentages(capsys) -> None:
    display_usage_report(
        {
            "flag": _UsageAgent(usage_accumulator=_UsageAccumulator(True)),
            "nan": _UsageAgent(usage_accumulator=_UsageAccumulator(float("nan"))),
        },
        show_if_progress_disabled=True,
    )

    output = capsys.readouterr().out
    assert "flag" in output
    assert "nan" in output
    assert "True%" not in output
    assert "nan%" not in output


def test_display_usage_report_omits_negative_summary_values(capsys) -> None:
    display_usage_report(
        {
            "negative": _UsageAgent(
                usage_accumulator=_UsageAccumulator(
                    summary={
                        "turn_count": 1,
                        "cumulative_input_tokens": -1,
                        "cumulative_output_tokens": 2,
                        "cumulative_billing_tokens": 3,
                        "cumulative_tool_calls": 4,
                    }
                )
            )
        },
        show_if_progress_disabled=True,
    )

    output = capsys.readouterr().out
    assert output == ""


def test_format_usage_row_escapes_rich_markup_in_agent_and_model_names() -> None:
    row = _UsageDisplayRow(
        name="[red]agent[/red]",
        model="model_[draft]",
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        turns=1,
        tool_calls=0,
        context_percentage=None,
    )

    rendered = _format_usage_row(row, agent_width=20, subdued_colors=False)

    assert "\\[red]agent\\[/red]" in rendered
    assert "model_\\[draft]" in rendered
