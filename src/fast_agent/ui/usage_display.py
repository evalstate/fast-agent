"""
Utility module for displaying usage statistics in a consistent format.
Consolidates the usage display logic that was duplicated between fastagent.py and interactive_prompt.py.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rich.console import Console
from rich.markup import escape as escape_markup

from fast_agent.llm.model_display_name import resolve_llm_display_name
from fast_agent.ui.context_usage_display import normalize_context_usage_percent
from fast_agent.utils.numeric import nonnegative_int_or_none

if TYPE_CHECKING:
    from fast_agent.interfaces import FastAgentLLMProtocol


@dataclass(frozen=True, slots=True)
class _UsageDisplayRow:
    name: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    turns: int
    tool_calls: int
    context_percentage: float | None


@dataclass(frozen=True, slots=True)
class _UsageDisplayData:
    rows: list[_UsageDisplayRow]
    total_input: int
    total_output: int
    total_tokens: int
    total_tool_calls: int


@runtime_checkable
class _NamedAgent(Protocol):
    name: str


@runtime_checkable
class _RegisteredAgentsProvider(Protocol):
    def registered_agents(self) -> Mapping[str, object]: ...


@runtime_checkable
class _SingleAgentProvider(Protocol):
    agent: object


@runtime_checkable
class _UsageAccumulatorSource(Protocol):
    @property
    def context_usage_percentage(self) -> float | None: ...

    def get_summary(self) -> Mapping[str, object]: ...


@runtime_checkable
class _UsageReportAgent(Protocol):
    @property
    def usage_accumulator(self) -> _UsageAccumulatorSource | None: ...

    @property
    def llm(self) -> "FastAgentLLMProtocol | None": ...


def _summary_int(summary: Mapping[str, object], key: str) -> int | None:
    return nonnegative_int_or_none(summary.get(key))


def _truncate_agent_name(agent_name: str, width: int) -> str:
    if len(agent_name) <= width:
        return agent_name
    return f"{agent_name[: width - 3]}..."


def _format_context_percentage(context_percentage: float | None) -> str:
    if context_percentage is None:
        return "-"
    return f"{context_percentage:.1f}%"


def _progress_display_enabled() -> bool:
    try:
        from fast_agent import config

        settings = config.get_settings()
        return bool(settings.logger.progress_display)
    except (ImportError, AttributeError):
        # If we can't check settings, assume we should display.
        return True


def _usage_row(agent_name: str, agent: object) -> _UsageDisplayRow | None:
    if not isinstance(agent, _UsageReportAgent):
        return None
    usage_accumulator = agent.usage_accumulator
    if usage_accumulator is None:
        return None

    summary = usage_accumulator.get_summary()
    turns = _summary_int(summary, "turn_count")
    input_tokens = _summary_int(summary, "cumulative_input_tokens")
    output_tokens = _summary_int(summary, "cumulative_output_tokens")
    billing_tokens = _summary_int(summary, "cumulative_billing_tokens")
    tool_calls = _summary_int(summary, "cumulative_tool_calls")
    if (
        turns is None
        or turns <= 0
        or input_tokens is None
        or output_tokens is None
        or billing_tokens is None
        or tool_calls is None
    ):
        return None

    model = "unknown"
    if agent.llm:
        model = resolve_llm_display_name(agent.llm, max_len=25) or "unknown"

    return _UsageDisplayRow(
        name=agent_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=billing_tokens,
        turns=turns,
        tool_calls=tool_calls,
        context_percentage=normalize_context_usage_percent(
            usage_accumulator.context_usage_percentage
        ),
    )


def _collect_usage_display_data(
    agents: Mapping[str, object],
) -> _UsageDisplayData | None:
    rows: list[_UsageDisplayRow] = []
    total_input = 0
    total_output = 0
    total_tokens = 0
    total_tool_calls = 0

    for agent_name, agent in agents.items():
        row = _usage_row(agent_name, agent)
        if row is None:
            continue

        rows.append(row)
        total_input += row.input_tokens
        total_output += row.output_tokens
        total_tokens += row.total_tokens
        total_tool_calls += row.tool_calls

    if not rows:
        return None

    return _UsageDisplayData(
        rows=rows,
        total_input=total_input,
        total_output=total_output,
        total_tokens=total_tokens,
        total_tool_calls=total_tool_calls,
    )


def _agent_column_width(rows: list[_UsageDisplayRow]) -> int:
    max_agent_width = min(15, max(len(row.name) for row in rows))
    return max(max_agent_width, 5)


def _print_usage_header(console: Console, agent_width: int) -> None:
    console.print()
    console.print("─" * console.size.width, style="dim")
    console.print()
    console.print("[dim]▎[/dim] [bold dim]Usage Summary[/bold dim]")
    console.print()
    console.print(
        f"[dim]{'Agent':<{agent_width}} {'Input':>9} {'Output':>9} {'Total':>9} {'Turns':>6} {'Tools':>6} {'Context%':>9}  {'Model':<25}[/dim]"
    )


def _format_usage_row(row: _UsageDisplayRow, agent_width: int, subdued_colors: bool) -> str:
    agent_name = escape_markup(_truncate_agent_name(row.name, agent_width))
    model = escape_markup(row.model)
    line = (
        f"{agent_name:<{agent_width}} "
        f"{row.input_tokens:>9,} "
        f"{row.output_tokens:>9,} "
        f"[bold]{row.total_tokens:>9,}[/bold] "
        f"{row.turns!s:>6} "
        f"{row.tool_calls!s:>6} "
        f"{_format_context_percentage(row.context_percentage):>9}  "
    )
    if subdued_colors:
        return f"[dim]{line}{model:<25}[/dim]"
    return f"{line}[dim]{model:<25}[/dim]"


def _format_total_row(
    usage_data: _UsageDisplayData, agent_width: int, subdued_colors: bool
) -> str:
    if subdued_colors:
        return (
            f"[bold dim]{'TOTAL':<{agent_width}} "
            f"{usage_data.total_input:>9,} "
            f"{usage_data.total_output:>9,} "
            f"[bold]{usage_data.total_tokens:>9,}[/bold] "
            f"{'':<6} "
            f"{usage_data.total_tool_calls!s:>6} "
            f"{'':<9}  "
            f"{'':<25}[/bold dim]"
        )
    return (
        f"[bold]{'TOTAL':<{agent_width}}[/bold] "
        f"[bold]{usage_data.total_input:>9,}[/bold] "
        f"[bold]{usage_data.total_output:>9,}[/bold] "
        f"[bold]{usage_data.total_tokens:>9,}[/bold] "
        f"{'':<6} "
        f"[bold]{usage_data.total_tool_calls!s:>6}[/bold] "
        f"{'':<9}  "
        f"{'':<25}"
    )


def _print_usage_rows(
    console: Console,
    usage_data: _UsageDisplayData,
    agent_width: int,
    subdued_colors: bool,
) -> None:
    for row in usage_data.rows:
        console.print(_format_usage_row(row, agent_width, subdued_colors))

    if len(usage_data.rows) > 1:
        console.print()
        console.print(_format_total_row(usage_data, agent_width, subdued_colors))

    console.print()


def display_usage_report(
    agents: Mapping[str, object],
    show_if_progress_disabled: bool = False,
    subdued_colors: bool = False,
) -> None:
    """
    Display a formatted table of token usage for all agents.

    Args:
        agents: Dictionary of agent name -> agent object
        show_if_progress_disabled: If True, show even when progress display is disabled
        subdued_colors: If True, use dim styling for a more subdued appearance
    """
    if not show_if_progress_disabled and not _progress_display_enabled():
        return

    usage_data = _collect_usage_display_data(agents)
    if usage_data is None:
        return

    console = Console()
    agent_width = _agent_column_width(usage_data.rows)
    _print_usage_header(console, agent_width)
    _print_usage_rows(console, usage_data, agent_width, subdued_colors)


def collect_agents_from_provider(prompt_provider: object) -> dict[str, object]:
    """
    Collect agents from a prompt provider for usage display.

    Args:
        prompt_provider: Provider that has access to agents

    Returns:
        Dictionary of agent name -> agent object
    """
    if isinstance(prompt_provider, _RegisteredAgentsProvider):
        return dict(prompt_provider.registered_agents())

    if isinstance(prompt_provider, _SingleAgentProvider):
        # Single agent
        agent = prompt_provider.agent
        if isinstance(agent, _NamedAgent):
            return {agent.name: agent}

    return {}
