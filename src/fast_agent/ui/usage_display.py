"""
Utility module for displaying usage statistics in a consistent format.
Consolidates the usage display logic that was duplicated between fastagent.py and interactive_prompt.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rich.table import Table
from rich.text import Text

from fast_agent.llm.model_display_name import resolve_llm_display_name
from fast_agent.ui.console import SurrogateSafeConsole
from fast_agent.ui.context_usage_display import normalize_context_usage_percent
from fast_agent.utils.count_display import format_compact_count

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rich.console import Console

    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.usage_tracking import TurnUsage, UsageSummary


@dataclass(frozen=True, slots=True)
class _UsageDisplayRow:
    name: str
    model: str
    input_tokens: int
    cache_read_tokens: int | None
    output_tokens: int
    provider_attempts: int
    tool_calls: int
    context_percentage: float | None


@dataclass(frozen=True, slots=True)
class _UsageDisplayData:
    rows: list[_UsageDisplayRow]
    total_input: int
    total_cache_read: int | None
    total_output: int
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

    @property
    def summary(self) -> UsageSummary: ...

    @property
    def turns(self) -> Sequence[TurnUsage]: ...


@runtime_checkable
class _UsageReportAgent(Protocol):
    @property
    def usage_accumulator(self) -> _UsageAccumulatorSource | None: ...

    @property
    def llm(self) -> "FastAgentLLMProtocol | None": ...


@runtime_checkable
class _SubagentUsageReportAgent(Protocol):
    @property
    def subagent_usage_accumulator(self) -> _UsageAccumulatorSource: ...


def _format_context_percentage(context_percentage: float | None) -> str:
    if context_percentage is None:
        return "-"
    return f"{context_percentage:.1f}%"


def _format_cache_percentage(cache_read_tokens: int | None, input_tokens: int) -> str:
    if cache_read_tokens is None or input_tokens <= 0:
        return "-"
    percentage = cache_read_tokens / input_tokens * 100
    if percentage < 100 and round(percentage) == 100:
        return ">99%"
    return f"{percentage:.0f}%"


def _progress_display_enabled() -> bool:
    try:
        from fast_agent import config

        settings = config.get_settings()
        return bool(settings.logger.progress_display)
    except (ImportError, AttributeError):
        # If we can't check settings, assume we should display.
        return True


def _usage_row(
    agent_name: str,
    agent: object,
    *,
    subtract: _UsageAccumulatorSource | None = None,
) -> _UsageDisplayRow | None:
    if not isinstance(agent, _UsageReportAgent):
        return None
    usage_accumulator = agent.usage_accumulator
    if usage_accumulator is None:
        return None

    summary = usage_accumulator.summary
    input_tokens = summary.prompt.total
    output_tokens = summary.completion.total
    cache_read_tokens = summary.prompt.cache_read
    provider_attempts = summary.provider_attempts
    tool_calls = summary.tool_calls
    if provider_attempts <= 0 or input_tokens is None or output_tokens is None:
        return None

    if subtract is not None and subtract.summary.provider_attempts > 0:
        child = subtract.summary
        if child.prompt.total is not None:
            input_tokens -= child.prompt.total
        if child.completion.total is not None:
            output_tokens -= child.completion.total
        if cache_read_tokens is not None and child.prompt.cache_read is not None:
            cache_read_tokens -= child.prompt.cache_read
        else:
            cache_read_tokens = None
        provider_attempts -= child.provider_attempts
        tool_calls -= child.tool_calls

    if provider_attempts <= 0:
        return None

    model = "unknown"
    if agent.llm:
        model = resolve_llm_display_name(agent.llm, max_len=25) or "unknown"

    return _UsageDisplayRow(
        name=agent_name,
        model=model,
        input_tokens=input_tokens,
        cache_read_tokens=cache_read_tokens,
        output_tokens=output_tokens,
        provider_attempts=provider_attempts,
        tool_calls=tool_calls,
        context_percentage=normalize_context_usage_percent(
            usage_accumulator.context_usage_percentage
        ),
    )


def _subagent_usage_row(
    agent_name: str,
    usage_accumulator: _UsageAccumulatorSource,
) -> _UsageDisplayRow | None:
    summary = usage_accumulator.summary
    input_tokens = summary.prompt.total
    output_tokens = summary.completion.total
    if summary.provider_attempts <= 0 or input_tokens is None or output_tokens is None:
        return None

    turns = usage_accumulator.turns
    models = {turn.model for turn in turns}
    model = next(iter(models)) if len(models) == 1 else "multiple"
    return _UsageDisplayRow(
        name=f"{agent_name} › subagents",
        model=model,
        input_tokens=input_tokens,
        cache_read_tokens=summary.prompt.cache_read,
        output_tokens=output_tokens,
        provider_attempts=summary.provider_attempts,
        tool_calls=summary.tool_calls,
        context_percentage=None,
    )


def _collect_usage_display_data(
    agents: Mapping[str, object],
) -> _UsageDisplayData | None:
    rows: list[_UsageDisplayRow] = []
    total_input = 0
    total_cache_read = 0
    cache_read_complete = True
    total_output = 0
    total_tool_calls = 0

    for agent_name, agent in agents.items():
        subagent_usage = (
            agent.subagent_usage_accumulator
            if isinstance(agent, _SubagentUsageReportAgent)
            else None
        )
        agent_rows = [
            _usage_row(agent_name, agent, subtract=subagent_usage),
            (
                _subagent_usage_row(agent_name, subagent_usage)
                if subagent_usage is not None
                else None
            ),
        ]
        for row in agent_rows:
            if row is None:
                continue

            rows.append(row)
            total_input += row.input_tokens
            total_output += row.output_tokens
            total_tool_calls += row.tool_calls
            if row.cache_read_tokens is None:
                cache_read_complete = False
            else:
                total_cache_read += row.cache_read_tokens

    if not rows:
        return None

    return _UsageDisplayData(
        rows=rows,
        total_input=total_input,
        total_cache_read=total_cache_read if cache_read_complete else None,
        total_output=total_output,
        total_tool_calls=total_tool_calls,
    )


def _print_usage_header(console: Console) -> None:
    console.print()
    console.print("─" * console.size.width, style="dim")
    console.print()
    console.print("[dim]▎[/dim] [bold dim]Usage Summary[/bold dim]")
    console.print()


def _usage_cells(row: _UsageDisplayRow) -> tuple[Text, ...]:
    return (
        Text(row.name),
        Text(format_compact_count(row.input_tokens, significant_digits=4), style="blue"),
        Text(_format_cache_percentage(row.cache_read_tokens, row.input_tokens), style="blue"),
        Text(format_compact_count(row.output_tokens, significant_digits=4), style="green"),
        Text(str(row.tool_calls), style="blue"),
        Text(_format_context_percentage(row.context_percentage), style="blue"),
        Text(row.model, style="dim"),
    )


def _total_cells(usage_data: _UsageDisplayData) -> tuple[Text, ...]:
    return (
        Text("TOTAL", style="bold"),
        Text(
            format_compact_count(usage_data.total_input, significant_digits=4),
            style="bold blue",
        ),
        Text(
            _format_cache_percentage(usage_data.total_cache_read, usage_data.total_input),
            style="bold blue",
        ),
        Text(
            format_compact_count(usage_data.total_output, significant_digits=4),
            style="bold green",
        ),
        Text(str(usage_data.total_tool_calls), style="bold blue"),
        Text(),
        Text(),
    )


def _usage_table(
    usage_data: _UsageDisplayData,
    *,
    subdued_colors: bool,
) -> Table:
    table = Table(
        box=None,
        collapse_padding=True,
        expand=True,
        pad_edge=False,
        header_style="dim",
    )
    table.add_column("Agent", min_width=5, max_width=24, overflow="ellipsis", no_wrap=True)
    table.add_column("▶ Input", justify="right", overflow="ellipsis", no_wrap=True)
    table.add_column("Cache hit", justify="right", overflow="ellipsis", no_wrap=True)
    table.add_column("◀ Output", justify="right", overflow="ellipsis", no_wrap=True)
    table.add_column("Tool calls", justify="right", overflow="ellipsis", no_wrap=True)
    table.add_column("Last context", justify="right", overflow="ellipsis", no_wrap=True)
    table.add_column("Model", ratio=2, min_width=4, max_width=25, overflow="ellipsis", no_wrap=True)

    row_style = "dim" if subdued_colors else None
    for row in usage_data.rows:
        table.add_row(*_usage_cells(row), style=row_style)

    if len(usage_data.rows) > 1:
        table.add_section()
        table.add_row(*_total_cells(usage_data), style="bold dim" if subdued_colors else "bold")
    return table


def _markdown_table_row(cells: tuple[Text, ...]) -> str:
    values = (cell.plain.replace("|", "\\|").replace("\n", " ") for cell in cells)
    return "| " + " | ".join(values) + " |"


def format_usage_markdown(agents: Mapping[str, object]) -> str:
    """Render model-visible usage data without terminal styling."""
    usage_data = _collect_usage_display_data(agents)
    if usage_data is None:
        return "No usage data available."

    lines = [
        "| Agent | Input | Cache hit | Output | Tool calls | Last context | Model |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in usage_data.rows:
        lines.append(_markdown_table_row(_usage_cells(row)))
    if len(usage_data.rows) > 1:
        lines.append(_markdown_table_row(_total_cells(usage_data)))
    return "\n".join(lines)


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

    usage_console = SurrogateSafeConsole()
    _print_usage_header(usage_console)
    usage_console.print(_usage_table(usage_data, subdued_colors=subdued_colors))
    usage_console.print()


def finalize_usage_report(
    agents: Mapping[str, object],
    *,
    show: bool,
) -> None:
    """Stop transient progress and optionally render the final usage report."""
    from fast_agent.ui.progress_display import progress_display

    progress_display.stop()
    if show:
        display_usage_report(
            agents,
            show_if_progress_disabled=True,
            subdued_colors=True,
        )


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
