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
    from fast_agent.session.subagent_runs import HcChildUsageRollup


@dataclass(frozen=True, slots=True)
class UsageDisplayRow:
    name: str
    model: str
    input_tokens: int | None
    cache_read_tokens: int | None
    output_tokens: int | None
    provider_attempts: int
    tool_calls: int
    context_percentage: float | None
    cost_usd: float | None = None


_UsageDisplayRow = UsageDisplayRow


@dataclass(frozen=True, slots=True)
class _UsageDisplayData:
    rows: list[UsageDisplayRow]
    total_input: int | None
    total_cache_read: int | None
    total_output: int | None
    total_tool_calls: int
    total_cost_usd: float | None
    has_any_cost: bool


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


def _format_token_count(tokens: int | None) -> str:
    if tokens is None:
        return "-"
    return format_compact_count(tokens, significant_digits=4)


def _format_cache_percentage(cache_read_tokens: int | None, input_tokens: int | None) -> str:
    if cache_read_tokens is None or input_tokens is None or input_tokens <= 0:
        return "-"
    percentage = cache_read_tokens / input_tokens * 100
    if percentage < 100 and round(percentage) == 100:
        return ">99%"
    return f"{percentage:.0f}%"


def _format_cost_usd(cost_usd: float | None) -> str:
    if cost_usd is None:
        return "-"
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:.2f}"


def _accumulator_cost_usd(usage_accumulator: _UsageAccumulatorSource) -> float | None:
    turns = usage_accumulator.turns
    if not turns:
        return None
    total = 0.0
    for turn in turns:
        if turn.cost_usd is None:
            return None
        total += turn.cost_usd
    return total


def usage_rows_from_hc_rollups(
    rollups: Sequence[HcChildUsageRollup],
) -> tuple[UsageDisplayRow, ...]:
    """Convert persisted ``hc`` child usage rollups into pure display rows."""
    return tuple(
        UsageDisplayRow(
            name=(
                f"{rollup.parent_agent_name} › hc"
                if rollup.child_count == 1
                else f"{rollup.parent_agent_name} › hc ({rollup.child_count})"
            ),
            model=rollup.model,
            input_tokens=rollup.input_tokens,
            cache_read_tokens=rollup.cache_read_tokens,
            output_tokens=rollup.output_tokens,
            provider_attempts=rollup.provider_attempts,
            tool_calls=rollup.tool_calls,
            context_percentage=None,
            cost_usd=rollup.cost_usd,
        )
        for rollup in rollups
    )


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
    cost_usd = _accumulator_cost_usd(usage_accumulator)
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
        child_cost = _accumulator_cost_usd(subtract)
        if cost_usd is not None and child_cost is not None:
            cost_usd = max(0.0, cost_usd - child_cost)
        else:
            cost_usd = None
        provider_attempts -= child.provider_attempts
        tool_calls -= child.tool_calls

    if provider_attempts <= 0:
        return None

    model = "unknown"
    if agent.llm:
        model = resolve_llm_display_name(agent.llm, max_len=25) or "unknown"

    return UsageDisplayRow(
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
        cost_usd=cost_usd,
    )


def _subagent_usage_row(
    agent_name: str,
    usage_accumulator: _UsageAccumulatorSource,
) -> UsageDisplayRow | None:
    summary = usage_accumulator.summary
    input_tokens = summary.prompt.total
    output_tokens = summary.completion.total
    if summary.provider_attempts <= 0 or input_tokens is None or output_tokens is None:
        return None

    turns = usage_accumulator.turns
    models = {turn.model for turn in turns}
    model = next(iter(models)) if len(models) == 1 else "multiple"
    return UsageDisplayRow(
        name=f"{agent_name} › subagents",
        model=model,
        input_tokens=input_tokens,
        cache_read_tokens=summary.prompt.cache_read,
        output_tokens=output_tokens,
        provider_attempts=summary.provider_attempts,
        tool_calls=summary.tool_calls,
        context_percentage=None,
        cost_usd=_accumulator_cost_usd(usage_accumulator),
    )


def _collect_usage_display_data(
    agents: Mapping[str, object],
    *,
    extra_rows: Sequence[UsageDisplayRow] = (),
) -> _UsageDisplayData | None:
    rows: list[UsageDisplayRow] = []
    total_input = 0
    input_complete = True
    total_cache_read = 0
    cache_read_complete = True
    total_output = 0
    output_complete = True
    total_tool_calls = 0
    total_cost = 0.0
    cost_complete = True
    has_any_cost = False

    def _append_row(row: UsageDisplayRow | None) -> None:
        nonlocal total_input, input_complete, total_cache_read, cache_read_complete
        nonlocal \
            total_output, \
            output_complete, \
            total_tool_calls, \
            total_cost, \
            cost_complete, \
            has_any_cost
        if row is None:
            return
        rows.append(row)
        if row.input_tokens is None:
            input_complete = False
        else:
            total_input += row.input_tokens
        if row.output_tokens is None:
            output_complete = False
        else:
            total_output += row.output_tokens
        total_tool_calls += row.tool_calls
        if row.cache_read_tokens is None:
            cache_read_complete = False
        else:
            total_cache_read += row.cache_read_tokens
        if row.cost_usd is None:
            cost_complete = False
        else:
            has_any_cost = True
            total_cost += row.cost_usd

    for agent_name, agent in agents.items():
        if isinstance(agent, UsageDisplayRow):
            _append_row(agent)
            continue
        subagent_usage = (
            agent.subagent_usage_accumulator
            if isinstance(agent, _SubagentUsageReportAgent)
            else None
        )
        _append_row(_usage_row(agent_name, agent, subtract=subagent_usage))
        if subagent_usage is not None:
            _append_row(_subagent_usage_row(agent_name, subagent_usage))

    for extra_row in extra_rows:
        _append_row(extra_row)

    if not rows:
        return None

    return _UsageDisplayData(
        rows=rows,
        total_input=total_input if input_complete else None,
        total_cache_read=total_cache_read if cache_read_complete else None,
        total_output=total_output if output_complete else None,
        total_tool_calls=total_tool_calls,
        total_cost_usd=total_cost if cost_complete else None,
        has_any_cost=has_any_cost,
    )


def _print_usage_header(console: Console) -> None:
    console.print()
    console.print("─" * console.size.width, style="dim")
    console.print()
    console.print("[dim]▎[/dim] [bold dim]Usage Summary[/bold dim]")
    console.print()


def _usage_cells(row: UsageDisplayRow, *, include_cost: bool = False) -> tuple[Text, ...]:
    cells = [
        Text(row.name),
        Text(_format_token_count(row.input_tokens), style="blue"),
        Text(_format_cache_percentage(row.cache_read_tokens, row.input_tokens), style="blue"),
        Text(_format_token_count(row.output_tokens), style="green"),
        Text(str(row.tool_calls), style="blue"),
        Text(_format_context_percentage(row.context_percentage), style="blue"),
    ]
    if include_cost:
        cells.append(Text(_format_cost_usd(row.cost_usd), style="yellow"))
    cells.append(Text(row.model, style="dim"))
    return tuple(cells)


def _total_cells(usage_data: _UsageDisplayData) -> tuple[Text, ...]:
    cells = [
        Text("TOTAL", style="bold"),
        Text(_format_token_count(usage_data.total_input), style="bold blue"),
        Text(
            _format_cache_percentage(usage_data.total_cache_read, usage_data.total_input),
            style="bold blue",
        ),
        Text(_format_token_count(usage_data.total_output), style="bold green"),
        Text(str(usage_data.total_tool_calls), style="bold blue"),
        Text(),
    ]
    if usage_data.has_any_cost:
        cells.append(
            Text(
                _format_cost_usd(usage_data.total_cost_usd)
                if usage_data.total_cost_usd is not None
                else "",
                style="bold yellow",
            )
        )
    cells.append(Text())
    return tuple(cells)


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
    if usage_data.has_any_cost:
        table.add_column("Cost", justify="right", overflow="ellipsis", no_wrap=True)
    table.add_column("Model", ratio=2, min_width=4, max_width=25, overflow="ellipsis", no_wrap=True)

    row_style = "dim" if subdued_colors else None
    for row in usage_data.rows:
        table.add_row(*_usage_cells(row, include_cost=usage_data.has_any_cost), style=row_style)

    if len(usage_data.rows) > 1:
        table.add_section()
        table.add_row(*_total_cells(usage_data), style="bold dim" if subdued_colors else "bold")
    return table


def _markdown_table_row(cells: tuple[Text, ...]) -> str:
    values = (cell.plain.replace("|", "\\|").replace("\n", " ") for cell in cells)
    return "| " + " | ".join(values) + " |"


def format_usage_markdown(
    agents: Mapping[str, object],
    *,
    extra_rows: Sequence[UsageDisplayRow] = (),
) -> str:
    """Render model-visible usage data without terminal styling."""
    usage_data = _collect_usage_display_data(agents, extra_rows=extra_rows)
    if usage_data is None:
        return "No usage data available."

    if usage_data.has_any_cost:
        lines = [
            "| Agent | Input | Cache hit | Output | Tool calls | Last context | Cost | Model |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "| Agent | Input | Cache hit | Output | Tool calls | Last context | Model |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    for row in usage_data.rows:
        lines.append(_markdown_table_row(_usage_cells(row, include_cost=usage_data.has_any_cost)))
    if len(usage_data.rows) > 1:
        lines.append(_markdown_table_row(_total_cells(usage_data)))
    return "\n".join(lines)


def display_usage_report(
    agents: Mapping[str, object],
    show_if_progress_disabled: bool = False,
    subdued_colors: bool = False,
    *,
    extra_rows: Sequence[UsageDisplayRow] = (),
) -> None:
    """
    Display a formatted table of token usage for all agents.

    Args:
        agents: Dictionary of agent name -> agent object
        show_if_progress_disabled: If True, show even when progress display is disabled
        subdued_colors: If True, use dim styling for a more subdued appearance
        extra_rows: Additional pre-aggregated rows (e.g. persisted ``hc`` children)
    """
    if not show_if_progress_disabled and not _progress_display_enabled():
        return

    usage_data = _collect_usage_display_data(agents, extra_rows=extra_rows)
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
    extra_rows: Sequence[UsageDisplayRow] = (),
) -> None:
    """Stop transient progress and optionally render the final usage report."""
    from fast_agent.ui.progress_display import progress_display

    progress_display.stop()
    if show:
        display_usage_report(
            agents,
            show_if_progress_disabled=True,
            subdued_colors=True,
            extra_rows=extra_rows,
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
