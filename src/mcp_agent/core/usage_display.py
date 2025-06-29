"""
Utility module for displaying usage statistics in a consistent format.
Consolidates the usage display logic that was duplicated between fastagent.py and interactive_prompt.py.
"""

from typing import Any, Dict, Optional

from rich.console import Console


def display_usage_report(
    agents: Dict[str, Any], show_if_progress_disabled: bool = False, subdued_colors: bool = False
) -> None:
    """
    Display a formatted table of token usage for all agents.

    Args:
        agents: Dictionary of agent name -> agent object
        show_if_progress_disabled: If True, show even when progress display is disabled
        subdued_colors: If True, use dim styling for a more subdued appearance
    """
    # Check if progress display is enabled (only relevant for fastagent context)
    if not show_if_progress_disabled:
        try:
            from mcp_agent import config

            settings = config.get_settings()
            if not settings.logger.progress_display:
                return
        except (ImportError, AttributeError):
            # If we can't check settings, assume we should display
            pass

    # Collect usage data from all agents
    usage_data = []
    total_input = 0
    total_output = 0
    total_tokens = 0

    for agent_name, agent in agents.items():
        if agent.usage_accumulator:
            summary = agent.usage_accumulator.get_summary()
            if summary["turn_count"] > 0:
                input_tokens = summary["cumulative_input_tokens"]
                output_tokens = summary["cumulative_output_tokens"]
                billing_tokens = summary["cumulative_billing_tokens"]
                turns = summary["turn_count"]

                # Get context percentage for this agent
                context_percentage = agent.usage_accumulator.context_usage_percentage

                # Get model name from LLM's default_request_params
                model = "unknown"
                if hasattr(agent, "_llm") and agent._llm:
                    llm = agent._llm
                    if (
                        hasattr(llm, "default_request_params")
                        and llm.default_request_params
                        and hasattr(llm.default_request_params, "model")
                    ):
                        model = llm.default_request_params.model or "unknown"

                # Standardize model name truncation - use consistent 25 char width with 22+... truncation
                if len(model) > 25:
                    model = model[:22] + "..."

                usage_data.append(
                    {
                        "name": agent_name,
                        "model": model,
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": billing_tokens,
                        "turns": turns,
                        "context": context_percentage,
                    }
                )

                total_input += input_tokens
                total_output += output_tokens
                total_tokens += billing_tokens

    if not usage_data:
        return

    # Calculate dynamic agent column width (max 15)
    max_agent_width = min(15, max(len(data["name"]) for data in usage_data) if usage_data else 8)
    agent_width = max(max_agent_width, 5)  # Minimum of 5 for "Agent" header

    # Display the table
    console = Console()
    console.print()
    console.print("[dim]Usage Summary (Cumulative)[/dim]")

    # Print header with proper spacing
    console.print(
        f"[dim]{'Agent':<{agent_width}} {'Input':>9} {'Output':>9} {'Total':>9} {'Turns':>6} {'Context%':>9}  {'Model':<25}[/dim]"
    )

    # Print agent rows - use styling based on subdued_colors flag
    for data in usage_data:
        input_str = f"{data['input']:,}"
        output_str = f"{data['output']:,}"
        total_str = f"{data['total']:,}"
        turns_str = str(data["turns"])
        context_str = f"{data['context']:.1f}%" if data["context"] is not None else "-"

        # Truncate agent name if needed
        agent_name = data["name"]
        if len(agent_name) > agent_width:
            agent_name = agent_name[: agent_width - 3] + "..."

        if subdued_colors:
            # Original fastagent.py style with dim wrapper
            console.print(
                f"[dim]{agent_name:<{agent_width}} "
                f"{input_str:>9} "
                f"{output_str:>9} "
                f"[bold]{total_str:>9}[/bold] "
                f"{turns_str:>6} "
                f"{context_str:>9}  "
                f"{data['model']:<25}[/dim]"
            )
        else:
            # Original interactive_prompt.py style
            console.print(
                f"{agent_name:<{agent_width}} "
                f"{input_str:>9} "
                f"{output_str:>9} "
                f"[bold]{total_str:>9}[/bold] "
                f"{turns_str:>6} "
                f"{context_str:>9}  "
                f"[dim]{data['model']:<25}[/dim]"
            )

    # Add total row if multiple agents
    if len(usage_data) > 1:
        console.print()
        total_input_str = f"{total_input:,}"
        total_output_str = f"{total_output:,}"
        total_tokens_str = f"{total_tokens:,}"

        if subdued_colors:
            # Original fastagent.py style with dim wrapper on bold
            console.print(
                f"[bold dim]{'TOTAL':<{agent_width}} "
                f"{total_input_str:>9} "
                f"{total_output_str:>9} "
                f"[bold]{total_tokens_str:>9}[/bold] "
                f"{'':<6} "
                f"{'':<9}  "
                f"{'':<25}[/bold dim]"
            )
        else:
            # Original interactive_prompt.py style
            console.print(
                f"[bold]{'TOTAL':<{agent_width}}[/bold] "
                f"[bold]{total_input_str:>9}[/bold] "
                f"[bold]{total_output_str:>9}[/bold] "
                f"[bold]{total_tokens_str:>9}[/bold] "
                f"{'':<6} "
                f"{'':<9}  "
                f"{'':<25}"
            )

    console.print()


def collect_agents_from_provider(
    prompt_provider: Any, agent_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collect agents from a prompt provider for usage display.

    Args:
        prompt_provider: Provider that has access to agents
        agent_name: Name of the current agent (for context)

    Returns:
        Dictionary of agent name -> agent object
    """
    agents_to_show = {}

    if hasattr(prompt_provider, "_agents"):
        # Multi-agent app - show all agents
        agents_to_show = prompt_provider._agents
    elif hasattr(prompt_provider, "agent"):
        # Single agent
        agent = prompt_provider.agent
        if hasattr(agent, "name"):
            agents_to_show = {agent.name: agent}

    return agents_to_show
