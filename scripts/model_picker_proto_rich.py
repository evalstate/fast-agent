from __future__ import annotations

import argparse
import json
from pathlib import Path

from _model_picker_common import (
    DEFAULT_VALUE,
    active_provider_names,
    apply_option_overrides,
    build_snapshot,
    model_capabilities,
    model_options_for_provider,
    web_search_display,
)
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

console = Console()


def _select_provider_index(total: int, default_index: int) -> int:
    while True:
        value = IntPrompt.ask("Provider number", default=default_index)
        if 1 <= value <= total:
            return value
        console.print(f"[red]Please enter a number between 1 and {total}.[/red]")


def _select_model_index(total: int) -> int:
    while True:
        value = IntPrompt.ask("Model number", default=1)
        if 1 <= value <= total:
            return value
        console.print(f"[red]Please enter a number between 1 and {total}.[/red]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rich prototype #3: table-driven model picker")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to fastagent.config.yaml",
    )
    args = parser.parse_args()

    snapshot = build_snapshot(args.config)

    if not snapshot.providers:
        console.print("[red]No providers found in model catalog.[/red]")
        return 1

    active_names = active_provider_names(snapshot)
    active_text = ", ".join(active_names) if active_names else "none"
    console.print(Panel.fit(f"[bold]Model picker prototype (rich)[/bold]\nActive providers: {active_text}"))

    provider_table = Table(title="Providers")
    provider_table.add_column("#", justify="right")
    provider_table.add_column("Provider")
    provider_table.add_column("Status")
    provider_table.add_column("Curated", justify="right")

    for index, option in enumerate(snapshot.providers, start=1):
        provider_table.add_row(
            str(index),
            option.provider.display_name,
            "[green]active[/green]" if option.active else "[dim]inactive[/dim]",
            str(len(option.curated_entries)),
        )

    console.print(provider_table)

    default_provider_index = next(
        (index for index, option in enumerate(snapshot.providers, start=1) if option.active),
        1,
    )
    chosen_provider_index = _select_provider_index(
        total=len(snapshot.providers),
        default_index=default_provider_index,
    )
    provider = snapshot.providers[chosen_provider_index - 1].provider

    use_all = Confirm.ask("Show all catalog models?", default=False)
    source = "all" if use_all else "curated"

    model_options = model_options_for_provider(snapshot, provider, source=source)
    if not model_options:
        console.print(f"[red]No models found for {provider.display_name}.[/red]")
        return 1

    model_table = Table(title=f"Models · {provider.display_name}")
    model_table.add_column("#", justify="right")
    model_table.add_column("Model")

    for index, option in enumerate(model_options, start=1):
        model_table.add_row(str(index), option.label)

    console.print(model_table)
    chosen_model_index = _select_model_index(total=len(model_options))
    selected_model = model_options[chosen_model_index - 1].spec

    capabilities = model_capabilities(selected_model)

    reasoning_override: str | None = None
    if capabilities.reasoning_values:
        reasoning_choices = ["keep", "default", *capabilities.reasoning_values]
        reasoning_text = Prompt.ask(
            f"Reasoning ({'/'.join(reasoning_choices)})",
            default="keep",
            choices=reasoning_choices,
        )
        if reasoning_text == "default":
            reasoning_override = DEFAULT_VALUE
        elif reasoning_text != "keep":
            reasoning_override = reasoning_text

    web_search_override: str | None = None
    if capabilities.web_search_supported:
        current = web_search_display(capabilities.current_web_search)
        web_search_choice = Prompt.ask(
            f"Web search (keep/default/on/off) [current: {current}]",
            default="keep",
            choices=["keep", "default", "on", "off"],
        )
        if web_search_choice == "default":
            web_search_override = DEFAULT_VALUE
        elif web_search_choice != "keep":
            web_search_override = web_search_choice

    resolved_model = apply_option_overrides(
        selected_model,
        reasoning_value=reasoning_override,
        web_search_value=web_search_override,
    )

    payload = {
        "prototype": "rich",
        "provider": provider.config_name,
        "selected_model": selected_model,
        "resolved_model": resolved_model,
        "reasoning_override": reasoning_override,
        "web_search_override": web_search_override,
    }

    console.print(Panel.fit(f"[bold green]Resolved model[/bold green]\n{resolved_model}"))
    console.print_json(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
