from __future__ import annotations

import argparse
import json
from pathlib import Path

from _model_picker_common import (
    DEFAULT_VALUE,
    KEEP_VALUE,
    active_provider_names,
    apply_option_overrides,
    build_provider_label,
    build_snapshot,
    find_provider,
    model_capabilities,
    model_options_for_provider,
    web_search_display,
)
from prompt_toolkit.shortcuts import button_dialog, message_dialog, radiolist_dialog


def _abort(message: str) -> int:
    print(message)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prompt Toolkit prototype #1: dialog-based model picker"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to fastagent.config.yaml",
    )
    args = parser.parse_args()

    snapshot = build_snapshot(args.config)

    provider_rows = [
        (option.provider.config_name, build_provider_label(option)) for option in snapshot.providers
    ]
    if not provider_rows:
        return _abort("No providers found in the model catalog.")

    active_names = active_provider_names(snapshot)
    active_text = ", ".join(active_names) if active_names else "none"

    provider_key = radiolist_dialog(
        title="Model picker prototype (dialogs)",
        text=(
            "Select provider\n"
            f"Active providers in your environment: {active_text}\n"
            "\n"
            "Arrow keys move, <space> selects, <enter> confirms."
        ),
        values=provider_rows,
        ok_text="Next",
        cancel_text="Cancel",
    ).run()

    if provider_key is None:
        return _abort("Cancelled at provider selection.")

    provider_option = find_provider(snapshot, provider_key)

    source = button_dialog(
        title="Model source",
        text="Choose the model list scope",
        buttons=[
            ("Curated (default)", "curated"),
            ("All static catalog models", "all"),
            ("Cancel", None),
        ],
    ).run()

    if source is None:
        return _abort("Cancelled at model scope selection.")

    model_rows = model_options_for_provider(snapshot, provider_option.provider, source=source)
    if not model_rows:
        return _abort(
            f"No models found for provider '{provider_option.provider.display_name}' in scope '{source}'."
        )

    selected_model = radiolist_dialog(
        title=f"Model selection · {provider_option.provider.display_name}",
        text="Curated models are shown first. Select one model:",
        values=[(option.spec, option.label) for option in model_rows],
        ok_text="Next",
        cancel_text="Cancel",
    ).run()

    if selected_model is None:
        return _abort("Cancelled at model selection.")

    capabilities = model_capabilities(selected_model)

    reasoning_override: str | None = None
    if capabilities.reasoning_values:
        reasoning_values = [
            (KEEP_VALUE, f"Keep current ({capabilities.current_reasoning})"),
            (DEFAULT_VALUE, "Use model/provider default (remove explicit override)"),
        ]
        reasoning_values.extend((value, value) for value in capabilities.reasoning_values)

        chosen_reasoning = radiolist_dialog(
            title="Reasoning",
            text="Optional reasoning override",
            values=reasoning_values,
            ok_text="Next",
            cancel_text="Cancel",
        ).run()

        if chosen_reasoning is None:
            return _abort("Cancelled at reasoning selection.")
        if chosen_reasoning != KEEP_VALUE:
            reasoning_override = chosen_reasoning

    web_search_override: str | None = None
    if capabilities.web_search_supported:
        current_web_search = web_search_display(capabilities.current_web_search)
        chosen_web_search = radiolist_dialog(
            title="Web search",
            text="Optional web_search override",
            values=[
                (KEEP_VALUE, f"Keep current ({current_web_search})"),
                (DEFAULT_VALUE, "Use provider default (remove explicit override)"),
                ("on", "on"),
                ("off", "off"),
            ],
            ok_text="Finish",
            cancel_text="Cancel",
        ).run()

        if chosen_web_search is None:
            return _abort("Cancelled at web_search selection.")
        if chosen_web_search != KEEP_VALUE:
            web_search_override = chosen_web_search

    resolved_model = apply_option_overrides(
        selected_model,
        reasoning_value=reasoning_override,
        web_search_value=web_search_override,
    )

    payload = {
        "prototype": "dialogs",
        "provider": provider_option.provider.config_name,
        "selected_model": selected_model,
        "resolved_model": resolved_model,
        "reasoning_override": reasoning_override,
        "web_search_override": web_search_override,
    }

    message_dialog(
        title="Selection complete",
        text=(
            f"Provider: {provider_option.provider.display_name}\n"
            f"Model: {selected_model}\n"
            f"Resolved: {resolved_model}\n"
            "\n"
            "JSON payload was printed to stdout."
        ),
    ).run()

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
