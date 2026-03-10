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
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text


def _normalize(value: str) -> str:
    return value.strip().lower()


def _prompt_choice(
    session: PromptSession[str],
    *,
    message: str,
    options: dict[str, str],
    default: str,
) -> str | None:
    key_display = sorted(options)
    completer = FuzzyWordCompleter(key_display, WORD=True)
    normalized = {_normalize(key): value for key, value in options.items()}

    while True:
        try:
            entered = session.prompt(
                message,
                completer=completer,
                complete_while_typing=True,
                default=default,
            )
        except (EOFError, KeyboardInterrupt):
            return None

        cleaned = _normalize(entered)
        if not cleaned:
            cleaned = _normalize(default)

        resolved = normalized.get(cleaned)
        if resolved is not None:
            return resolved

        print_formatted_text(HTML("<ansired>Invalid choice. Try one of the suggested values.</ansired>"))


def _print_provider_summary(active_text: str, provider_lines: list[str]) -> None:
    print_formatted_text(HTML("<b>Model picker prototype (prompt + fuzzy completion)</b>"))
    print_formatted_text(HTML(f"Active providers: <b>{active_text}</b>"))
    for line in provider_lines:
        print_formatted_text(f"  {line}")
    print_formatted_text()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prompt Toolkit prototype #2: fuzzy prompt workflow"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to fastagent.config.yaml",
    )
    args = parser.parse_args()

    snapshot = build_snapshot(args.config)
    session: PromptSession[str] = PromptSession()

    provider_options = {option.provider.config_name: option.provider.config_name for option in snapshot.providers}
    if not provider_options:
        print("No providers found in model catalog.")
        return 1

    provider_lines = [build_provider_label(option) for option in snapshot.providers]
    active_names = active_provider_names(snapshot)
    active_text = ", ".join(active_names) if active_names else "none"
    _print_provider_summary(active_text, provider_lines)

    default_provider_name = next(
        (option.provider.config_name for option in snapshot.providers if option.active),
        snapshot.providers[0].provider.config_name,
    )

    provider_name = _prompt_choice(
        session,
        message="Provider> ",
        options=provider_options,
        default=default_provider_name,
    )
    if provider_name is None:
        print("Cancelled.")
        return 1

    provider_option = find_provider(snapshot, provider_name)

    source = _prompt_choice(
        session,
        message="Scope [curated/all]> ",
        options={
            "curated": "curated",
            "all": "all",
            "c": "curated",
            "a": "all",
        },
        default="curated",
    )
    if source is None:
        print("Cancelled.")
        return 1

    model_options = model_options_for_provider(snapshot, provider_option.provider, source=source)
    if not model_options:
        print(f"No models found for provider '{provider_option.provider.display_name}'.")
        return 1

    print_formatted_text(HTML("<u>Model choices</u>"))
    for index, option in enumerate(model_options[:30], start=1):
        print_formatted_text(f"  {index:>2}. {option.label}")
    if len(model_options) > 30:
        print_formatted_text(f"  ... ({len(model_options)} total)")
    print_formatted_text()

    model_lookup: dict[str, str] = {}
    for index, option in enumerate(model_options, start=1):
        model_lookup[str(index)] = option.spec
        model_lookup[option.spec] = option.spec
        if option.alias:
            model_lookup[option.alias] = option.spec

    model_default = model_options[0].spec
    selected_model = _prompt_choice(
        session,
        message="Model (index, alias, or full spec)> ",
        options=model_lookup,
        default=model_default,
    )
    if selected_model is None:
        print("Cancelled.")
        return 1

    capabilities = model_capabilities(selected_model)

    reasoning_override: str | None = None
    if capabilities.reasoning_values:
        reasoning_options = {
            "keep": KEEP_VALUE,
            "default": DEFAULT_VALUE,
            **{value: value for value in capabilities.reasoning_values},
        }
        chosen_reasoning = _prompt_choice(
            session,
            message=f"Reasoning [{'/'.join(reasoning_options)}]> ",
            options=reasoning_options,
            default="keep",
        )
        if chosen_reasoning is None:
            print("Cancelled.")
            return 1
        if chosen_reasoning != KEEP_VALUE:
            reasoning_override = chosen_reasoning

    web_search_override: str | None = None
    if capabilities.web_search_supported:
        current = web_search_display(capabilities.current_web_search)
        print_formatted_text(f"Current web_search: {current}")
        chosen_web_search = _prompt_choice(
            session,
            message="Web search [keep/default/on/off]> ",
            options={
                "keep": KEEP_VALUE,
                "default": DEFAULT_VALUE,
                "on": "on",
                "off": "off",
            },
            default="keep",
        )
        if chosen_web_search is None:
            print("Cancelled.")
            return 1
        if chosen_web_search != KEEP_VALUE:
            web_search_override = chosen_web_search

    resolved_model = apply_option_overrides(
        selected_model,
        reasoning_value=reasoning_override,
        web_search_value=web_search_override,
    )

    payload = {
        "prototype": "prompt",
        "provider": provider_option.provider.config_name,
        "selected_model": selected_model,
        "resolved_model": resolved_model,
        "reasoning_override": reasoning_override,
        "web_search_override": web_search_override,
    }

    print_formatted_text(HTML("<b>Final selection</b>"))
    print_formatted_text(f"Resolved model: {resolved_model}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
