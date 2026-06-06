"""Markdown labels for slash-command metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fast_agent.commands.summary_utils import optional_string
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.markdown import markdown_code_span


def metadata_argument_label(argument: Mapping[str, object]) -> str | None:
    name = optional_string(argument.get("name"))
    if name is None:
        return None

    value_name = optional_string(argument.get("value_name"))
    if value_name is not None:
        return f"{markdown_code_span(name)} ({markdown_code_span(value_name)})"
    return markdown_code_span(name)


def metadata_option_label(option: Mapping[str, object]) -> str | None:
    name = optional_string(option.get("name"))
    if name is None:
        return None

    labels = [markdown_code_span(name)]
    value_name = optional_string(option.get("value_name"))
    if value_name is not None:
        labels[0] = markdown_code_span(f"{name} {value_name}")

    aliases = option.get("aliases")
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
        for alias in aliases:
            alias_label = optional_string(alias)
            if alias_label is not None:
                labels.append(markdown_code_span(alias_label))

    return ", ".join(unique_preserve_order(labels, key=str.casefold))
