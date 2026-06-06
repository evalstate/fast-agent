"""Shared prompt command handlers."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypedDict, cast, runtime_checkable

from rich.text import Text

from fast_agent.commands.handlers._text_formatting import indexed_row
from fast_agent.commands.handlers._text_utils import truncate_description
from fast_agent.commands.handlers.shared import (
    load_prompt_messages_result,
    prompt_selection_after_message,
    replace_agent_history,
)
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.commands.summary_utils import optional_string
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.mcp_aggregator import SEP
from fast_agent.mcp.prompts.prompt_load import prompt_file_template_variables
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.progress_display import progress_display
from fast_agent.utils.count_display import format_count, plural_label
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext


@dataclass(frozen=True, slots=True)
class PromptArguments:
    names: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CollectedPromptArguments:
    values: dict[str, str] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PromptArgumentFields:
    name: object
    required: object
    description: object


@dataclass(frozen=True, slots=True)
class LoadedPromptMessages:
    history_messages: list[PromptMessageExtended]
    buffered_text: str | None


class PromptSummary(TypedDict):
    server: str
    name: str
    namespaced_name: str
    title: object
    description: object
    arg_count: int
    arguments: list[object]
    arg_names: list[str]
    required_args: list[str]
    optional_args: list[str]
    arg_descriptions: dict[str, str]


@runtime_checkable
class _PromptArgumentLike(Protocol):
    @property
    def name(self) -> object: ...

    @property
    def required(self) -> object: ...

    @property
    def description(self) -> object: ...


@runtime_checkable
class _PromptListLike(Protocol):
    @property
    def prompts(self) -> object: ...


@runtime_checkable
class _PromptLike(Protocol):
    @property
    def name(self) -> object: ...

    @property
    def title(self) -> object: ...

    @property
    def description(self) -> object: ...

    @property
    def arguments(self) -> object: ...


@runtime_checkable
class _TurnUsageDisplayProvider(Protocol):
    def _show_turn_usage(self, agent_name: str) -> None: ...


def _format_prompt_args(prompt: PromptSummary) -> str:
    arg_names = prompt.get("arg_names", [])
    required_args = set(prompt.get("required_args", []))
    if arg_names:
        arg_list = [f"{name}*" if name in required_args else name for name in arg_names]
        args_text = ", ".join(arg_list)
        if len(args_text) > 80:
            args_text = args_text[:77] + "..."
        return args_text

    arg_count = prompt.get("arg_count", 0)
    return format_count(arg_count, "parameter")


def _supplied_prompt_argument_value(value: str | None) -> str | None:
    return strip_to_none(value)


def _prompt_argument_fields(argument: object) -> PromptArgumentFields:
    if isinstance(argument, Mapping):
        argument_mapping = cast("Mapping[str, object]", argument)
        return PromptArgumentFields(
            name=argument_mapping.get("name"),
            required=argument_mapping.get("required"),
            description=argument_mapping.get("description"),
        )

    if not isinstance(argument, _PromptArgumentLike):
        return PromptArgumentFields(name=None, required=None, description=None)

    return PromptArgumentFields(
        name=argument.name,
        required=argument.required,
        description=argument.description,
    )


def _prompt_argument_is_required(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return bool(value)


def _extract_prompt_arguments(arguments: object) -> PromptArguments:
    if not isinstance(arguments, list):
        return PromptArguments()

    arg_names: list[str] = []
    required_args: list[str] = []
    optional_args: list[str] = []
    arg_descriptions: dict[str, str] = {}

    for arg in arguments:
        fields = _prompt_argument_fields(arg)

        name = optional_string(fields.name)
        if name is None:
            continue

        arg_names.append(name)

        description = optional_string(fields.description)
        if description is not None:
            arg_descriptions[name] = description

        if _prompt_argument_is_required(fields.required):
            required_args.append(name)
        else:
            optional_args.append(name)

    return PromptArguments(
        names=arg_names,
        required=required_args,
        optional=optional_args,
        descriptions=arg_descriptions,
    )


def _build_prompt_list_text(
    prompts: "Sequence[PromptSummary]",
    *,
    include_usage: bool,
) -> Text:
    content = Text()

    for index, prompt in enumerate(prompts, 1):
        if content.plain:
            content.append("\n")

        line = indexed_row(index)
        line.append(f"{prompt['server']}•", style="dim green")
        line.append(prompt["name"], style="bright_blue bold")

        title = optional_string(prompt.get("title"))
        if title is not None:
            line.append(f" {title}", style="default")

        content.append_text(line)

        description = optional_string(prompt.get("description"))
        if description is not None:
            truncated = truncate_description(description)
            for line_text in textwrap.wrap(truncated, width=72):
                content.append("\n")
                content.append("     ", style="dim")
                content.append(line_text, style="white")

        if prompt.get("arg_count", 0) > 0:
            args_text = _format_prompt_args(prompt)
            content.append("\n")
            content.append("     ", style="dim")
            content.append(f"args: {args_text}", style="dim magenta")

        content.append("\n")

    if include_usage:
        content.append("\n")
        content.append(
            "Usage: /prompt <number> to select by number, or /prompts for interactive selection",
            style="dim",
        )

    return content


def _prompt_matches_name(prompt: PromptSummary, requested_name: str) -> bool:
    return prompt["name"] == requested_name or prompt["namespaced_name"] == requested_name


def _prompt_summary(
    *,
    server_name: str,
    prompt_name: str,
    title: object,
    description: object,
    arguments: "Sequence[object]",
) -> PromptSummary:
    argument_list = list(arguments)
    prompt_args = _extract_prompt_arguments(argument_list)
    return {
        "server": server_name,
        "name": prompt_name,
        "namespaced_name": f"{server_name}{SEP}{prompt_name}",
        "title": title,
        "description": description,
        "arg_count": len(prompt_args.names) if prompt_args.names else len(argument_list),
        "arguments": argument_list,
        "arg_names": prompt_args.names,
        "required_args": prompt_args.required,
        "optional_args": prompt_args.optional,
        "arg_descriptions": prompt_args.descriptions,
    }


def _prompt_summary_from_entry(
    *,
    server_name: object,
    prompt: object,
) -> PromptSummary | None:
    normalized_server_name = optional_string(server_name)
    if normalized_server_name is None:
        return None

    if isinstance(prompt, Mapping):
        prompt_mapping = cast("Mapping[str, object]", prompt)
        prompt_name = optional_string(prompt_mapping.get("name"))
        if prompt_name is None:
            return None
        arguments = prompt_mapping.get("arguments", [])
        prompt_arguments = arguments if isinstance(arguments, list) else []
        return _prompt_summary(
            server_name=normalized_server_name,
            prompt_name=prompt_name,
            title=prompt_mapping.get("title"),
            description=prompt_mapping.get("description", "No description"),
            arguments=prompt_arguments,
        )

    if not isinstance(prompt, _PromptLike):
        return None

    prompt_name = optional_string(prompt.name)
    if prompt_name is None:
        return None

    arguments = prompt.arguments if isinstance(prompt.arguments, list) else []
    description = prompt.description if prompt.description else "No description"
    return _prompt_summary(
        server_name=normalized_server_name,
        prompt_name=prompt_name,
        title=prompt.title or None,
        description=description,
        arguments=arguments,
    )


def _format_prompt_argument_header(
    prompt_name: str,
    *,
    required_count: int,
    optional_count: int,
) -> str:
    parts: list[str] = []
    if required_count:
        parts.append(f"requires {format_count(required_count, 'argument')}")
    if optional_count:
        prefix = "and has" if parts else "has"
        parts.append(f"{prefix} {format_count(optional_count, 'optional argument')}")
    return f"Prompt {prompt_name} {' '.join(parts)}:"


def _format_missing_required_prompt_arguments(missing: list[str]) -> str:
    missing_list = ", ".join(missing)
    argument_label = plural_label(len(missing), "argument")
    return f"Missing required prompt {argument_label}: {missing_list}"


def _selected_prompt_from_selection(
    prompts: "Sequence[PromptSummary]",
    selection: str | None,
    *,
    outcome: CommandOutcome,
    agent_name: str,
) -> PromptSummary | None:
    selected_index = strip_to_none(selection)
    if selected_index is None:
        outcome.add_message(
            "Prompt selection cancelled.",
            channel="warning",
            right_info="prompt selection",
            agent_name=agent_name,
        )
        return None

    try:
        idx = int(selected_index) - 1
    except ValueError:
        outcome.add_message(
            "Invalid input, please enter a number.",
            channel="error",
            right_info="prompt selection",
            agent_name=agent_name,
        )
        return None

    if not (0 <= idx < len(prompts)):
        outcome.add_message(
            "Invalid selection.",
            channel="error",
            right_info="prompt selection",
            agent_name=agent_name,
        )
        return None

    return prompts[idx]


async def _collect_prompt_argument_values(
    ctx: CommandContext,
    *,
    prompt_name: str,
    required_args: list[str],
    optional_args: list[str],
    arg_descriptions: dict[str, str],
    agent_name: str,
    right_info: str,
    fail_on_missing_required: bool,
) -> CollectedPromptArguments:
    arg_values: dict[str, str] = {}
    missing_required: list[str] = []

    if not required_args and not optional_args:
        return CollectedPromptArguments()

    await ctx.io.emit(
        CommandMessage(
            text=Text(
                _format_prompt_argument_header(
                    prompt_name,
                    required_count=len(required_args),
                    optional_count=len(optional_args),
                ),
                style="cyan",
            ),
            right_info=right_info,
            agent_name=agent_name,
        )
    )

    for arg_name in required_args:
        description = arg_descriptions.get(arg_name, "")
        arg_value = await ctx.io.prompt_argument(
            arg_name,
            description=description,
            required=True,
        )
        if supplied_value := _supplied_prompt_argument_value(arg_value):
            arg_values[arg_name] = supplied_value
        elif fail_on_missing_required:
            missing_required.append(arg_name)

    for arg_name in optional_args:
        description = arg_descriptions.get(arg_name, "")
        arg_value = await ctx.io.prompt_argument(
            arg_name,
            description=description,
            required=False,
        )
        if supplied_value := _supplied_prompt_argument_value(arg_value):
            arg_values[arg_name] = supplied_value

    return CollectedPromptArguments(values=arg_values, missing_required=missing_required)


async def _get_all_prompts(
    ctx: CommandContext, agent_name: str | None = None
) -> list[PromptSummary]:
    prompt_servers = await _list_prompt_servers(ctx, agent_name=agent_name)
    if not isinstance(prompt_servers, Mapping):
        return []

    all_prompts: list[PromptSummary] = []

    for server_name, prompts_info in prompt_servers.items():
        prompts_list: list[object] | None = None
        if isinstance(prompts_info, list):
            prompts_list = list(prompts_info)
        else:
            if isinstance(prompts_info, _PromptListLike) and isinstance(
                prompts_info.prompts,
                list,
            ):
                prompts_list = list(prompts_info.prompts)

        if not prompts_list:
            continue

        for prompt in prompts_list:
            summary = _prompt_summary_from_entry(server_name=server_name, prompt=prompt)
            if summary is not None:
                all_prompts.append(summary)

    all_prompts.sort(key=lambda p: (p["server"], p["name"]))

    return all_prompts


async def _list_prompt_servers(
    ctx: CommandContext, *, agent_name: str | None = None
) -> object:
    with suppress(Exception):
        return await ctx.agent_provider.list_prompts(namespace=None, agent_name=agent_name)
    return {}


async def handle_list_prompts(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    all_prompts = await _get_all_prompts(ctx, agent_name)
    if not all_prompts:
        outcome.add_message(
            "No prompts available for this agent.",
            channel="warning",
            right_info="prompt list",
            agent_name=agent_name,
        )
        return outcome

    content = _build_prompt_list_text(all_prompts, include_usage=True)
    outcome.add_message(
        content,
        right_info="prompt list",
        agent_name=agent_name,
    )
    return outcome


async def _collect_local_prompt_template_values(
    ctx: CommandContext,
    *,
    filename: str,
    outcome: CommandOutcome,
    agent_name: str,
) -> dict[str, str] | None:
    template_variables = _local_prompt_template_variables(filename)
    if not template_variables:
        return {}

    collected_args = await _collect_prompt_argument_values(
        ctx,
        prompt_name=filename,
        required_args=template_variables,
        optional_args=[],
        arg_descriptions={},
        agent_name=agent_name,
        right_info="prompt load",
        fail_on_missing_required=True,
    )
    if not collected_args.missing_required:
        return collected_args.values

    outcome.add_message(
        _format_missing_required_prompt_arguments(collected_args.missing_required),
        channel="error",
        right_info="prompt load",
        agent_name=agent_name,
    )
    return None


def _local_prompt_template_variables(filename: str) -> list[str]:
    with suppress(Exception):
        return sorted(prompt_file_template_variables(filename))
    return []


def _split_buffer_prefill(
    messages: list[PromptMessageExtended],
) -> LoadedPromptMessages:
    if not messages:
        return LoadedPromptMessages(history_messages=messages, buffered_text=None)

    last_message = messages[-1]
    if last_message.role != "user" or last_message.tool_results:
        return LoadedPromptMessages(history_messages=messages, buffered_text=None)

    content = last_message.content or []
    if not content:
        return LoadedPromptMessages(history_messages=messages, buffered_text=None)

    buffered_text = strip_to_none(get_text(content[0]))
    if buffered_text is None or buffered_text == "<no text>":
        return LoadedPromptMessages(history_messages=messages, buffered_text=None)

    return LoadedPromptMessages(history_messages=messages[:-1], buffered_text=buffered_text)


def _add_loaded_prompt_message(
    outcome: CommandOutcome,
    *,
    filename: str,
    agent_name: str,
    loaded_count: int,
    buffered_text: str | None,
) -> None:
    loaded_messages = format_count(loaded_count, "message")
    if buffered_text:
        outcome.add_message(
            f"Loaded {loaded_messages} from {filename}. Last user message placed in input buffer.",
            channel="info",
            agent_name=agent_name,
        )
        return

    outcome.add_message(
        f"Loaded {loaded_messages} from {filename}",
        channel="info",
        agent_name=agent_name,
    )


async def handle_load_prompt(
    ctx: CommandContext,
    *,
    agent_name: str,
    filename: str | None,
    error: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    if error:
        outcome.add_message(error, channel="error", agent_name=agent_name)
        return outcome

    if filename is None:
        outcome.add_message("Filename required for /prompt load", channel="error")
        return outcome

    arg_values = await _collect_local_prompt_template_values(
        ctx,
        filename=filename,
        outcome=outcome,
        agent_name=agent_name,
    )
    if arg_values is None:
        return outcome

    agent_obj = ctx.agent_provider._agent(agent_name)
    load_result = load_prompt_messages_result(
        filename,
        label="prompt",
        arguments=arg_values or None,
    )
    if load_result.error is not None:
        outcome.add_message(load_result.error, channel="error", agent_name=agent_name)
        return outcome

    messages = load_result.messages
    if messages is None:
        return outcome
    if not messages:
        outcome.add_message(
            f"No messages found in {filename}",
            channel="warning",
            agent_name=agent_name,
        )
        return outcome

    loaded = _split_buffer_prefill(messages)
    if loaded.buffered_text:
        outcome.buffer_prefill = loaded.buffered_text

    replace_agent_history(agent_obj, loaded.history_messages)

    loaded_count = len(loaded.history_messages) + (1 if loaded.buffered_text else 0)
    _add_loaded_prompt_message(
        outcome,
        filename=filename,
        agent_name=agent_name,
        loaded_count=loaded_count,
        buffered_text=loaded.buffered_text,
    )

    return outcome


def _add_prompt_not_found_message(
    outcome: CommandOutcome,
    *,
    requested_name: str,
    all_prompts: "Sequence[PromptSummary]",
    agent_name: str,
) -> None:
    missing = Text()
    missing.append(f"Prompt '{requested_name}' not found.\n", style="red")
    missing.append("Available prompts:\n", style="yellow")
    for prompt in all_prompts:
        missing.append(f"  {prompt['namespaced_name']}\n", style="dim")
    outcome.add_message(
        missing,
        right_info="prompt selection",
        agent_name=agent_name,
    )


async def _select_from_matching_prompts(
    ctx: CommandContext,
    *,
    matching_prompts: "Sequence[PromptSummary]",
    requested_name: str,
    outcome: CommandOutcome,
    agent_name: str,
) -> PromptSummary | None:
    if len(matching_prompts) == 1:
        return matching_prompts[0]

    multi = Text()
    multi.append(
        f"Multiple prompts match '{requested_name}':\n",
        style="yellow",
    )
    for index, prompt in enumerate(matching_prompts, 1):
        description = prompt.get("description") or "No description"
        multi.append(
            f"  {index}. {prompt['namespaced_name']} - {description}\n",
            style="dim",
        )
    selection = await prompt_selection_after_message(
        ctx,
        content=multi,
        right_info="prompt selection",
        agent_name=agent_name,
        prompt="Enter prompt number to select: ",
        options=[str(i) for i, _ in enumerate(matching_prompts, 1)],
        allow_cancel=False,
        default="1",
    )

    return _selected_prompt_from_selection(
        matching_prompts,
        selection,
        outcome=outcome,
        agent_name=agent_name,
    )


async def _select_prompt_by_request(
    ctx: CommandContext,
    *,
    all_prompts: "Sequence[PromptSummary]",
    requested_name: str | None,
    prompt_index: int | None,
    outcome: CommandOutcome,
    agent_name: str,
) -> PromptSummary | None:
    if prompt_index is not None:
        if 1 <= prompt_index <= len(all_prompts):
            return all_prompts[prompt_index - 1]
        outcome.add_message(
            f"Invalid prompt number: {prompt_index}. Valid range is 1-{len(all_prompts)}.",
            channel="error",
            right_info="prompt selection",
            agent_name=agent_name,
        )
        return None

    if requested_name:
        matching_prompts = [
            prompt for prompt in all_prompts if _prompt_matches_name(prompt, requested_name)
        ]
        if not matching_prompts:
            _add_prompt_not_found_message(
                outcome,
                requested_name=requested_name,
                all_prompts=all_prompts,
                agent_name=agent_name,
            )
            return None
        return await _select_from_matching_prompts(
            ctx,
            matching_prompts=matching_prompts,
            requested_name=requested_name,
            outcome=outcome,
            agent_name=agent_name,
        )

    content = _build_prompt_list_text(all_prompts, include_usage=False)

    prompt_names = [str(i) for i, _ in enumerate(all_prompts, 1)]
    selection = await prompt_selection_after_message(
        ctx,
        content=content,
        right_info="prompt selection",
        agent_name=agent_name,
        prompt="Enter prompt number to select (or press Enter to cancel): ",
        options=prompt_names,
        allow_cancel=True,
    )

    return _selected_prompt_from_selection(
        all_prompts,
        selection,
        outcome=outcome,
        agent_name=agent_name,
    )


async def handle_select_prompt(
    ctx: CommandContext,
    *,
    agent_name: str,
    requested_name: str | None = None,
    prompt_index: int | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    requested_name = strip_to_none(requested_name)

    all_prompts = await _get_all_prompts(ctx, agent_name)
    if not all_prompts:
        outcome.add_message(
            "No prompts available for this agent.",
            channel="warning",
            right_info="prompt selection",
            agent_name=agent_name,
        )
        return outcome

    selected_prompt = await _select_prompt_by_request(
        ctx,
        all_prompts=all_prompts,
        requested_name=requested_name,
        prompt_index=prompt_index,
        outcome=outcome,
        agent_name=agent_name,
    )
    if selected_prompt is None:
        return outcome

    collected_args = await _collect_prompt_argument_values(
        ctx,
        prompt_name=selected_prompt["name"],
        required_args=selected_prompt.get("required_args", []),
        optional_args=selected_prompt.get("optional_args", []),
        arg_descriptions=selected_prompt.get("arg_descriptions", {}),
        agent_name=agent_name,
        right_info="prompt selection",
        fail_on_missing_required=False,
    )
    arg_values = collected_args.values

    namespaced_name = selected_prompt["namespaced_name"]
    await ctx.io.emit(
        CommandMessage(
            text=f"Applying prompt {namespaced_name}...",
            right_info="prompt selection",
            agent_name=agent_name,
        )
    )

    agent = ctx.agent_provider._agent(agent_name)

    try:
        prompt_result = await agent.get_prompt(namespaced_name, arg_values)
    except Exception as exc:
        outcome.add_message(
            f"Error applying prompt: {exc}",
            channel="error",
            right_info="prompt selection",
            agent_name=agent_name,
        )
        return outcome

    if not prompt_result or not prompt_result.messages:
        outcome.add_message(
            f"Prompt '{namespaced_name}' could not be found or contains no messages.",
            channel="error",
            right_info="prompt selection",
            agent_name=agent_name,
        )
        return outcome

    multipart_messages = PromptMessageExtended.from_get_prompt_result(prompt_result)

    progress_display.resume()
    try:
        await agent.generate(multipart_messages, None)
    finally:
        progress_display.pause()

    if isinstance(ctx.agent_provider, _TurnUsageDisplayProvider):
        ctx.agent_provider._show_turn_usage(agent_name)

    return outcome
