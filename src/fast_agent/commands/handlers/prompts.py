"""Shared prompt command handlers."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from rich.text import Text

from fast_agent.commands.handlers.shared import (
    load_prompt_messages_from_file,
    replace_agent_history,
)
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.mcp.mcp_aggregator import SEP
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.progress_display import progress_display

if TYPE_CHECKING:
    from mcp.types import Prompt

    from fast_agent.commands.context import CommandContext


def _truncate_prompt_description(description: str, char_limit: int = 240) -> str:
    description = description.strip()
    if len(description) <= char_limit:
        return description

    truncate_pos = char_limit
    sentence_break = description.rfind(". ", 0, char_limit + 20)
    if sentence_break > char_limit - 50:
        truncate_pos = sentence_break + 1
    else:
        word_break = description.rfind(" ", 0, char_limit + 10)
        if word_break > char_limit - 30:
            truncate_pos = word_break

    return description[:truncate_pos].rstrip() + "..."


def _format_prompt_args(prompt: dict[str, Any]) -> str:
    arg_names = prompt.get("arg_names", [])
    required_args = set(prompt.get("required_args", []))
    if arg_names:
        arg_list = [f"{name}*" if name in required_args else name for name in arg_names]
        args_text = ", ".join(arg_list)
        if len(args_text) > 80:
            args_text = args_text[:77] + "..."
        return args_text

    arg_count = prompt.get("arg_count", 0)
    plural = "s" if arg_count != 1 else ""
    return f"{arg_count} parameter{plural}"


def _build_prompt_list_text(
    prompts: list[dict[str, Any]],
    *,
    include_usage: bool,
) -> Text:
    content = Text()

    for index, prompt in enumerate(prompts, 1):
        if content.plain:
            content.append("\n")

        line = Text()
        line.append(f"[{index:2}] ", style="dim cyan")
        line.append(f"{prompt['server']}•", style="dim green")
        line.append(prompt["name"], style="bright_blue bold")

        if prompt.get("title") and str(prompt["title"]).strip():
            line.append(f" {prompt['title']}", style="default")

        content.append_text(line)

        description = (prompt.get("description") or "").strip()
        if description:
            truncated = _truncate_prompt_description(description)
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


def _prompt_matches_name(prompt: dict[str, Any], requested_name: str) -> bool:
    return prompt["name"] == requested_name or prompt["namespaced_name"] == requested_name


async def _get_all_prompts(
    ctx: CommandContext, agent_name: str | None = None
) -> list[dict[str, Any]]:
    try:
        prompt_servers = await ctx.agent_provider.list_prompts(namespace=None, agent_name=agent_name)
    except Exception:
        return []

    if not isinstance(prompt_servers, Mapping):
        return []

    all_prompts: list[dict[str, Any]] = []

    for server_name, prompts_info in prompt_servers.items():
        prompts_list: list[Any] | None = None
        if isinstance(prompts_info, list):
            prompts_list = prompts_info
        else:
            prompts_attr = getattr(prompts_info, "prompts", None)
            if isinstance(prompts_attr, list):
                prompts_list = prompts_attr

        if not prompts_list:
            continue

        for prompt in prompts_list:
            if isinstance(prompt, dict):
                prompt_dict = cast("dict[str, Any]", prompt)
                prompt_name = prompt_dict.get("name")
                if not isinstance(prompt_name, str):
                    continue
                arguments = prompt_dict.get("arguments", [])
                all_prompts.append(
                    {
                        "server": server_name,
                        "name": prompt_name,
                        "namespaced_name": f"{server_name}{SEP}{prompt_name}",
                        "title": prompt_dict.get("title", None),
                        "description": prompt_dict.get("description", "No description"),
                        "arg_count": len(arguments) if isinstance(arguments, list) else 0,
                        "arguments": arguments if isinstance(arguments, list) else [],
                    }
                )
                continue

            prompt_obj = cast("Prompt", prompt)
            all_prompts.append(
                {
                    "server": server_name,
                    "name": prompt_obj.name,
                    "namespaced_name": f"{server_name}{SEP}{prompt_obj.name}",
                    "title": prompt_obj.title or None,
                    "description": prompt_obj.description or "No description",
                    "arg_count": len(prompt_obj.arguments or []),
                    "arguments": prompt_obj.arguments or [],
                }
            )

    all_prompts.sort(key=lambda p: (p["server"], p["name"]))

    return all_prompts


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

    agent_obj = ctx.agent_provider._agent(agent_name)
    messages = load_prompt_messages_from_file(filename, label="prompt")
    if messages is None:
        return outcome
    if not messages:
        outcome.add_message(
            f"No messages found in {filename}",
            channel="warning",
            agent_name=agent_name,
        )
        return outcome

    buffered_text = None
    last_message = messages[-1]
    if last_message.role == "user" and not last_message.tool_results:
        content = getattr(last_message, "content", []) or []
        if content:
            from fast_agent.mcp.helpers.content_helpers import get_text

            buffered_text = get_text(content[0])
        if buffered_text and buffered_text != "<no text>":
            messages = messages[:-1]
            outcome.buffer_prefill = buffered_text

    replace_agent_history(agent_obj, messages)

    loaded_count = len(messages) + (1 if buffered_text else 0)
    if buffered_text:
        outcome.add_message(
            f"Loaded {loaded_count} messages from {filename}. Last user message placed in input buffer.",
            channel="info",
            agent_name=agent_name,
        )
    else:
        outcome.add_message(
            f"Loaded {loaded_count} messages from {filename}",
            channel="info",
            agent_name=agent_name,
        )

    return outcome


async def handle_select_prompt(
    ctx: CommandContext,
    *,
    agent_name: str,
    requested_name: str | None = None,
    prompt_index: int | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    all_prompts = await _get_all_prompts(ctx, agent_name)
    if not all_prompts:
        outcome.add_message(
            "No prompts available for this agent.",
            channel="warning",
            right_info="prompt selection",
            agent_name=agent_name,
        )
        return outcome

    if prompt_index is not None:
        if 1 <= prompt_index <= len(all_prompts):
            selected_prompt = all_prompts[prompt_index - 1]
        else:
            outcome.add_message(
                f"Invalid prompt number: {prompt_index}. Valid range is 1-{len(all_prompts)}.",
                channel="error",
                right_info="prompt selection",
                agent_name=agent_name,
            )
            return outcome
    elif requested_name:
        matching_prompts = [
            prompt for prompt in all_prompts if _prompt_matches_name(prompt, requested_name)
        ]

        if not matching_prompts:
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
            return outcome

        if len(matching_prompts) == 1:
            selected_prompt = matching_prompts[0]
        else:
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
            await ctx.io.emit(
                CommandMessage(
                    text=multi,
                    right_info="prompt selection",
                    agent_name=agent_name,
                )
            )

            selection = await ctx.io.prompt_selection(
                "Enter prompt number to select: ",
                options=[str(i) for i, _ in enumerate(matching_prompts, 1)],
                allow_cancel=False,
                default="1",
            )

            if not selection:
                outcome.add_message(
                    "Prompt selection cancelled.",
                    channel="warning",
                    right_info="prompt selection",
                    agent_name=agent_name,
                )
                return outcome

            try:
                idx = int(selection) - 1
            except ValueError:
                outcome.add_message(
                    "Invalid input, please enter a number.",
                    channel="error",
                    right_info="prompt selection",
                    agent_name=agent_name,
                )
                return outcome

            if not (0 <= idx < len(matching_prompts)):
                outcome.add_message(
                    "Invalid selection.",
                    channel="error",
                    right_info="prompt selection",
                    agent_name=agent_name,
                )
                return outcome

            selected_prompt = matching_prompts[idx]
    else:
        content = _build_prompt_list_text(all_prompts, include_usage=False)
        await ctx.io.emit(
            CommandMessage(
                text=content,
                right_info="prompt selection",
                agent_name=agent_name,
            )
        )

        prompt_names = [str(i) for i, _ in enumerate(all_prompts, 1)]
        selection = await ctx.io.prompt_selection(
            "Enter prompt number to select (or press Enter to cancel): ",
            options=prompt_names,
            allow_cancel=True,
        )

        if not selection:
            outcome.add_message(
                "Prompt selection cancelled.",
                channel="warning",
                right_info="prompt selection",
                agent_name=agent_name,
            )
            return outcome

        try:
            idx = int(selection) - 1
        except ValueError:
            outcome.add_message(
                "Invalid input, please enter a number.",
                channel="error",
                right_info="prompt selection",
                agent_name=agent_name,
            )
            return outcome

        if not (0 <= idx < len(all_prompts)):
            outcome.add_message(
                "Invalid selection.",
                channel="error",
                right_info="prompt selection",
                agent_name=agent_name,
            )
            return outcome

        selected_prompt = all_prompts[idx]

    required_args = selected_prompt.get("required_args", [])
    optional_args = selected_prompt.get("optional_args", [])
    arg_descriptions = selected_prompt.get("arg_descriptions", {})
    arg_values: dict[str, str] = {}

    if required_args or optional_args:
        if required_args and optional_args:
            arg_header = (
                f"Prompt {selected_prompt['name']} requires {len(required_args)} "
                f"arguments and has {len(optional_args)} optional arguments:"
            )
        elif required_args:
            arg_header = (
                f"Prompt {selected_prompt['name']} requires {len(required_args)} arguments:"
            )
        else:
            arg_header = (
                f"Prompt {selected_prompt['name']} has {len(optional_args)} optional arguments:"
            )

        await ctx.io.emit(
            CommandMessage(
                text=Text(arg_header, style="cyan"),
                right_info="prompt selection",
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
            if arg_value is not None:
                arg_values[arg_name] = arg_value

        for arg_name in optional_args:
            description = arg_descriptions.get(arg_name, "")
            arg_value = await ctx.io.prompt_argument(
                arg_name,
                description=description,
                required=False,
            )
            if arg_value:
                arg_values[arg_name] = arg_value

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
    except Exception as exc:  # noqa: BLE001
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

    show_usage = getattr(ctx.agent_provider, "_show_turn_usage", None)
    if callable(show_usage):
        show_usage(agent_name)

    return outcome
