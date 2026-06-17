"""Reusable completion providers for prompt commands."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from prompt_toolkit.completion import Completion

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.command_catalog import (
    command_action_tokens,
    get_command_action_spec,
    get_command_spec,
)
from fast_agent.commands.mcp_command_intents import (
    MCP_TOP_LEVEL_ACTION_DESCRIPTIONS,
)
from fast_agent.commands.session_export_help import build_session_export_action_detail
from fast_agent.commands.shared_command_intents import (
    ADD_TOOL_TOKEN_DESCRIPTIONS,
    DUMP_TOOL_TOKEN_DESCRIPTIONS,
    HISTORY_COMMAND_COMPLETION_DESCRIPTIONS,
    MODEL_MANAGER_COMMAND_ACTIONS,
    MODEL_VALUE_COMMAND_ACTIONS,
    REMOVE_TOOL_TOKEN_DESCRIPTIONS,
    SESSION_COMMAND_COMPLETION_DESCRIPTIONS,
)
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.mcp.connect_targets import mcp_connect_flag_descriptions
from fast_agent.utils.commandline import join_commandline, split_commandline
from fast_agent.utils.text import starts_with_casefold, strip_casefold

if TYPE_CHECKING:
    from fast_agent.ui.prompt.completer import AgentCompleter


MODEL_TOOL_STATE_VALUES = ("on", "off", "default")
ModelValueCompletionSpec = tuple["Iterable[str]", str]
_MODEL_VALUE_COMPLETION_SUBCOMMANDS = MODEL_VALUE_COMMAND_ACTIONS - frozenset(("switch",))
ModelManagerCompletionHandler = Callable[
    ["AgentCompleter", str, list[Completion]],
    list[Completion],
]
MarketplaceCompletionHandler = Callable[
    ["AgentCompleter", str, list[Completion]],
    list[Completion],
]
MarketplaceCompletionDispatch = tuple[tuple[frozenset[str], MarketplaceCompletionHandler], ...]
SessionPrefixCompletionHandler = Callable[
    ["AgentCompleter", str, str],
    list[Completion] | None,
]
HistoryPrefixCompletionHandler = Callable[
    ["AgentCompleter", str],
    list[Completion],
]


def _catalog_action_tokens(command_name: str, action_name: str) -> frozenset[str]:
    tokens = command_action_tokens(command_name, action_name)
    if not tokens:
        raise ValueError(f"unknown /{command_name} action: {action_name}")
    return frozenset(tokens)


_SKILLS_ADD_ACTIONS = _catalog_action_tokens("skills", "add")
_SKILLS_REMOVE_ACTIONS = _catalog_action_tokens("skills", "remove")
_SKILLS_UPDATE_ACTIONS = _catalog_action_tokens("skills", "update")
_SKILLS_REGISTRY_ACTIONS = _catalog_action_tokens("skills", "registry")
_SKILLS_SEARCH_ACTIONS = _catalog_action_tokens("skills", "search")

_CARDS_ADD_ACTIONS = _catalog_action_tokens("cards", "add")
_CARDS_REMOVE_ACTIONS = _catalog_action_tokens("cards", "remove")
_CARDS_UPDATE_ACTIONS = _catalog_action_tokens("cards", "update")
_CARDS_REGISTRY_ACTIONS = _catalog_action_tokens("cards", "registry")
_CARDS_README_ACTIONS = _catalog_action_tokens("cards", "readme")
_CARDS_PUBLISH_ACTIONS = _catalog_action_tokens("cards", "publish")

_PLUGINS_ADD_ACTIONS = _catalog_action_tokens("plugins", "add")
_PLUGINS_REMOVE_ACTIONS = _catalog_action_tokens("plugins", "remove")
_PLUGINS_UPDATE_ACTIONS = _catalog_action_tokens("plugins", "update")
_PLUGINS_REGISTRY_ACTIONS = _catalog_action_tokens("plugins", "registry")

_CARD_TOOL_FLAG_DESCRIPTIONS = {
    **ADD_TOOL_TOKEN_DESCRIPTIONS,
    **REMOVE_TOOL_TOKEN_DESCRIPTIONS,
}
_AGENT_TOOL_FLAG_DESCRIPTIONS = {
    **ADD_TOOL_TOKEN_DESCRIPTIONS,
    **REMOVE_TOOL_TOKEN_DESCRIPTIONS,
    **DUMP_TOOL_TOKEN_DESCRIPTIONS,
}


@runtime_checkable
class _AttachedServerAggregator(Protocol):
    def list_attached_servers(self) -> list[str]: ...


@runtime_checkable
class _AttachedServerAgent(Protocol):
    @property
    def aggregator(self) -> _AttachedServerAggregator: ...


def _completion_parts(text: str) -> list[str]:
    return text.split(maxsplit=1) if text else []


def _command_completion_context(
    completer: "AgentCompleter",
    *,
    command_name: str,
    text: str,
    subcommands: Mapping[str, str] | None = None,
) -> tuple[list[str], str, list[Completion], bool]:
    prefix = f"/{command_name} "
    remainder = text[len(prefix) :] or ""
    parts = _completion_parts(remainder)
    command_subcommands = subcommands or _catalog_subcommands(
        command_name,
        include_aliases=_include_subcommand_aliases(command_name, text),
    )
    results = list(completer._complete_subcommands(parts, remainder, command_subcommands))
    needs_subcommand_only = not parts or (len(parts) == 1 and not remainder.endswith(" "))
    return parts, remainder, results, needs_subcommand_only


def _include_subcommand_aliases(command_name: str, text: str) -> bool:
    prefix = f"/{command_name} "
    remainder = text[len(prefix) :] or ""
    parts = _completion_parts(remainder)
    return bool(parts and parts[0] and not remainder.endswith(" "))


def _hint_completion(display: str, display_meta: str) -> Completion:
    """Build an informational completion that inserts nothing.

    Used as a discoverability hint when a subcommand accepts an argument
    we can't complete synchronously (e.g. marketplace entries behind
    network fetches). Selecting the hint is a no-op; its only purpose is
    to surface the signature and guidance in the completion menu.

    Note: prompt-toolkit drops the completion menu when a single
    completion would "do nothing" (buffer.completion_does_nothing). Callers
    yielding hints for otherwise-empty argument slots should emit at least
    two entries (see :func:`_signature_hints`) or pair a hint with real
    completions.
    """
    return Completion(
        "",
        start_position=0,
        display=display,
        display_meta=display_meta,
    )


def _signature_hints(kind: str, *, empty_arg_list: bool = True) -> list[Completion]:
    """Build several hint completions describing an argument signature.

    Two or more entries are returned so the prompt-toolkit menu renders
    them; a single no-op completion is culled by the buffer.
    """
    hints = []
    if empty_arg_list:
        hints.append(
            _hint_completion(
                "(empty)",
                "press Enter with no argument to pick from a list",
            )
        )
    hints.extend(
        [
            _hint_completion("<number>", f"by index — {kind}"),
            _hint_completion("<name>", f"by name — {kind}"),
        ]
    )
    return hints


def _extend_managed_update_completions(
    results: list[Completion],
    argument: str,
    *,
    command_name: str,
    all_meta: str,
    name_completions: "Iterable[Completion]",
) -> list[Completion]:
    argument_lower = strip_casefold(argument)
    if "all".startswith(argument_lower):
        results.append(
            Completion(
                "all",
                start_position=-len(argument),
                display="all",
                display_meta=all_meta,
            )
        )
    results.extend(_catalog_option_completions(command_name, "update", argument))
    results.extend(name_completions)
    return results


def _extend_name_or_signature_completions(
    results: list[Completion],
    argument: str,
    *,
    name_completions: "Iterable[Completion]",
    signature_kind: str,
    empty_arg_list: bool = True,
) -> list[Completion]:
    completions = list(name_completions)
    if completions:
        results.extend(completions)
    elif not argument:
        results.extend(_signature_hints(signature_kind, empty_arg_list=empty_arg_list))
    return results


def _extend_registry_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
    *,
    registry_completions: "Iterable[Completion]",
    include_paths: bool = False,
) -> list[Completion]:
    results.extend(registry_completions)
    if include_paths:
        results.extend(completer._complete_registry_paths(argument))
    return results


def _dispatch_marketplace_action_completions(
    dispatch: MarketplaceCompletionDispatch,
    *,
    completer: "AgentCompleter",
    subcmd: str,
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    for action_tokens, handler in dispatch:
        if subcmd in action_tokens:
            return handler(completer, argument, results)
    return results


def _marketplace_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
    *,
    command_name: str,
    dispatch: MarketplaceCompletionDispatch,
) -> list[Completion] | None:
    if not text_lower.startswith(f"/{command_name} "):
        return None

    parts, _remainder, results, needs_subcommand_only = _command_completion_context(
        completer,
        command_name=command_name,
        text=text,
    )
    if needs_subcommand_only:
        return results

    subcmd = strip_casefold(parts[0])
    argument = parts[1] if len(parts) > 1 else ""
    return _dispatch_marketplace_action_completions(
        dispatch,
        completer=completer,
        subcmd=subcmd,
        argument=argument,
        results=results,
    )


def _attached_mcp_servers_for_completion(completer: "AgentCompleter") -> list[str]:
    if completer.agent_provider is not None and completer.current_agent:
        try:
            agent = completer.agent_provider._agent(completer.current_agent)
            if isinstance(agent, _AttachedServerAgent):
                return agent.aggregator.list_attached_servers()
        except Exception:
            return []
    return []


def _complete_attached_mcp_servers(completer: "AgentCompleter", partial: str) -> list[Completion]:
    attached = _attached_mcp_servers_for_completion(completer)
    return [
        Completion(
            server,
            start_position=-len(partial),
            display=server,
            display_meta="attached mcp server",
        )
        for server in attached
        if starts_with_casefold(server, partial)
    ]


def _mcp_connect_completions(completer: "AgentCompleter", remainder: str) -> list[Completion]:
    connect_context = completer._mcp_connect_context(remainder)

    if connect_context.context in {"target", "new_token"} and connect_context.target_count == 0:
        results = [completer._mcp_connect_target_hint(connect_context.partial)]
        results.extend(list(completer._complete_configured_mcp_servers(connect_context.partial)))
        return results

    if connect_context.context == "new_token" and connect_context.target_count > 0:
        return _mcp_connect_flag_completions("", start_position=0)

    if connect_context.context == "flag" and connect_context.target_count > 0:
        return _mcp_connect_flag_completions(connect_context.partial)

    return []


def _mcp_connect_flag_completions(
    partial: str,
    *,
    start_position: int | None = None,
) -> list[Completion]:
    return _flag_completions(
        partial,
        mcp_connect_flag_descriptions(),
        start_position=start_position,
    )


def _subcommand_completions(
    partial: str,
    subcommands: dict[str, str],
) -> list[Completion]:
    return [
        Completion(
            subcmd,
            start_position=-len(partial),
            display=subcmd,
            display_meta=description,
        )
        for subcmd, description in subcommands.items()
        if starts_with_casefold(subcmd, partial)
    ]


def _value_completions(
    argument: str,
    values: "Iterable[str]",
    *,
    display_meta: str,
) -> list[Completion]:
    return [
        Completion(
            value,
            start_position=-len(argument),
            display=value,
            display_meta=display_meta,
        )
        for value in values
        if starts_with_casefold(value, argument)
    ]


def _flag_completions(
    partial: str,
    flags: "Mapping[str, str]",
    *,
    start_position: int | None = None,
) -> list[Completion]:
    return [
        Completion(
            flag,
            start_position=-len(partial) if start_position is None else start_position,
            display=flag,
            display_meta=description,
        )
        for flag, description in flags.items()
        if starts_with_casefold(flag, partial)
    ]


def _catalog_subcommands(
    command_name: str,
    *,
    include_aliases: bool = False,
) -> dict[str, str]:
    spec = get_command_spec(command_name)
    if spec is None:
        raise ValueError(f"unknown command catalog entry: {command_name}")

    subcommands: dict[str, str] = {action.action: action.help for action in spec.actions}
    if include_aliases:
        for action in spec.actions:
            subcommands.update({alias: f"alias for {action.action}" for alias in action.aliases})
    return subcommands


def _current_argument_token(argument: str) -> str:
    if argument and not argument.endswith(" "):
        return argument.split()[-1]
    return ""


def _completion_subject_is_finished(argument: str) -> bool:
    return argument.endswith(" ") or len(_completion_parts(argument)) > 1


def _catalog_option_completions(
    command_name: str,
    action_name: str,
    argument: str,
) -> list[Completion]:
    current_token = _current_argument_token(argument)
    if current_token and not current_token.startswith("-"):
        return []

    action_spec = get_command_action_spec(command_name, action_name)
    if action_spec is None:
        return []

    option_descriptions = {
        option_name: option.summary
        for option in action_spec.options
        for option_name in (option.name, *option.aliases)
    }
    return _flag_completions(current_token, option_descriptions)


def _model_subcommands_for_completion(
    completer: "AgentCompleter",
    *,
    include_aliases: bool = False,
) -> dict[str, str]:
    subcommands = _catalog_subcommands("model", include_aliases=include_aliases)
    supported_features = _model_value_completion_specs(completer)
    for subcommand in _MODEL_VALUE_COMPLETION_SUBCOMMANDS:
        if subcommand not in supported_features:
            subcommands.pop(subcommand, None)
    return subcommands


def _models_subcommands_for_completion(*, include_aliases: bool = False) -> dict[str, str]:
    spec = get_command_spec("models")
    if spec is None:
        return {}

    subcommands: dict[str, str] = {}
    for action in spec.actions:
        if action.action not in MODEL_MANAGER_COMMAND_ACTIONS:
            continue
        subcommands[action.action] = action.help
        if include_aliases:
            subcommands.update({alias: f"alias for {action.action}" for alias in action.aliases})
    return subcommands


def _history_webclear_completions(
    completer: "AgentCompleter",
    partial: str,
) -> list[Completion]:
    if not completer._current_agent_has_web_tools_enabled():
        return []

    stripped = partial.strip()
    return [
        Completion(
            name,
            start_position=-len(stripped),
            display=name,
            display_meta="Strip web metadata channels for this agent",
        )
        for name in sorted(completer.agents)
        if not stripped or starts_with_casefold(name, stripped)
    ]


def _history_load_completions(
    completer: "AgentCompleter",
    partial: str,
) -> list[Completion]:
    return list(completer._complete_history_files(partial))


def _history_turn_completions(
    completer: "AgentCompleter",
    partial: str,
) -> list[Completion]:
    return list(completer._complete_history_rewind(partial))


def _history_clear_completions(
    _completer: "AgentCompleter",
    partial: str,
) -> list[Completion]:
    subcommands = {
        "all": "Clear the full history",
        "last": "Remove the most recent message",
    }
    return [
        Completion(
            subcmd,
            start_position=-len(partial),
            display=subcmd,
            display_meta=description,
        )
        for subcmd, description in subcommands.items()
        if starts_with_casefold(subcmd, partial)
    ]


_HISTORY_PREFIX_COMPLETION_HANDLERS: tuple[
    tuple[str, HistoryPrefixCompletionHandler],
    ...,
] = (
    ("/history load ", _history_load_completions),
    ("/history rewind ", _history_turn_completions),
    ("/history detail ", _history_turn_completions),
    ("/history review ", _history_turn_completions),
    ("/history clear ", _history_clear_completions),
    ("/history webclear ", _history_webclear_completions),
)


def _history_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    for prefix, handler in _HISTORY_PREFIX_COMPLETION_HANDLERS:
        if text_lower.startswith(prefix):
            partial = text[len(prefix) :]
            return handler(completer, partial)

    if not text_lower.startswith("/history "):
        return None

    partial = text[len("/history ") :]
    subcommands = dict(HISTORY_COMMAND_COMPLETION_DESCRIPTIONS)
    if not completer._current_agent_has_web_tools_enabled():
        subcommands.pop("webclear", None)
    return _subcommand_completions(partial, subcommands)


def _compact_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    del completer
    if not text_lower.startswith("/compact "):
        return None

    partial = text[len("/compact ") :]
    return _subcommand_completions(
        partial,
        {
            "preview": "Show what compaction would keep (no model call)",
            "prompt": "Show the active compaction prompt",
        },
    )


def _prompt_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    if not text_lower.startswith("/prompt "):
        return None

    partial = text[len("/prompt ") :]
    if text_lower.startswith("/prompt load "):
        return list(completer._complete_prompt_files(text[len("/prompt load ") :]))

    return _subcommand_completions(partial, {"load": "Load prompt template by file"})


def _attach_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    if not text_lower.startswith("/attach "):
        return None

    remainder = text[len("/attach ") :]
    if not remainder:
        results = _attach_static_completions("", start_position=0)
        results.extend(list(completer._complete_shell_paths("", 0)))
        return results

    try:
        parts = split_commandline(remainder)
    except ValueError:
        return []

    if remainder.endswith((" ", "\t")):
        partial = ""
        token_count = len(parts)
    else:
        partial = parts[-1] if parts else remainder
        token_count = len(parts) if parts else 1

    results: list[Completion] = []
    if token_count <= 1:
        results.extend(_attach_static_completions(partial))
    for completion in completer._complete_shell_paths(partial, len(partial)):
        completion_text = join_commandline([completion.text])
        results.append(
            Completion(
                completion_text,
                start_position=completion.start_position,
                display=completion.display,
                display_meta=completion.display_meta,
            )
        )
    return results


def _attach_static_completions(
    partial: str,
    *,
    start_position: int | None = None,
) -> list[Completion]:
    start = -len(partial) if start_position is None else start_position
    return [
        Completion(
            value,
            start_position=start,
            display=value,
            display_meta=description,
        )
        for value, description in _ATTACH_STATIC_COMPLETIONS.items()
        if starts_with_casefold(value, partial)
    ]


_ATTACH_STATIC_COMPLETIONS = {
    "clear": "remove staged file or URL attachments from the next draft buffer",
    "https://": "stage a remote URL attachment for the next prompt",
}


def _session_delete_completions(
    completer: "AgentCompleter",
    partial: str,
) -> list[Completion]:
    results: list[Completion] = []
    if starts_with_casefold("all", partial):
        results.append(
            Completion(
                "all",
                start_position=-len(partial),
                display="all",
                display_meta="Delete all sessions",
            )
        )
    results.extend(list(completer._complete_session_ids(partial)))
    return results


def _session_pin_state_completions(
    prefix: str = "",
    *,
    start_position: int = 0,
) -> list[Completion]:
    return [
        Completion(
            option,
            start_position=start_position,
            display=option,
            display_meta="Toggle session pin",
        )
        for option in ("on", "off")
        if starts_with_casefold(option, prefix)
    ]


def _session_pin_completions(
    completer: "AgentCompleter",
    text: str,
) -> list[Completion]:
    remainder = text[len("/session pin ") :]
    parts = _completion_parts(remainder)
    if not parts:
        results = _session_pin_state_completions()
        results.extend(list(completer._complete_session_ids("")))
        return results

    first_token = parts[0]
    first = strip_casefold(first_token)
    if len(parts) == 1 and not remainder.endswith(" "):
        state_completions = _session_pin_state_completions(
            first_token,
            start_position=-len(first_token),
        )
        if state_completions:
            return state_completions

    if first in {"on", "off"}:
        suffix = parts[1] if len(parts) > 1 else ""
        start_position = -len(suffix) if suffix else 0
        return list(completer._complete_session_ids(suffix, start_position=start_position))

    return list(completer._complete_session_ids(remainder))


def _session_export_completions(
    completer: "AgentCompleter",
    partial: str,
) -> list[Completion] | None:
    export_option_completions = _session_export_option_completions(partial)
    if export_option_completions is not None:
        return export_option_completions

    results: list[Completion] = []
    if starts_with_casefold("latest", partial):
        results.append(
            Completion(
                "latest",
                start_position=-len(partial),
                display="latest",
                display_meta="Most recent session",
            )
        )
    results.extend(list(completer._complete_session_ids(partial)))
    return results


def _session_export_option_completions(partial: str) -> list[Completion] | None:
    tokens = partial.split()
    current_token = tokens[-1] if tokens and not partial.endswith(" ") else ""
    previous_token = (
        tokens[-1] if partial.endswith(" ") and tokens else (tokens[-2] if len(tokens) >= 2 else "")
    )

    option_details = build_session_export_action_detail()["options"]
    value_options = {
        option_name: option
        for option in option_details
        if option["value_name"] is not None
        for option_name in (option["name"], *option["aliases"])
    }
    if previous_token in value_options:
        values = _pipe_delimited_option_values(value_options[previous_token]["value_name"])
        return _value_completions(current_token, values, display_meta=previous_token)

    if not current_token.startswith("-"):
        return None

    option_descriptions = {
        option_name: option["summary"]
        for option in option_details
        for option_name in (option["name"], *option["aliases"])
    }
    return _flag_completions(current_token, option_descriptions)


def _pipe_delimited_option_values(value_name: str | None) -> tuple[str, ...]:
    if value_name is None or "|" not in value_name:
        return ()
    return tuple(part for part in value_name.split("|") if part)


def _session_resume_prefix_completions(
    completer: "AgentCompleter",
    text: str,
    prefix: str,
) -> list[Completion]:
    partial = text[len(prefix) :]
    return list(completer._complete_session_ids(partial))


def _session_delete_prefix_completions(
    completer: "AgentCompleter",
    text: str,
    prefix: str,
) -> list[Completion]:
    partial = text[len(prefix) :]
    return _session_delete_completions(completer, partial)


def _session_pin_prefix_completions(
    completer: "AgentCompleter",
    text: str,
    _prefix: str,
) -> list[Completion]:
    return _session_pin_completions(completer, text)


def _session_export_prefix_completions(
    completer: "AgentCompleter",
    text: str,
    prefix: str,
) -> list[Completion] | None:
    partial = text[len(prefix) :]
    return _session_export_completions(completer, partial)


_SESSION_PREFIX_COMPLETION_HANDLERS: tuple[
    tuple[str, SessionPrefixCompletionHandler],
    ...,
] = (
    ("/resume ", _session_resume_prefix_completions),
    ("/session resume ", _session_resume_prefix_completions),
    ("/session delete ", _session_delete_prefix_completions),
    ("/session clear ", _session_delete_prefix_completions),
    ("/session pin ", _session_pin_prefix_completions),
    ("/session export ", _session_export_prefix_completions),
)


def _session_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    for prefix, handler in _SESSION_PREFIX_COMPLETION_HANDLERS:
        if text_lower.startswith(prefix):
            completions = handler(completer, text, prefix)
            if completions is not None:
                return completions

    if not text_lower.startswith("/session "):
        return None

    partial = text[len("/session ") :]
    return _subcommand_completions(partial, SESSION_COMMAND_COMPLETION_DESCRIPTIONS)


def _skills_add_completions(
    _completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    results.extend(_catalog_option_completions("skills", "add", argument))
    if not argument:
        results.extend(_signature_hints("marketplace skill"))
        results.extend(
            [
                _hint_completion("<github-url>", "direct GitHub SKILL.md URL"),
                _hint_completion("<path>", "direct local SKILL.md file or skill directory"),
            ]
        )
    return results


def _skills_search_completions(
    _completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    if not argument:
        results.extend(
            [
                _hint_completion("<query>", "filter marketplace skills by name/description"),
                _hint_completion("(empty)", "no arg lists all; run /skills available to browse"),
            ]
        )
    return results


def _skills_remove_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_name_or_signature_completions(
        results,
        argument,
        name_completions=completer._complete_local_skill_names(argument),
        signature_kind="managed skill",
    )


def _skills_update_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_managed_update_completions(
        results,
        argument,
        command_name="skills",
        all_meta="update all managed skills",
        name_completions=(
            completer._complete_local_skill_names(
                argument,
                managed_only=True,
                include_indices=False,
            )
        ),
    )


def _skills_registry_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_registry_completions(
        completer,
        argument,
        results,
        registry_completions=completer._complete_skill_registries(argument),
        include_paths=True,
    )


_SKILLS_COMPLETION_DISPATCH: MarketplaceCompletionDispatch = (
    (_SKILLS_ADD_ACTIONS, _skills_add_completions),
    (_SKILLS_SEARCH_ACTIONS, _skills_search_completions),
    (_SKILLS_REMOVE_ACTIONS, _skills_remove_completions),
    (_SKILLS_UPDATE_ACTIONS, _skills_update_completions),
    (_SKILLS_REGISTRY_ACTIONS, _skills_registry_completions),
)
_skills_command_completions = partial(
    _marketplace_command_completions,
    command_name="skills",
    dispatch=_SKILLS_COMPLETION_DISPATCH,
)


def _cards_add_completions(
    _completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    results.extend(_catalog_option_completions("cards", "add", argument))
    if not argument:
        results.extend(_signature_hints("marketplace card pack"))
    return results


def _cards_publish_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    results.extend(_catalog_option_completions("cards", "publish", argument))

    results.extend(
        list(
            completer._complete_local_card_pack_names(
                argument,
                managed_only=True,
                include_indices=False,
            )
        )
    )
    return results


def _cards_remove_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_name_or_signature_completions(
        results,
        argument,
        name_completions=completer._complete_local_card_pack_names(argument),
        signature_kind="installed card pack",
    )


def _cards_readme_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_name_or_signature_completions(
        results,
        argument,
        name_completions=completer._complete_local_card_pack_names(argument),
        signature_kind="installed card pack",
        empty_arg_list=False,
    )


def _cards_update_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_managed_update_completions(
        results,
        argument,
        command_name="cards",
        all_meta="update all managed card packs",
        name_completions=(
            completer._complete_local_card_pack_names(
                argument,
                managed_only=True,
                include_indices=False,
            )
        ),
    )


def _cards_registry_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_registry_completions(
        completer,
        argument,
        results,
        registry_completions=completer._complete_card_registries(argument),
    )


_CARDS_COMPLETION_DISPATCH: MarketplaceCompletionDispatch = (
    (_CARDS_ADD_ACTIONS, _cards_add_completions),
    (_CARDS_REMOVE_ACTIONS, _cards_remove_completions),
    (_CARDS_README_ACTIONS, _cards_readme_completions),
    (_CARDS_UPDATE_ACTIONS, _cards_update_completions),
    (_CARDS_REGISTRY_ACTIONS, _cards_registry_completions),
    (_CARDS_PUBLISH_ACTIONS, _cards_publish_completions),
)
_cards_command_completions = partial(
    _marketplace_command_completions,
    command_name="cards",
    dispatch=_CARDS_COMPLETION_DISPATCH,
)


def _plugins_add_completions(
    _completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    results.extend(_catalog_option_completions("plugins", "add", argument))
    if not argument:
        results.extend(_signature_hints("marketplace plugin"))
    return results


def _plugins_remove_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_name_or_signature_completions(
        results,
        argument,
        name_completions=completer._complete_local_plugin_names(argument),
        signature_kind="installed plugin",
    )


def _plugins_update_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_managed_update_completions(
        results,
        argument,
        command_name="plugins",
        all_meta="update all managed plugins",
        name_completions=(
            completer._complete_local_plugin_names(
                argument,
                managed_only=True,
                include_indices=False,
            )
        ),
    )


def _plugins_registry_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _extend_registry_completions(
        completer,
        argument,
        results,
        registry_completions=completer._complete_plugin_registries(argument),
        include_paths=True,
    )


_PLUGINS_COMPLETION_DISPATCH: MarketplaceCompletionDispatch = (
    (_PLUGINS_ADD_ACTIONS, _plugins_add_completions),
    (_PLUGINS_REMOVE_ACTIONS, _plugins_remove_completions),
    (_PLUGINS_UPDATE_ACTIONS, _plugins_update_completions),
    (_PLUGINS_REGISTRY_ACTIONS, _plugins_registry_completions),
)
_plugins_command_completions = partial(
    _marketplace_command_completions,
    command_name="plugins",
    dispatch=_PLUGINS_COMPLETION_DISPATCH,
)


def _model_references_completions(
    completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    alias_parts = _completion_parts(argument)
    alias_subcommands = {
        "list": "List configured reference mappings",
        "set": "Set an alias: set [<token> [<model-spec>]]",
        "unset": "Unset an alias: unset [<token>]",
    }
    results.extend(completer._complete_subcommands(alias_parts, argument, alias_subcommands))

    if not alias_parts or (len(alias_parts) == 1 and not argument.endswith(" ")):
        return results

    alias_action = strip_casefold(alias_parts[0])
    if alias_action not in {"set", "unset"}:
        return results

    alias_argument = alias_parts[1] if len(alias_parts) > 1 else ""
    alias_tokens = alias_argument.split()
    current_token = ""
    if alias_argument and not alias_argument.endswith(" ") and alias_tokens:
        current_token = alias_tokens[-1]

    if alias_tokens and alias_tokens[-1] == "--target" and alias_argument.endswith(" "):
        results.extend(_model_reference_target_completions(""))
        return results

    if len(alias_tokens) >= 2 and alias_tokens[-2] == "--target" and current_token:
        results.extend(_model_reference_target_completions(current_token))
        return results

    if (not current_token and alias_argument.endswith(" ")) or current_token.startswith("--"):
        alias_flags = {
            "--target": "choose write target (env|project)",
            "--dry-run": "preview changes without writing",
        }
        if _model_reference_target_seen(alias_tokens):
            alias_flags.pop("--target", None)
        results.extend(_flag_completions(current_token, alias_flags))
    return results


def _model_reference_target_seen(tokens: list[str]) -> bool:
    return any(token == "--target" or token.startswith("--target=") for token in tokens)


def _model_reference_target_completions(partial: str) -> list[Completion]:
    return [
        Completion(
            target,
            start_position=-len(partial) if partial else 0,
            display=target,
            display_meta="alias write target",
        )
        for target in ("env", "project")
        if starts_with_casefold(target, partial)
    ]


def _model_catalog_completions(argument: str, results: list[Completion]) -> list[Completion]:
    provider_names = sorted(
        {provider.config_name for provider in ModelSelectionCatalog.CATALOG_ENTRIES_BY_PROVIDER}
    )
    catalog_tokens = argument.split()
    current_token = ""
    if argument and not argument.endswith(" "):
        current_token = catalog_tokens[-1]

    if (not catalog_tokens and not argument) or (
        current_token and not current_token.startswith("--")
    ):
        partial = strip_casefold(current_token)
        results.extend(
            Completion(
                provider_name,
                start_position=-len(current_token),
                display=provider_name,
                display_meta="provider",
            )
            for provider_name in provider_names
            if provider_name.startswith(partial)
        )

    if (not current_token and argument.endswith(" ")) or current_token.startswith("--"):
        results.extend(
            _flag_completions(
                current_token,
                {"--all": "include all known models"},
            )
        )
    return results


def _model_catalog_completion_handler(
    _completer: "AgentCompleter",
    argument: str,
    results: list[Completion],
) -> list[Completion]:
    return _model_catalog_completions(argument, results)


_MODEL_MANAGER_COMPLETION_HANDLERS: dict[str, ModelManagerCompletionHandler] = {
    "references": _model_references_completions,
    "catalog": _model_catalog_completion_handler,
}

_MODEL_COMMAND_COMPLETION_MODES = {
    "model": True,
    "models": False,
}


def _model_value_completion_specs(
    completer: "AgentCompleter",
) -> dict[str, ModelValueCompletionSpec]:
    specs: dict[str, ModelValueCompletionSpec] = {
        "reasoning": (completer._resolve_reasoning_values(), "reasoning"),
    }
    if verbosity_values := completer._resolve_verbosity_values():
        specs["verbosity"] = (verbosity_values, "verbosity")
    if completer._supports_task_budget_setting():
        specs["task_budget"] = (completer._resolve_task_budget_values(), "task budget")
    if completer._supports_service_tier_setting():
        specs["fast"] = (completer._resolve_service_tier_values(), "fast")
    if completer._supports_web_search_setting():
        specs["web_search"] = (MODEL_TOOL_STATE_VALUES, "web_search")
    if completer._supports_x_search_setting():
        specs["x_search"] = (MODEL_TOOL_STATE_VALUES, "x_search")
    if completer._supports_web_fetch_setting():
        specs["web_fetch"] = (MODEL_TOOL_STATE_VALUES, "web_fetch")
    return specs


def _model_command_completions_for_name(
    completer: "AgentCompleter",
    *,
    command_name: str,
    text: str,
    include_value_actions: bool,
) -> list[Completion] | None:
    include_aliases = _include_subcommand_aliases(command_name, text)
    parts, _remainder, results, needs_subcommand_only = _command_completion_context(
        completer,
        command_name=command_name,
        text=text,
        subcommands=(
            _model_subcommands_for_completion(completer, include_aliases=include_aliases)
            if include_value_actions
            else _models_subcommands_for_completion(include_aliases=include_aliases)
        ),
    )
    if needs_subcommand_only:
        return results

    subcmd = strip_casefold(parts[0])
    argument = parts[1] if len(parts) > 1 else ""
    if include_value_actions and (
        value_completion_spec := _model_value_completion_specs(completer).get(subcmd)
    ):
        values, display_meta = value_completion_spec
        results.extend(_value_completions(argument, values, display_meta=display_meta))
        return results
    manager_handler = _MODEL_MANAGER_COMPLETION_HANDLERS.get(subcmd)
    if manager_handler is not None:
        return manager_handler(completer, argument, results)
    return results


def _model_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    for command_name, include_value_actions in _MODEL_COMMAND_COMPLETION_MODES.items():
        if text_lower.startswith(f"/{command_name} "):
            return _model_command_completions_for_name(
                completer,
                command_name=command_name,
                text=text,
                include_value_actions=include_value_actions,
            )
    return None


def _mcp_prefix_completion(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    for prefix, completion_fn in (
        ("/mcp disconnect ", _complete_attached_mcp_servers),
        ("/mcp reconnect ", _complete_attached_mcp_servers),
        ("/mcp connect ", _mcp_connect_completions),
        ("/connect ", _mcp_connect_completions),
    ):
        if text_lower.startswith(prefix):
            return completion_fn(completer, text[len(prefix) :])
    return None


def _mcp_subcommand_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    if not text_lower.startswith("/mcp "):
        return None

    _parts, _remainder, results, _needs_subcommand_only = _command_completion_context(
        completer,
        command_name="mcp",
        text=text,
        subcommands=MCP_TOP_LEVEL_ACTION_DESCRIPTIONS,
    )
    return results


def _mcp_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    prefix_completions = _mcp_prefix_completion(completer, text, text_lower)
    if prefix_completions is not None:
        return prefix_completions

    return _mcp_subcommand_completions(completer, text, text_lower)


def _card_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    if not text_lower.startswith("/card "):
        return None

    partial = text[len("/card ") :]
    current_token = _current_argument_token(partial)
    if current_token.startswith("-") or _completion_subject_is_finished(partial):
        return _flag_completions(current_token, _CARD_TOOL_FLAG_DESCRIPTIONS)
    return list(completer._complete_agent_card_files(partial))


def _agent_name_completions(
    completer: "AgentCompleter",
    partial: str,
) -> list[Completion]:
    partial = partial.removeprefix("@")
    return [
        Completion(
            agent,
            start_position=-len(partial),
            display=agent,
            display_meta=completer.agent_types.get(agent, AgentType.BASIC).value,
        )
        for agent in completer.agents
        if agent != completer.current_agent and starts_with_casefold(agent, partial)
    ]


def _agent_command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    if not text_lower.startswith("/agent "):
        return None

    partial = text[len("/agent ") :].lstrip()
    current_token = _current_argument_token(partial)
    if current_token.startswith("-") or _completion_subject_is_finished(partial):
        return _flag_completions(current_token, _AGENT_TOOL_FLAG_DESCRIPTIONS)
    return _agent_name_completions(completer, current_token)


def command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    """Return command-specific completions or ``None`` when unhandled."""
    providers: tuple[
        "Callable[[AgentCompleter, str, str], list[Completion] | None]",
        ...,
    ] = (
        _history_command_completions,
        _compact_command_completions,
        _prompt_command_completions,
        _attach_command_completions,
        _session_command_completions,
        _skills_command_completions,
        _cards_command_completions,
        _plugins_command_completions,
        _model_command_completions,
        _mcp_command_completions,
        _card_command_completions,
        _agent_command_completions,
    )
    for provider in providers:
        result = provider(completer, text, text_lower)
        if result is not None:
            return result
    return None
