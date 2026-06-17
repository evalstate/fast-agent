"""Slash command discovery rendering helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeVar

from fast_agent.commands.command_catalog import (
    COMMAND_SPECS,
    CommandActionSpec,
    get_command_action_spec,
    get_command_spec,
)
from fast_agent.commands.mcp_command_intents import MCP_TOP_LEVEL_ACTION_DESCRIPTIONS
from fast_agent.commands.metadata_labels import (
    metadata_argument_label,
    metadata_option_label,
)
from fast_agent.commands.session_export_help import (
    SESSION_EXPORT_EXAMPLES,
    build_session_export_action_detail,
)
from fast_agent.commands.session_summaries import FULL_SESSION_USAGE
from fast_agent.commands.shared_command_intents import (
    HISTORY_COMMAND_COMPLETION_DESCRIPTIONS,
    SESSION_COMMAND_COMPLETION_DESCRIPTIONS,
)
from fast_agent.commands.summary_utils import optional_string
from fast_agent.utils.action_normalization import is_help_flag, normalize_action_token
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.markdown import escape_markdown_text, markdown_code_span
from fast_agent.utils.numeric import finite_number_or_none
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

SCHEMA_VERSION = "1"
_DISCOVERY_USAGE = "Usage: /commands [<command> [<action>]] [--json]"


class ActionArgumentPayload(TypedDict):
    name: str
    summary: str
    value_name: str | None
    required: bool


class ActionOptionPayload(TypedDict):
    name: str
    summary: str
    value_name: str | None
    aliases: list[str]


MetadataItemPayload = TypeVar("MetadataItemPayload", ActionArgumentPayload, ActionOptionPayload)


class ActionPayload(TypedDict):
    name: str
    summary: str
    aliases: NotRequired[list[str]]
    usage: NotRequired[str]
    examples: NotRequired[list[str]]
    arguments: NotRequired[list[ActionArgumentPayload]]
    options: NotRequired[list[ActionOptionPayload]]
    notes: NotRequired[list[str]]


type CommandIndexAction = str | ActionPayload


class CommandIndexEntry(TypedDict):
    name: str
    summary: str
    usage: str
    actions: list[CommandIndexAction]
    examples: list[str]


class CommandIndexPayload(TypedDict):
    name: str
    summary: str
    usage: str
    actions: list[ActionPayload]
    examples: list[str]


class CommandDetailEntry(TypedDict):
    name: str
    summary: str
    usage: str
    actions: list[ActionPayload]
    examples: list[str]


@dataclass(frozen=True, slots=True)
class DiscoveryRequest:
    """Parsed request for /commands rendering."""

    command_name: str | None
    action_name: str | None
    as_json: bool


def parse_commands_discovery_arguments(arguments: str) -> DiscoveryRequest:
    """Parse /commands arguments into a request object."""

    trimmed = strip_to_none(arguments)
    if trimmed is None:
        return DiscoveryRequest(command_name=None, action_name=None, as_json=False)

    try:
        tokens = split_commandline(trimmed, syntax="posix")
    except ValueError as exc:
        raise ValueError(f"Invalid /commands arguments: {exc}") from exc

    command_name: str | None = None
    action_name: str | None = None
    as_json = False

    for token in tokens:
        lowered = normalize_action_token(token)
        if lowered == "--json":
            as_json = True
            continue
        if lowered.startswith("--"):
            raise ValueError(f"Unknown /commands option: {token}")
        command_name, action_name = _assign_discovery_positional(
            command_name,
            action_name,
            lowered,
        )

    return DiscoveryRequest(command_name=command_name, action_name=action_name, as_json=as_json)


def _assign_discovery_positional(
    command_name: str | None,
    action_name: str | None,
    token: str,
) -> tuple[str | None, str | None]:
    if not token:
        raise ValueError(_DISCOVERY_USAGE)
    if command_name is None:
        return token, action_name
    if action_name is None:
        return command_name, token
    raise ValueError(_DISCOVERY_USAGE)


def _action_payload_from_catalog(action: CommandActionSpec) -> ActionPayload:
    payload: ActionPayload = {
        "name": action.action,
        "summary": action.help,
        "aliases": list(action.aliases),
    }
    if action.usage:
        payload["usage"] = action.usage
    if action.examples:
        payload["examples"] = list(action.examples)
    if action.arguments:
        payload["arguments"] = [
            {
                "name": item.name,
                "summary": item.summary,
                "value_name": item.value_name,
                "required": item.required,
            }
            for item in action.arguments
        ]
    if action.options:
        payload["options"] = [
            {
                "name": item.name,
                "summary": item.summary,
                "value_name": item.value_name,
                "aliases": list(item.aliases),
            }
            for item in action.options
        ]
    if action.notes:
        payload["notes"] = list(action.notes)
    return payload


def _session_export_action_payload() -> ActionPayload:
    detail = build_session_export_action_detail()
    return {
        "name": detail["name"],
        "summary": detail["summary"],
        "usage": detail["usage"],
        "examples": detail["examples"],
        "arguments": detail["arguments"],
        "options": detail["options"],
        "notes": detail["notes"],
    }


def _usage_without_label(usage: str) -> str:
    return strip_to_none(usage.removeprefix("Usage: ")) or ""


def _session_action_payload(
    name: str,
    *,
    usage: str | None = None,
    aliases: list[str] | None = None,
) -> ActionPayload:
    payload: ActionPayload = {
        "name": name,
        "summary": SESSION_COMMAND_COMPLETION_DESCRIPTIONS[name],
    }
    if usage is not None:
        payload["usage"] = usage
    if aliases:
        payload["aliases"] = aliases
    return payload


def _session_detail_entry() -> CommandIndexEntry:
    return {
        "name": "session",
        "summary": "List, manage, or export sessions",
        "usage": _usage_without_label(FULL_SESSION_USAGE),
        "actions": [
            _session_action_payload("list", usage="/session list"),
            _session_action_payload("new", usage="/session new [title]"),
            _session_action_payload("resume", usage="/session resume [id|number]"),
            _session_action_payload("title", usage="/session title <text>"),
            _session_action_payload("fork", usage="/session fork [title]"),
            _session_action_payload(
                "delete",
                usage="/session delete <id|number|all>",
                aliases=["clear"],
            ),
            _session_action_payload("pin", usage="/session pin [on|off|id|number]"),
            _session_export_action_payload(),
            {"name": "help", "summary": "Show session usage"},
        ],
        "examples": [
            "/session list",
            "/session resume 1",
            *SESSION_EXPORT_EXAMPLES[:2],
        ],
    }


def _simple_command_entry(
    name: str,
    *,
    summary: str,
    usage: str,
    examples: list[str],
) -> CommandIndexEntry:
    return {
        "name": name,
        "summary": summary,
        "usage": usage,
        "actions": [],
        "examples": examples,
    }


def _discovery_top_level_catalog() -> tuple[CommandIndexEntry, ...]:
    families: list[CommandIndexEntry] = [
        {
            "name": spec.command,
            "summary": spec.summary,
            "usage": spec.usage,
            "actions": [action.action for action in spec.actions],
            "examples": list(spec.examples),
        }
        for spec in COMMAND_SPECS
    ]

    extras: tuple[CommandIndexEntry, ...] = (
        _simple_command_entry(
            "commands",
            summary="Command map + help",
            usage="/commands [<command> [<action>]] [--json]",
            examples=[
                "/commands",
                "/commands skills",
                "/commands skills add",
                "/commands --json",
            ],
        ),
        _simple_command_entry(
            "agent",
            summary="Attach, detach, or inspect an existing agent as a tool",
            usage="/agent <name> [--tool [remove]|--dump]",
            examples=["/agent reviewer --tool", "/agent reviewer --dump"],
        ),
        _simple_command_entry(
            "attach",
            summary="Stage file or URL attachments for the next prompt",
            usage="/attach [clear|path|url ...]",
            examples=["/attach README.md", "/attach clear"],
        ),
        _simple_command_entry(
            "card",
            summary="Load an agent card, optionally as a tool",
            usage="/card <path> [--tool [remove]]",
            examples=["/card sizer.md", "/card sizer.md --tool"],
        ),
        {
            "name": "compact",
            "summary": "Compact history into a checkpoint summary",
            "usage": "/compact [preview|prompt|<instructions>]",
            "actions": [
                {"name": "preview", "summary": "show what compaction would keep (no model call)"},
                {"name": "prompt", "summary": "show the active compaction prompt"},
            ],
            "examples": ["/compact", "/compact preview", "/compact focus on the DB migration"],
        },
        _simple_command_entry(
            "connect",
            summary="Attach a runtime MCP server",
            usage="/connect <target> [--name <server>] [options]",
            examples=["/connect filesystem --name docs"],
        ),
        _simple_command_entry(
            "fast",
            summary="Shortcut for /model fast",
            usage="/fast [on|off|flex|status]",
            examples=["/fast flex"],
        ),
        {
            "name": "history",
            "summary": "Inspect, save, load, or edit chat history",
            "usage": "/history [list|show|detail|review|save|load|clear|rewind|fix|webclear] [args]",
            "actions": [
                {"name": name, "summary": summary}
                for name, summary in HISTORY_COMMAND_COMPLETION_DESCRIPTIONS.items()
            ],
            "examples": ["/history", "/history detail 3", "/history save history.json"],
        },
        _simple_command_entry(
            "load",
            summary="Shortcut for /history load",
            usage="/load <file>",
            examples=["/load history.json"],
        ),
        _simple_command_entry(
            "mcpstatus",
            summary="Show MCP server connection status",
            usage="/mcpstatus",
            examples=["/mcpstatus"],
        ),
        {
            "name": "mcp",
            "summary": "Runtime MCP control",
            "usage": "/mcp [list|connect|disconnect|reconnect] [args]",
            "actions": [
                {"name": name, "summary": summary}
                for name, summary in MCP_TOP_LEVEL_ACTION_DESCRIPTIONS.items()
            ],
            "examples": ["/mcp list", "/mcp connect <target>", "/mcp disconnect <server>"],
        },
        _session_detail_entry(),
        _simple_command_entry(
            "tools",
            summary="List callable tools",
            usage="/tools",
            examples=["/tools"],
        ),
        _simple_command_entry(
            "prompts",
            summary="List prompt templates",
            usage="/prompts",
            examples=["/prompts"],
        ),
        {
            "name": "prompt",
            "summary": "Load or select prompt templates",
            "usage": "/prompt [load|<number|name>]",
            "actions": [
                {"name": "load", "summary": "load prompt template by file"},
            ],
            "examples": ["/prompts", "/prompt load prompt.md"],
        },
        _simple_command_entry(
            "reload",
            summary="Reload agent definitions",
            usage="/reload",
            examples=["/reload"],
        ),
        _simple_command_entry(
            "resume",
            summary="Shortcut for /session resume",
            usage="/resume [session-id|number]",
            examples=["/resume 1"],
        ),
        _simple_command_entry(
            "save",
            summary="Shortcut for /history save",
            usage="/save [file]",
            examples=["/save history.json"],
        ),
        _simple_command_entry(
            "usage",
            summary="Token/cost summary",
            usage="/usage",
            examples=["/usage"],
        ),
        _simple_command_entry(
            "system",
            summary="Show resolved instruction",
            usage="/system",
            examples=["/system"],
        ),
        _simple_command_entry(
            "markdown",
            summary="Show markdown buffer",
            usage="/markdown",
            examples=["/markdown"],
        ),
    )

    families.extend(extras)
    families.sort(key=lambda item: item["name"])
    return tuple(families)


def command_discovery_names() -> tuple[str, ...]:
    """Return discoverable command names for /commands."""

    return tuple(item["name"] for item in _discovery_top_level_catalog())


def _structured_action_payloads(actions: list[CommandIndexAction]) -> list[ActionPayload]:
    return [
        action if not isinstance(action, str) else {"name": action, "summary": ""}
        for action in actions
    ]


def _command_index_payload(entry: CommandIndexEntry) -> CommandIndexPayload:
    return {
        "name": entry["name"],
        "summary": entry["summary"],
        "usage": entry["usage"],
        "actions": _structured_action_payloads(entry["actions"]),
        "examples": entry["examples"],
    }


def _build_command_detail(name: str) -> CommandDetailEntry | None:
    normalized = normalize_action_token(name)
    spec = get_command_spec(normalized)
    if spec is not None:
        return {
            "name": spec.command,
            "summary": spec.summary,
            "usage": spec.usage,
            "actions": [_action_payload_from_catalog(action) for action in spec.actions],
            "examples": list(spec.examples),
        }

    for entry in _discovery_top_level_catalog():
        if entry["name"] != normalized:
            continue
        return {
            "name": entry["name"],
            "summary": entry["summary"],
            "usage": entry["usage"],
            "actions": _structured_action_payloads(entry["actions"]),
            "examples": entry["examples"],
        }
    return None


def _build_command_action_detail(command_name: str, action_name: str) -> ActionPayload | None:
    detail = _build_command_detail(command_name)
    if detail is None:
        return None
    return _find_action_detail(detail, action_name)


def _find_action_detail(detail: CommandDetailEntry, action_name: str) -> ActionPayload | None:
    normalized_action = normalize_action_token(action_name)
    for action_map in detail["actions"]:
        if _action_payload_matches(action_map, normalized_action):
            return action_map

    action_spec = get_command_action_spec(detail["name"], normalized_action)
    if action_spec is None:
        return None
    return _action_payload_from_catalog(action_spec)


def _action_payload_matches(action: ActionPayload, normalized_action: str) -> bool:
    if not normalized_action:
        return False

    names = (action.get("name"), *action.get("aliases", ()))
    return any(
        normalize_action_token(name) == normalized_action
        for name in names
        if isinstance(name, str) and name
    )


def _append_labeled_metadata_item(
    lines: list[str],
    *,
    indent: str,
    label: str,
    summary: object,
) -> None:
    summary_text = _metadata_summary_text(summary)
    if summary_text is not None:
        lines.append(f"{indent}  - {label} — {escape_markdown_text(summary_text)}")
    else:
        lines.append(f"{indent}  - {label}")


def _metadata_summary_text(summary: object) -> str | None:
    text = optional_string(summary)
    if text is not None:
        return text

    number = finite_number_or_none(summary)
    if number is not None:
        return strip_to_none(str(number))

    return None


def _render_labeled_metadata(
    lines: list[str],
    *,
    items: Sequence[MetadataItemPayload],
    title: str,
    label_for_item: Callable[[MetadataItemPayload], str | None],
    indent: str,
) -> None:
    labeled_items = [(label, item) for item in items if (label := label_for_item(item)) is not None]
    if not labeled_items:
        return

    lines.append(f"{indent}- {title}:")
    for label, item in labeled_items:
        _append_labeled_metadata_item(
            lines,
            indent=indent,
            label=label,
            summary=item.get("summary"),
        )


def _render_argument_label(argument_map: ActionArgumentPayload) -> str | None:
    return metadata_argument_label(argument_map)


def _render_option_label(option_map: ActionOptionPayload) -> str | None:
    return metadata_option_label(option_map)


def _render_argument_metadata(
    lines: list[str],
    arguments: list[ActionArgumentPayload],
    *,
    indent: str,
) -> None:
    _render_labeled_metadata(
        lines,
        items=arguments,
        title="arguments",
        label_for_item=_render_argument_label,
        indent=indent,
    )


def _render_option_metadata(
    lines: list[str],
    options: list[ActionOptionPayload],
    *,
    indent: str,
) -> None:
    _render_labeled_metadata(
        lines,
        items=options,
        title="options",
        label_for_item=_render_option_label,
        indent=indent,
    )


def _render_action_metadata(
    lines: list[str],
    action_map: ActionPayload,
    *,
    indent: str,
    include_usage: bool = True,
) -> None:
    usage = action_map.get("usage")
    usage_text = optional_string(usage)
    if usage_text is not None and include_usage:
        lines.append(f"{indent}- usage: {markdown_code_span(usage_text)}")

    arguments = action_map.get("arguments")
    if isinstance(arguments, list) and arguments:
        _render_argument_metadata(lines, arguments, indent=indent)

    options = action_map.get("options")
    if isinstance(options, list) and options:
        _render_option_metadata(lines, options, indent=indent)

    notes = action_map.get("notes")
    if isinstance(notes, list) and notes:
        note_lines = [
            f"{indent}  - {escape_markdown_text(note_text)}"
            for note in notes
            if (note_text := optional_string(note)) is not None
        ]
        if note_lines:
            lines.append(f"{indent}- notes:")
            lines.extend(note_lines)

    examples = action_map.get("examples")
    if isinstance(examples, list):
        lines.extend(
            f"{indent}- example: {markdown_code_span(example_text)}"
            for example in examples
            if (example_text := optional_string(example)) is not None
        )


def _render_action_list_item(action_map: ActionPayload) -> str | None:
    action_name = strip_to_none(action_map["name"])
    if action_name is None:
        return None

    aliases = action_map.get("aliases")
    alias_text = ""
    if isinstance(aliases, list) and aliases:
        alias_labels = [
            escape_markdown_text(alias_value)
            for alias in aliases
            if (alias_value := strip_to_none(str(alias))) is not None
        ]
        if alias_labels:
            alias_text = " (aliases: " + ", ".join(alias_labels) + ")"

    action_summary = strip_to_none(action_map["summary"])
    label = markdown_code_span(action_name)
    if action_summary is not None:
        return f"- {label} — {escape_markdown_text(action_summary)}{alias_text}"
    return f"- {label}{alias_text}"


def render_commands_index_markdown(*, command_names: Collection[str] | None = None) -> str:
    """Render markdown for /commands index."""

    allowed = _normalized_command_name_filter(command_names)
    lines = ["# commands", "", "Command map:"]
    for entry in _discovery_top_level_catalog():
        name = entry["name"]
        if allowed is not None and name not in allowed:
            continue

        lines.append(
            f"- {markdown_code_span(f'/{name}')} — {escape_markdown_text(entry['summary'])}"
        )
        actions = entry["actions"]
        if not actions:
            continue

        action_names: list[str] = []
        for action in actions:
            if isinstance(action, str):
                action_names.append(action)
                continue
            action_name = action.get("name")
            if action_name:
                action_names.append(action_name)
        if action_names:
            lines.append(f"  - {', '.join(action_names)}")

    lines.extend(
        [
            "",
            "Next:",
            "- `/commands <name>` for detailed help",
            "- `/commands <name> <action>` for action-level help",
            "- `/commands --json` for machine-readable map",
        ]
    )
    return "\n".join(lines)


def render_command_detail_markdown(command_name: str, action_name: str | None = None) -> str | None:
    """Render markdown for /commands <name> [<action>]."""

    if action_name is not None:
        return _render_command_action_detail_markdown(command_name, action_name)

    detail = _build_command_detail(command_name)
    if detail is None:
        return None
    return _render_command_detail_markdown(detail)


def _render_command_action_detail_markdown(
    command_name: str,
    action_name: str,
) -> str | None:
    detail = _build_command_detail(command_name)
    if detail is None:
        return None

    action = _find_action_detail(detail, action_name)
    if action is None:
        return None

    action_heading = action.get("name", action_name)
    default_summary = f"{markdown_code_span('/' + detail['name'])} action"
    json_command = f"/commands {detail['name']} {action_heading} --json"
    lines = [
        f"# commands {detail['name']} {action_heading}",
        "",
        escape_markdown_text(strip_to_none(action.get("summary", "")) or "") or default_summary,
    ]
    usage = action.get("usage")
    usage_text = optional_string(usage)
    if usage_text is not None:
        lines.extend(["", f"Usage: {markdown_code_span(usage_text)}"])
    _render_action_metadata(lines, action, indent="", include_usage=False)
    lines.extend(
        [
            "",
            f"JSON: {markdown_code_span(json_command)}",
        ]
    )
    return "\n".join(lines)


def _render_command_detail_markdown(detail: CommandDetailEntry) -> str:
    json_command = f"/commands {detail['name']} --json"
    lines = [
        f"# commands {detail['name']}",
        "",
        escape_markdown_text(detail["summary"]),
        "",
        f"Usage: {markdown_code_span(detail['usage'])}",
    ]
    actions = detail["actions"]
    if actions:
        lines.extend(["", "Actions:"])
        for action_map in actions:
            action_line = _render_action_list_item(action_map)
            if action_line is None:
                continue
            lines.append(action_line)
            _render_action_metadata(lines, action_map, indent="  ")

    examples = detail["examples"]
    if examples:
        lines.extend(["", "Examples:"])
        lines.extend(f"- {markdown_code_span(example)}" for example in examples)

    lines.extend(
        [
            "",
            f"JSON: {markdown_code_span(json_command)}",
        ]
    )
    return "\n".join(lines)


def _render_discovery_json(kind: str, **payload: object) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            **payload,
        },
        indent=2,
        sort_keys=True,
    )


def render_commands_json(
    *,
    command_name: str | None = None,
    action_name: str | None = None,
    command_names: Collection[str] | None = None,
) -> str:
    """Render JSON payload for /commands outputs."""

    allowed = _normalized_command_name_filter(command_names)

    if command_name is None:
        commands = [
            _command_index_payload(item)
            for item in _discovery_top_level_catalog()
            if allowed is None or item["name"] in allowed
        ]
        return _render_discovery_json("command_index", commands=commands)

    detail = _build_command_detail(command_name)
    if detail is None:
        return _render_discovery_json(
            "error",
            error=f"Unknown command: {command_name}",
            suggestions=command_discovery_names(),
        )

    if allowed is not None and detail["name"] not in allowed:
        return _render_discovery_json(
            "error",
            error=f"Command '/{detail['name']}' is not available in this context.",
        )

    if action_name is None:
        return _render_discovery_json("command_detail", command=detail)

    action = _find_action_detail(detail, action_name)
    if action is None:
        return _render_discovery_json(
            "error",
            error=f"Unknown action '{action_name}' for '/{detail['name']}'.",
        )

    return _render_discovery_json(
        "command_action_detail",
        command={
            "name": detail["name"],
            "summary": detail["summary"],
            "usage": detail["usage"],
        },
        action=action,
    )


def render_direct_command_help(command_name: str, arguments: str | None) -> str | None:
    """Render action-specific help for direct slash commands when requested."""

    trimmed = strip_to_none(arguments)
    if trimmed is None:
        return None

    try:
        tokens = split_commandline(trimmed, syntax="posix")
    except ValueError:
        return None

    if not tokens:
        return None

    first = normalize_action_token(tokens[0])
    if is_help_flag(first):
        return render_command_detail_markdown(command_name)

    if len(tokens) >= 2 and is_help_flag(tokens[-1]):
        return render_command_detail_markdown(command_name, first)

    return None


def _normalized_command_name_filter(
    command_names: Collection[str] | None,
) -> set[str] | None:
    if command_names is None:
        return None
    return {normalized for name in command_names if (normalized := normalize_action_token(name))}
