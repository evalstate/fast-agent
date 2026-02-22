"""Reusable completion providers for prompt commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completion

from fast_agent.agents.agent_types import AgentType

if TYPE_CHECKING:
    from fast_agent.ui.prompt.completer import AgentCompleter


def command_completions(
    completer: "AgentCompleter",
    text: str,
    text_lower: str,
) -> list[Completion] | None:
    """Return command-specific completions or ``None`` when unhandled."""

    if text_lower.startswith("/history load "):
        partial = text[len("/history load ") :]
        return list(completer._complete_history_files(partial))

    if text_lower.startswith("/history rewind "):
        partial = text[len("/history rewind ") :]
        return list(completer._complete_history_rewind(partial))

    if text_lower.startswith("/history review "):
        partial = text[len("/history review ") :]
        return list(completer._complete_history_rewind(partial))

    if text_lower.startswith("/prompt load "):
        partial = text[len("/prompt load ") :]
        return list(completer._complete_history_files(partial))

    if text_lower.startswith("/history clear "):
        partial = text[len("/history clear ") :]
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
            if subcmd.startswith(partial.lower())
        ]

    if text_lower.startswith("/history webclear "):
        if not completer._current_agent_has_web_tools_enabled():
            return []
        partial = text[len("/history webclear ") :].strip()
        partial_lower = partial.lower()
        return [
            Completion(
                name,
                start_position=-len(partial),
                display=name,
                display_meta="Strip web metadata channels for this agent",
            )
            for name in sorted(completer.agents)
            if (not partial or name.lower().startswith(partial_lower))
        ]

    if text_lower.startswith("/resume "):
        partial = text[len("/resume ") :]
        return list(completer._complete_session_ids(partial))

    if text_lower.startswith("/session resume "):
        partial = text[len("/session resume ") :]
        return list(completer._complete_session_ids(partial))

    if text_lower.startswith("/session delete "):
        partial = text[len("/session delete ") :]
        results: list[Completion] = []
        if "all".startswith(partial.lower()):
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

    if text_lower.startswith("/session clear "):
        partial = text[len("/session clear ") :]
        results = []
        if "all".startswith(partial.lower()):
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

    if text_lower.startswith("/session pin "):
        remainder = text[len("/session pin ") :]
        parts = remainder.split(maxsplit=1) if remainder else []
        if not parts:
            results = [
                Completion(
                    option,
                    start_position=0,
                    display=option,
                    display_meta="Toggle session pin",
                )
                for option in ("on", "off")
            ]
            results.extend(list(completer._complete_session_ids("")))
            return results
        first = parts[0].lower()
        if first in {"on", "off"}:
            if len(parts) == 1 and not remainder.endswith(" "):
                return [
                    Completion(
                        option,
                        start_position=-len(first),
                        display=option,
                        display_meta="Toggle session pin",
                    )
                    for option in ("on", "off")
                    if option.startswith(first)
                ]
            suffix = parts[1] if len(parts) > 1 else ""
            start_position = -len(suffix) if suffix else 0
            return list(completer._complete_session_ids(suffix, start_position=start_position))
        return list(completer._complete_session_ids(remainder))

    if text_lower.startswith("/skills "):
        remainder = text[len("/skills ") :] or ""
        parts = remainder.split(maxsplit=1)
        subcommands = {
            "list": "List local skills",
            "add": "Install a skill",
            "remove": "Remove a local skill",
            "update": "Check or apply skill updates",
            "registry": "Set skills registry",
        }
        results = list(completer._complete_subcommands(parts, remainder, subcommands))
        if not parts or (len(parts) == 1 and not remainder.endswith(" ")):
            return results

        subcmd = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if subcmd in {"remove", "rm", "delete", "uninstall"}:
            results.extend(list(completer._complete_local_skill_names(argument)))
            return results
        if subcmd in {"update", "refresh", "upgrade"}:
            if "all".startswith(argument.lower()):
                results.append(
                    Completion(
                        "all",
                        start_position=-len(argument),
                        display="all",
                        display_meta="update all managed skills",
                    )
                )
            if "--force".startswith(argument.lower()):
                results.append(
                    Completion(
                        "--force",
                        start_position=-len(argument),
                        display="--force",
                        display_meta="overwrite local modifications",
                    )
                )
            if "--yes".startswith(argument.lower()):
                results.append(
                    Completion(
                        "--yes",
                        start_position=-len(argument),
                        display="--yes",
                        display_meta="confirm multi-skill apply",
                    )
                )
            results.extend(
                list(
                    completer._complete_local_skill_names(
                        argument,
                        managed_only=True,
                        include_indices=False,
                    )
                )
            )
            return results
        if subcmd in {"registry", "marketplace", "source"}:
            results.extend(list(completer._complete_skill_registries(argument)))
            return results
        return results

    if text_lower.startswith("/cards "):
        remainder = text[len("/cards ") :] or ""
        parts = remainder.split(maxsplit=1)
        subcommands = {
            "list": "List installed card packs",
            "add": "Install a card pack",
            "remove": "Remove an installed card pack",
            "update": "Check or apply card pack updates",
            "publish": "Publish local card pack changes",
            "registry": "Set card pack registry",
        }
        results = list(completer._complete_subcommands(parts, remainder, subcommands))
        if not parts or (len(parts) == 1 and not remainder.endswith(" ")):
            return results

        subcmd = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if subcmd in {"remove", "rm", "delete", "uninstall"}:
            results.extend(list(completer._complete_local_card_pack_names(argument)))
            return results
        if subcmd in {"update", "refresh", "upgrade"}:
            if "all".startswith(argument.lower()):
                results.append(
                    Completion(
                        "all",
                        start_position=-len(argument),
                        display="all",
                        display_meta="update all managed card packs",
                    )
                )
            if "--force".startswith(argument.lower()):
                results.append(
                    Completion(
                        "--force",
                        start_position=-len(argument),
                        display="--force",
                        display_meta="overwrite local modifications",
                    )
                )
            if "--yes".startswith(argument.lower()):
                results.append(
                    Completion(
                        "--yes",
                        start_position=-len(argument),
                        display="--yes",
                        display_meta="confirm multi-pack apply",
                    )
                )
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
        if subcmd in {"registry", "marketplace", "source"}:
            results.extend(list(completer._complete_card_registries(argument)))
            return results
        if subcmd in {"publish"}:
            current_token = ""
            if argument and not argument.endswith(" "):
                current_token = argument.split()[-1]
            elif argument.startswith("--"):
                current_token = argument

            if current_token.startswith("--") or not argument.strip():
                if "--no-push".startswith(current_token.lower()):
                    results.append(
                        Completion(
                            "--no-push",
                            start_position=-len(current_token),
                            display="--no-push",
                            display_meta="commit locally without push",
                        )
                    )
                if "--message".startswith(current_token.lower()):
                    results.append(
                        Completion(
                            "--message",
                            start_position=-len(current_token),
                            display="--message",
                            display_meta="set publish commit message",
                        )
                    )
                if "--temp-dir".startswith(current_token.lower()):
                    results.append(
                        Completion(
                            "--temp-dir",
                            start_position=-len(current_token),
                            display="--temp-dir",
                            display_meta="set parent directory for temp clone",
                        )
                    )
                if "--keep-temp".startswith(current_token.lower()):
                    results.append(
                        Completion(
                            "--keep-temp",
                            start_position=-len(current_token),
                            display="--keep-temp",
                            display_meta="retain temp clone after publish",
                        )
                    )

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
        return results

    if text_lower.startswith("/model "):
        remainder = text[len("/model ") :] or ""
        parts = remainder.split(maxsplit=1)
        subcommands: dict[str, str] = {
            "reasoning": (
                "Set reasoning effort (off/low/medium/high/max/xhigh or budgets like "
                "0/1024/16000/32000)"
            )
        }
        if completer._resolve_verbosity_values():
            subcommands["verbosity"] = "Set text verbosity (low/medium/high)"
        if completer._supports_web_search_setting():
            subcommands["web_search"] = "Set web search tool state (on/off/default)"
        if completer._supports_web_fetch_setting():
            subcommands["web_fetch"] = "Set web fetch tool state (on/off/default)"
        results = list(completer._complete_subcommands(parts, remainder, subcommands))
        if not parts or (len(parts) == 1 and not remainder.endswith(" ")):
            return results

        subcmd = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if subcmd == "reasoning":
            results.extend(
                Completion(
                    value,
                    start_position=-len(argument),
                    display=value,
                    display_meta="reasoning",
                )
                for value in completer._resolve_reasoning_values()
                if value.startswith(argument.lower())
            )
            return results
        if subcmd == "verbosity":
            results.extend(
                Completion(
                    value,
                    start_position=-len(argument),
                    display=value,
                    display_meta="verbosity",
                )
                for value in completer._resolve_verbosity_values()
                if value.startswith(argument.lower())
            )
            return results
        if subcmd == "web_search" and completer._supports_web_search_setting():
            results.extend(
                Completion(
                    value,
                    start_position=-len(argument),
                    display=value,
                    display_meta=subcmd,
                )
                for value in ("on", "off", "default")
                if value.startswith(argument.lower())
            )
            return results
        if subcmd == "web_fetch" and completer._supports_web_fetch_setting():
            results.extend(
                Completion(
                    value,
                    start_position=-len(argument),
                    display=value,
                    display_meta=subcmd,
                )
                for value in ("on", "off", "default")
                if value.startswith(argument.lower())
            )
            return results
        return results

    if text_lower.startswith("/mcp disconnect "):
        partial = text[len("/mcp disconnect ") :]
        attached: list[str] = []
        if completer.agent_provider is not None and completer.current_agent:
            try:
                agent = completer.agent_provider._agent(completer.current_agent)
                aggregator = getattr(agent, "aggregator", None)
                list_attached = getattr(aggregator, "list_attached_servers", None)
                if callable(list_attached):
                    attached = list_attached()
            except Exception:
                attached = []
        return [
            Completion(
                server,
                start_position=-len(partial),
                display=server,
                display_meta="attached mcp server",
            )
            for server in attached
            if server.lower().startswith(partial.lower())
        ]

    if text_lower.startswith("/mcp connect "):
        remainder = text[len("/mcp connect ") :]
        connect_flags = {
            "--name": "set attached server name",
            "--auth": "set bearer token for URL servers",
            "--timeout": "set startup timeout in seconds",
            "--oauth": "enable oauth flow",
            "--no-oauth": "disable oauth flow",
            "--reconnect": "force reconnect and refresh tools",
            "--no-reconnect": "disable reconnect-on-disconnect",
        }

        context, target_count, partial = completer._mcp_connect_context(remainder)

        if context in {"target", "new_token"} and target_count == 0:
            results = [completer._mcp_connect_target_hint(partial)]
            results.extend(list(completer._complete_configured_mcp_servers(partial)))
            return results

        if context == "new_token" and target_count > 0:
            return [
                Completion(
                    flag,
                    start_position=0,
                    display=flag,
                    display_meta=description,
                )
                for flag, description in connect_flags.items()
            ]

        if context == "flag" and target_count > 0:
            return [
                Completion(
                    flag,
                    start_position=-len(partial),
                    display=flag,
                    display_meta=description,
                )
                for flag, description in connect_flags.items()
                if flag.startswith(partial.lower())
            ]

    if text_lower.startswith("/mcp "):
        remainder = text[len("/mcp ") :]
        parts = remainder.split(maxsplit=1) if remainder else []
        subcommands = {
            "list": "List currently attached MCP servers",
            "connect": "Connect a new MCP server",
            "disconnect": "Disconnect an attached MCP server",
        }
        return list(completer._complete_subcommands(parts, remainder, subcommands))

    if text_lower.startswith("/history "):
        partial = text[len("/history ") :]
        subcommands = {
            "show": "Show history overview",
            "save": "Save history to a file",
            "load": "Load history from a file",
            "clear": "Clear history (all or last)",
            "rewind": "Rewind to a previous user turn",
            "review": "Review a previous user turn in full",
            "fix": "Remove the last pending tool call",
        }
        if completer._current_agent_has_web_tools_enabled():
            subcommands["webclear"] = "Strip web tool/citation metadata channels"
        return [
            Completion(
                subcmd,
                start_position=-len(partial),
                display=subcmd,
                display_meta=description,
            )
            for subcmd, description in subcommands.items()
            if subcmd.startswith(partial.lower())
        ]

    if text_lower.startswith("/session "):
        partial = text[len("/session ") :]
        subcommands = {
            "delete": "Delete a session (or all)",
            "pin": "Pin or unpin the current session",
            "clear": "Alias for delete",
            "list": "List recent sessions",
            "new": "Create a new session",
            "resume": "Resume a session",
            "title": "Set session title",
            "fork": "Fork current session",
        }
        return [
            Completion(
                subcmd,
                start_position=-len(partial),
                display=subcmd,
                display_meta=description,
            )
            for subcmd, description in subcommands.items()
            if subcmd.startswith(partial.lower())
        ]

    if text_lower.startswith("/card "):
        partial = text[len("/card ") :]
        return list(completer._complete_agent_card_files(partial))

    if text_lower.startswith("/agent "):
        partial = text[len("/agent ") :].lstrip()
        if partial.startswith("@"):
            partial = partial[1:]
        return [
            Completion(
                agent,
                start_position=-len(partial),
                display=agent,
                display_meta=completer.agent_types.get(agent, AgentType.BASIC).value,
            )
            for agent in completer.agents
            if agent != completer.current_agent and agent.lower().startswith(partial.lower())
        ]

    return None
