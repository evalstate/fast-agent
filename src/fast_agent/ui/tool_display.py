from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Mapping

from rich.syntax import Syntax
from rich.text import Text

from fast_agent.ui import console
from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from fast_agent.mcp.skybridge import SkybridgeServerConfig
    from fast_agent.ui.console_display import ConsoleDisplay


class ToolDisplay:
    """Encapsulates rendering logic for tool calls and results."""

    def __init__(self, display: "ConsoleDisplay") -> None:
        self._display = display

    @property
    def _markup(self) -> bool:
        return self._display._markup

    def show_tool_result(
        self,
        result: CallToolResult,
        *,
        name: str | None = None,
        tool_name: str | None = None,
        skybridge_config: "SkybridgeServerConfig | None" = None,
        timing_ms: float | None = None,
    ) -> None:
        """Display a tool result in the console."""
        config = self._display.config
        if config and not config.logger.show_tools:
            return

        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        content = result.content
        structured_content = getattr(result, "structuredContent", None)
        has_structured = structured_content is not None

        is_skybridge_tool = False
        skybridge_resource_uri: str | None = None
        if has_structured and tool_name and skybridge_config:
            for tool_cfg in skybridge_config.tools:
                if tool_cfg.tool_name == tool_name and tool_cfg.is_valid:
                    is_skybridge_tool = True
                    skybridge_resource_uri = tool_cfg.resource_uri
                    break

        if result.isError:
            status = "ERROR"
        else:
            if len(content) == 0:
                status = "No Content"
            elif len(content) == 1 and is_text_content(content[0]):
                text_content = get_text(content[0])
                char_count = len(text_content) if text_content else 0
                status = f"Text Only {char_count} chars"
            else:
                text_count = sum(1 for item in content if is_text_content(item))
                if text_count == len(content):
                    status = f"{len(content)} Text Blocks" if len(content) > 1 else "1 Text Block"
                else:
                    status = (
                        f"{len(content)} Content Blocks" if len(content) > 1 else "1 Content Block"
                    )

        channel = getattr(result, "transport_channel", None)
        bottom_metadata_items: list[str] = []
        if channel:
            if channel == "post-json":
                transport_info = "HTTP (JSON-RPC)"
            elif channel == "post-sse":
                transport_info = "HTTP (SSE)"
            elif channel == "get":
                transport_info = "Legacy SSE"
            elif channel == "resumption":
                transport_info = "Resumption"
            elif channel == "stdio":
                transport_info = "STDIO"
            else:
                transport_info = channel.upper()

            bottom_metadata_items.append(transport_info)

        # Use timing from FAST_AGENT_TOOL_TIMING (passed as parameter)
        if timing_ms is not None:
            # Convert ms to seconds for display
            timing_seconds = timing_ms / 1000.0
            bottom_metadata_items.append(self._display._format_elapsed(timing_seconds))

        if has_structured:
            bottom_metadata_items.append("Structured ■")

        bottom_metadata = bottom_metadata_items or None
        right_info = f"[dim]tool result - {status}[/dim]"

        if has_structured:
            config_map = MESSAGE_CONFIGS[MessageType.TOOL_RESULT]
            block_color = "red" if result.isError else config_map["block_color"]
            arrow = config_map["arrow"]
            arrow_style = config_map["arrow_style"]
            left = f"[{block_color}]▎[/{block_color}][{arrow_style}]{arrow}[/{arrow_style}]"
            if name:
                name_color = block_color if not result.isError else "red"
                left += f" [{name_color}]{name}[/{name_color}]"

            self._display._create_combined_separator_status(left, right_info)

            self._display._display_content(
                content,
                True,
                result.isError,
                MessageType.TOOL_RESULT,
                check_markdown_markers=False,
            )

            console.console.print()
            total_width = console.console.size.width

            if is_skybridge_tool:
                resource_label = (
                    f"skybridge resource: {skybridge_resource_uri}"
                    if skybridge_resource_uri
                    else "skybridge resource"
                )
                prefix = Text("─| ")
                prefix.stylize("dim")
                resource_text = Text(resource_label, style="magenta")
                suffix = Text(" |")
                suffix.stylize("dim")

                separator_line = Text()
                separator_line.append_text(prefix)
                separator_line.append_text(resource_text)
                separator_line.append_text(suffix)
                remaining = total_width - separator_line.cell_len
                if remaining > 0:
                    separator_line.append("─" * remaining, style="dim")
                console.console.print(separator_line, markup=self._markup)
                console.console.print()

                json_str = json.dumps(structured_content, indent=2)
                syntax_obj = Syntax(
                    json_str,
                    "json",
                    theme=self._display.code_style,
                    background_color="default",
                )
                console.console.print(syntax_obj, markup=self._markup)
            else:
                prefix = Text("─| ")
                prefix.stylize("dim")
                suffix = Text(" |")
                suffix.stylize("dim")
                available = max(0, total_width - prefix.cell_len - suffix.cell_len)

                metadata_text = self._display._format_bottom_metadata(
                    bottom_metadata_items,
                    None,
                    config_map["highlight_color"],
                    max_width=available,
                )

                line = Text()
                line.append_text(prefix)
                line.append_text(metadata_text)
                line.append_text(suffix)
                remaining = total_width - line.cell_len
                if remaining > 0:
                    line.append("─" * remaining, style="dim")
                console.console.print(line, markup=self._markup)
                console.console.print()
        else:
            self._display.display_message(
                content=content,
                message_type=MessageType.TOOL_RESULT,
                name=name,
                right_info=right_info,
                bottom_metadata=bottom_metadata,
                is_error=result.isError,
                truncate_content=True,
            )

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        *,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Display a tool call header and body."""
        config = self._display.config
        if config and not config.logger.show_tools:
            return

        tool_args = tool_args or {}
        metadata = metadata or {}

        right_info = f"[dim]tool request - {tool_name}[/dim]"
        content: Any = tool_args
        pre_content: Text | None = None
        truncate_content = True

        if metadata.get("variant") == "shell":
            bottom_items = list()
            max_item_length = 50
            command = metadata.get("command") or tool_args.get("command")

            command_text = Text()
            if command and isinstance(command, str):
                command_text.append("$ ", style="magenta")
                command_text.append(command, style="white")
            else:
                command_text.append("$ ", style="magenta")
                command_text.append("(no shell command provided)", style="dim")

            content = command_text

            shell_name = metadata.get("shell_name") or "shell"
            shell_path = metadata.get("shell_path")
            if shell_path:
                bottom_items.append(str(shell_path))

            right_parts: list[str] = []
            if shell_path and shell_path != shell_name:
                right_parts.append(f"{shell_name} ({shell_path})")
            elif shell_name:
                right_parts.append(shell_name)

            right_info = f"[dim]{' | '.join(right_parts)}[/dim]" if right_parts else ""
            truncate_content = False

            working_dir_display = metadata.get("working_dir_display") or metadata.get("working_dir")
            if working_dir_display:
                bottom_items.append(f"cwd: {working_dir_display}")

            timeout_seconds = metadata.get("timeout_seconds")
            warning_interval = metadata.get("warning_interval_seconds")

            if timeout_seconds and warning_interval:
                bottom_items.append(
                    f"timeout: {timeout_seconds}s, warning every {warning_interval}s"
                )

        self._display.display_message(
            content=content,
            message_type=MessageType.TOOL_CALL,
            name=name,
            pre_content=pre_content,
            right_info=right_info,
            bottom_metadata=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
            truncate_content=truncate_content,
        )

    async def show_tool_update(self, updated_server: str, *, agent_name: str | None = None) -> None:
        """Show a background tool update notification."""
        config = self._display.config
        if config and not config.logger.show_tools:
            return

        try:
            from prompt_toolkit.application.current import get_app

            app = get_app()
            from fast_agent.ui import notification_tracker

            notification_tracker.add_tool_update(updated_server)
            app.invalidate()
        except Exception:  # noqa: BLE001
            if agent_name:
                left = f"[magenta]▎[/magenta][dim magenta]▶[/dim magenta] [magenta]{agent_name}[/magenta]"
            else:
                left = "[magenta]▎[/magenta][dim magenta]▶[/dim magenta]"

            right = f"[dim]{updated_server}[/dim]"
            self._display._create_combined_separator_status(left, right)

            message = f"Updating tools for server {updated_server}"
            console.console.print(message, style="dim", markup=self._markup)

            console.console.print()
            console.console.print("─" * console.console.size.width, style="dim")
            console.console.print()

    @staticmethod
    def summarize_skybridge_configs(
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Convert Skybridge configs into display-ready structures."""
        server_rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        warning_seen: set[str] = set()

        if not configs:
            return server_rows, warnings

        def add_warning(message: str) -> None:
            formatted = message.strip()
            if not formatted:
                return
            if formatted not in warning_seen:
                warnings.append(formatted)
                warning_seen.add(formatted)

        for server_name in sorted(configs.keys()):
            config = configs.get(server_name)
            if not config:
                continue
            resources = list(config.ui_resources or [])
            has_skybridge_signal = bool(
                config.enabled or resources or config.tools or config.warnings
            )
            if not has_skybridge_signal:
                continue

            valid_resource_count = sum(1 for resource in resources if resource.is_skybridge)

            server_rows.append(
                {
                    "server_name": server_name,
                    "config": config,
                    "resources": resources,
                    "valid_resource_count": valid_resource_count,
                    "total_resource_count": len(resources),
                    "active_tools": [
                        {
                            "name": tool.display_name,
                            "template": str(tool.template_uri) if tool.template_uri else None,
                        }
                        for tool in config.tools
                        if tool.is_valid
                    ],
                    "enabled": config.enabled,
                }
            )

            for warning in config.warnings:
                message = warning.strip()
                if not message:
                    continue
                if not message.startswith(server_name):
                    message = f"{server_name} {message}"
                add_warning(message)

        return server_rows, warnings

    def show_skybridge_summary(
        self,
        agent_name: str,
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> None:
        """Display aggregated Skybridge status."""
        server_rows, warnings = self.summarize_skybridge_configs(configs)

        if not server_rows and not warnings:
            return

        heading = "[dim]OpenAI Apps SDK ([/dim][cyan]skybridge[/cyan][dim]) detected:[/dim]"
        console.console.print()
        console.console.print(heading, markup=self._markup)

        if not server_rows:
            console.console.print("[dim]  ● none detected[/dim]", markup=self._markup)
        else:
            for row in server_rows:
                server_name = row["server_name"]
                resource_count = row["valid_resource_count"]
                tool_infos = row["active_tools"]
                enabled = row["enabled"]

                tool_count = len(tool_infos)
                tool_word = "tool" if tool_count == 1 else "tools"
                resource_word = (
                    "skybridge resource" if resource_count == 1 else "skybridge resources"
                )
                tool_segment = f"[cyan]{tool_count}[/cyan][dim] {tool_word}[/dim]"
                resource_segment = f"[cyan]{resource_count}[/cyan][dim] {resource_word}[/dim]"
                name_style = "cyan" if enabled else "yellow"
                status_suffix = "" if enabled else "[dim] (issues detected)[/dim]"

                console.console.print(
                    f"[dim]  ● [/dim][{name_style}]{server_name}[/{name_style}]{status_suffix}"
                    f"[dim] — [/dim]{tool_segment}[dim], [/dim]{resource_segment}",
                    markup=self._markup,
                )

                if tool_infos:
                    for tool in tool_infos:
                        template_info = (
                            f" [dim]({tool['template']})[/dim]" if tool["template"] else ""
                        )
                        console.console.print(
                            f"[dim]     · [/dim]{tool['name']}{template_info}", markup=self._markup
                        )
                else:
                    console.console.print("[dim]     · no active tools[/dim]", markup=self._markup)

        if warnings:
            console.console.print()
            console.console.print(
                "[yellow]Warnings[/yellow]",
                markup=self._markup,
            )
            for warning in warnings:
                console.console.print(f"[yellow]- {warning}[/yellow]", markup=self._markup)
