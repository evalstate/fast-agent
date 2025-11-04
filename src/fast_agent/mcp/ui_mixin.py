"""
MCP UI Mixin - Clean mixin pattern for MCP UI functionality.

This module provides a mixin class that can be combined with McpAgent
to add UI resource handling without modifying the base agent implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from mcp.types import CallToolResult, ContentBlock, EmbeddedResource

from fast_agent.constants import MCP_UI
from fast_agent.ui.mcp_ui_utils import open_links_in_browser, ui_links_from_channel

if TYPE_CHECKING:
    from rich.text import Text

    from fast_agent.types import PromptMessageExtended


class McpUIMixin:
    """
    Mixin that adds MCP-UI resource handling to any agent.

    This mixin can be combined with any agent class to add UI resource
    extraction and display functionality. It overrides run_tools to
    intercept tool results and extract UI resources.

    Usage:
        class MyAgentWithUI(McpUIMixin, McpAgent):
            def __init__(self, *args, ui_mode: str = "auto", **kwargs):
                super().__init__(*args, **kwargs)
                self._ui_mode = ui_mode
    """

    def __init__(self, *args, ui_mode: str = "auto", **kwargs):
        """Initialize the mixin with UI mode configuration."""
        super().__init__(*args, **kwargs)
        self._ui_mode: str = ui_mode
        self._pending_ui_resources: List[ContentBlock] = []

    def set_ui_mode(self, mode: str) -> None:
        """
        Set the UI mode for handling MCP-UI resources.

        Args:
            mode: One of "disabled", "enabled", or "auto"
        """
        if mode not in ("disabled", "enabled", "auto"):
            mode = "auto"
        self._ui_mode = mode

    async def run_tools(self, request: "PromptMessageExtended") -> "PromptMessageExtended":
        """
        Override run_tools to extract and handle UI resources.

        This method intercepts tool results, extracts any UI resources,
        and adds them to the message channels for later display.
        """
        # If UI is disabled, just pass through to parent
        if self._ui_mode == "disabled":
            return await super().run_tools(request)  # type: ignore

        # Run the tools normally via parent implementation
        result = await super().run_tools(request)  # type: ignore

        # Extract UI resources from tool results
        if result and result.tool_results:
            processed_results, ui_blocks = self._extract_ui_from_tool_results(result.tool_results)

            # For mode 'auto', only act when we actually extracted something
            if self._ui_mode == "enabled" or (self._ui_mode == "auto" and ui_blocks):
                # Update tool_results with UI resources removed
                result.tool_results = processed_results

                # Add UI resources to channels
                channels = result.channels or {}
                current = channels.get(MCP_UI, [])
                channels[MCP_UI] = current + ui_blocks
                result.channels = channels

                # Store for display after assistant message
                self._pending_ui_resources = ui_blocks

        return result

    async def show_assistant_message(
        self,
        message: "PromptMessageExtended",
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Optional["Text"] = None,
    ) -> None:
        """Override to display UI resources after showing assistant message."""
        # Show the assistant message normally via parent
        await super().show_assistant_message(  # type: ignore
            message,
            bottom_items,
            highlight_items,
            max_item_length,
            name=name,
            model=model,
            additional_message=additional_message,
        )

        # Handle any pending UI resources from the previous user message
        if self._ui_mode != "disabled":
            await self._display_ui_resources_from_history()

    async def _display_ui_resources_from_history(self) -> None:
        """
        Check message history for UI resources and display them.

        This looks at the previous user message for any UI resources
        that should be displayed after the assistant's response.
        """
        try:
            history = self.message_history  # type: ignore
            if history and len(history) >= 2:
                prev = history[-2]
                if prev and prev.role == "user":
                    channels = prev.channels or {}
                    ui_resources = channels.get(MCP_UI, []) if isinstance(channels, dict) else []
                    if ui_resources:
                        await self._display_ui_resources(ui_resources)
        except Exception:
            # Silently handle any errors in UI display
            pass

    async def _display_ui_resources(self, resources: Sequence[ContentBlock]) -> None:
        """
        Display UI resources by creating links and optionally opening in browser.

        Args:
            resources: List of UI resource content blocks
        """
        links = ui_links_from_channel(resources)
        if links:
            # Display links in console
            await self.display.show_mcp_ui_links(links)  # type: ignore

            # Auto-open in browser if in auto mode
            if self._ui_mode == "auto":
                open_links_in_browser(links, mcp_ui_mode=self._ui_mode)

    def _extract_ui_from_tool_results(
        self,
        tool_results: Dict[str, CallToolResult],
    ) -> Tuple[Dict[str, CallToolResult], List[ContentBlock]]:
        """
        Extract UI resources from tool results.

        Returns a tuple of (cleaned_tool_results, extracted_ui_blocks).
        """
        if not tool_results:
            return tool_results, []

        extracted_ui: List[ContentBlock] = []
        new_results: Dict[str, CallToolResult] = {}

        for key, result in tool_results.items():
            try:
                ui_blocks, other_blocks = self._split_ui_blocks(list(result.content or []))
                if ui_blocks:
                    extracted_ui.extend(ui_blocks)

                # Recreate CallToolResult without UI blocks
                new_results[key] = CallToolResult(content=other_blocks, isError=result.isError)
            except Exception:
                # Pass through untouched on any error
                new_results[key] = result

        return new_results, extracted_ui

    def _split_ui_blocks(
        self, blocks: List[ContentBlock]
    ) -> Tuple[List[ContentBlock], List[ContentBlock]]:
        """
        Split content blocks into UI and non-UI blocks.

        Returns tuple of (ui_blocks, other_blocks).
        """
        ui_blocks: List[ContentBlock] = []
        other_blocks: List[ContentBlock] = []

        for block in blocks or []:
            if self._is_ui_embedded_resource(block):
                ui_blocks.append(block)
            else:
                other_blocks.append(block)

        return ui_blocks, other_blocks

    def _is_ui_embedded_resource(self, block: ContentBlock) -> bool:
        """Check if a content block is a UI embedded resource."""
        try:
            if isinstance(block, EmbeddedResource):
                res = block.resource
                uri = res.uri if res else None
                if uri is not None:
                    return str(uri).startswith("ui://")
        except Exception:
            pass
        return False
