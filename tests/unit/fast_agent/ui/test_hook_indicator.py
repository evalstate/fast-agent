"""Tests for hook indicator glyph in message headers."""


from fast_agent.ui.console_display import HOOK_INDICATOR_GLYPH, ConsoleDisplay


class TestBuildHeaderLeft:
    """Tests for ConsoleDisplay.build_header_left static method."""

    def test_basic_header_no_name_no_hook(self) -> None:
        """Basic header with just block and arrow."""
        result = ConsoleDisplay.build_header_left(
            block_color="green",
            arrow="◀",
            arrow_style="dim green",
        )
        assert result == "[green]▎[/green][dim green]◀[/dim green]"

    def test_header_with_name_no_hook(self) -> None:
        """Header with name but no hook indicator."""
        result = ConsoleDisplay.build_header_left(
            block_color="green",
            arrow="◀",
            arrow_style="dim green",
            name="test-agent",
        )
        assert result == "[green]▎[/green][dim green]◀[/dim green] [green]test-agent[/green]"

    def test_header_with_hook_indicator_no_name(self) -> None:
        """Header with hook indicator but no name."""
        result = ConsoleDisplay.build_header_left(
            block_color="green",
            arrow="◀",
            arrow_style="dim green",
            show_hook_indicator=True,
        )
        expected = f"[green]▎[/green][dim green]◀[/dim green] [green]{HOOK_INDICATOR_GLYPH}[/green]"
        assert result == expected

    def test_header_with_hook_indicator_and_name(self) -> None:
        """Header with both hook indicator and name - hook should appear before name."""
        result = ConsoleDisplay.build_header_left(
            block_color="green",
            arrow="◀",
            arrow_style="dim green",
            name="test-agent",
            show_hook_indicator=True,
        )
        expected = f"[green]▎[/green][dim green]◀[/dim green] [green]{HOOK_INDICATOR_GLYPH}[/green] [green]test-agent[/green]"
        assert result == expected

    def test_hook_indicator_uses_block_color(self) -> None:
        """Hook indicator should use the same color as the block."""
        result = ConsoleDisplay.build_header_left(
            block_color="blue",
            arrow="▶",
            arrow_style="dim blue",
            name="user",
            show_hook_indicator=True,
        )
        # The hook indicator should be in blue (matching block_color)
        expected = f"[blue]▎[/blue][dim blue]▶[/dim blue] [blue]{HOOK_INDICATOR_GLYPH}[/blue] [blue]user[/blue]"
        assert result == expected

    def test_error_state_name_is_red(self) -> None:
        """When is_error=True, name should be red but hook indicator uses block color."""
        result = ConsoleDisplay.build_header_left(
            block_color="magenta",
            arrow="▶",
            arrow_style="dim magenta",
            name="error-agent",
            is_error=True,
            show_hook_indicator=True,
        )
        # Hook indicator stays magenta, name becomes red
        expected = f"[magenta]▎[/magenta][dim magenta]▶[/dim magenta] [magenta]{HOOK_INDICATOR_GLYPH}[/magenta] [red]error-agent[/red]"
        assert result == expected

    def test_tool_call_header_with_hook(self) -> None:
        """Tool call header (magenta) with hook indicator."""
        result = ConsoleDisplay.build_header_left(
            block_color="magenta",
            arrow="◀",
            arrow_style="dim magenta",
            name="dev",
            show_hook_indicator=True,
        )
        expected = f"[magenta]▎[/magenta][dim magenta]◀[/dim magenta] [magenta]{HOOK_INDICATOR_GLYPH}[/magenta] [magenta]dev[/magenta]"
        assert result == expected

    def test_tool_result_header_with_hook(self) -> None:
        """Tool result header (magenta, arrow pointing right) with hook indicator."""
        result = ConsoleDisplay.build_header_left(
            block_color="magenta",
            arrow="▶",
            arrow_style="dim magenta",
            name="dev",
            show_hook_indicator=True,
        )
        expected = f"[magenta]▎[/magenta][dim magenta]▶[/dim magenta] [magenta]{HOOK_INDICATOR_GLYPH}[/magenta] [magenta]dev[/magenta]"
        assert result == expected

    def test_user_message_header_with_hook(self) -> None:
        """User message header (blue) with hook indicator."""
        result = ConsoleDisplay.build_header_left(
            block_color="blue",
            arrow="▶",
            arrow_style="dim blue",
            name="user",
            show_hook_indicator=True,
        )
        expected = f"[blue]▎[/blue][dim blue]▶[/dim blue] [blue]{HOOK_INDICATOR_GLYPH}[/blue] [blue]user[/blue]"
        assert result == expected

    def test_assistant_message_header_with_hook(self) -> None:
        """Assistant message header (green) with hook indicator."""
        result = ConsoleDisplay.build_header_left(
            block_color="green",
            arrow="◀",
            arrow_style="dim green",
            name="assistant",
            show_hook_indicator=True,
        )
        expected = f"[green]▎[/green][dim green]◀[/dim green] [green]{HOOK_INDICATOR_GLYPH}[/green] [green]assistant[/green]"
        assert result == expected


class TestHookIndicatorGlyph:
    """Tests for the hook indicator glyph constant."""

    def test_glyph_is_diamond(self) -> None:
        """Verify the glyph is a diamond character."""
        assert HOOK_INDICATOR_GLYPH == "◆"

    def test_glyph_is_single_character(self) -> None:
        """Verify the glyph is a single character."""
        assert len(HOOK_INDICATOR_GLYPH) == 1
