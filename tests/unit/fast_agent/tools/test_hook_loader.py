"""Unit tests for the hook loader."""

from pathlib import Path

import pytest

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.tools.hook_loader import (
    VALID_HOOK_TYPES,
    load_hook_function,
    load_tool_runner_hooks,
)


@pytest.mark.unit
class TestLoadHookFunction:
    """Tests for load_hook_function."""

    def test_valid_spec_loads_function(self, tmp_path: Path):
        """Test loading a function from a valid spec."""
        # Create a temporary module with a hook function
        hook_file = tmp_path / "hooks.py"
        hook_file.write_text(
            """
async def my_hook(ctx):
    pass
"""
        )

        func = load_hook_function(f"{hook_file}:my_hook")

        assert callable(func)
        assert func.__name__ == "my_hook"

    def test_invalid_spec_missing_colon(self):
        """Test that spec without colon raises error."""
        with pytest.raises(AgentConfigError) as exc_info:
            load_hook_function("hooks.py")

        assert "Invalid hook spec" in str(exc_info.value)
        assert "Expected format" in str(exc_info.value)

    def test_missing_file_raises_error(self, tmp_path: Path):
        """Test that non-existent file raises error."""
        nonexistent = tmp_path / "nonexistent.py"

        with pytest.raises(AgentConfigError) as exc_info:
            load_hook_function(f"{nonexistent}:my_hook")

        assert "not found" in str(exc_info.value)

    def test_missing_function_raises_error(self, tmp_path: Path):
        """Test that missing function in module raises error."""
        hook_file = tmp_path / "hooks.py"
        hook_file.write_text(
            """
async def other_hook(ctx):
    pass
"""
        )

        with pytest.raises(AgentConfigError) as exc_info:
            load_hook_function(f"{hook_file}:my_hook")

        assert "not found" in str(exc_info.value)

    def test_non_callable_raises_error(self, tmp_path: Path):
        """Test that non-callable attribute raises error."""
        hook_file = tmp_path / "hooks.py"
        hook_file.write_text(
            """
my_hook = "not a function"
"""
        )

        with pytest.raises(AgentConfigError) as exc_info:
            load_hook_function(f"{hook_file}:my_hook")

        assert "not callable" in str(exc_info.value)

    def test_relative_path_with_base_path(self, tmp_path: Path):
        """Test loading with relative path and base_path."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        hook_file = subdir / "hooks.py"
        hook_file.write_text(
            """
async def relative_hook(ctx):
    pass
"""
        )

        # Use relative path from subdir
        func = load_hook_function("hooks.py:relative_hook", base_path=subdir)

        assert callable(func)
        assert func.__name__ == "relative_hook"


@pytest.mark.unit
class TestLoadToolRunnerHooks:
    """Tests for load_tool_runner_hooks."""

    def test_creates_hooks_from_config(self, tmp_path: Path):
        """Test that dict config creates ToolRunnerHooks."""
        hook_file = tmp_path / "hooks.py"
        hook_file.write_text(
            """
async def before_hook(ctx):
    pass

async def after_hook(ctx):
    pass
"""
        )

        # Create a mock agent
        class MockAgent:
            @property
            def message_history(self):
                return []

            def load_message_history(self, msgs):
                pass

        config = {
            "before_llm_call": f"{hook_file}:before_hook",
            "after_llm_call": f"{hook_file}:after_hook",
        }

        hooks = load_tool_runner_hooks(config, MockAgent(), base_path=tmp_path)

        assert hooks is not None
        assert hooks.before_llm_call is not None
        assert hooks.after_llm_call is not None
        assert hooks.before_tool_call is None
        assert hooks.after_tool_call is None

    def test_invalid_hook_type_raises_error(self, tmp_path: Path):
        """Test that invalid hook type raises error."""
        hook_file = tmp_path / "hooks.py"
        hook_file.write_text(
            """
async def my_hook(ctx):
    pass
"""
        )

        class MockAgent:
            @property
            def message_history(self):
                return []

            def load_message_history(self, msgs):
                pass

        config = {
            "invalid_hook_type": f"{hook_file}:my_hook",
        }

        with pytest.raises(AgentConfigError) as exc_info:
            load_tool_runner_hooks(config, MockAgent())

        assert "Invalid hook types" in str(exc_info.value)

    def test_empty_config_returns_none(self):
        """Test that empty config returns None."""

        class MockAgent:
            pass

        hooks = load_tool_runner_hooks(None, MockAgent())
        assert hooks is None

        hooks = load_tool_runner_hooks({}, MockAgent())
        assert hooks is None

    def test_all_valid_hook_types(self, tmp_path: Path):
        """Test that all valid hook types can be loaded."""
        hook_file = tmp_path / "hooks.py"
        hook_file.write_text(
            """
async def hook1(ctx): pass
async def hook2(ctx): pass
async def hook3(ctx): pass
async def hook4(ctx): pass
async def hook5(ctx): pass
"""
        )

        class MockAgent:
            @property
            def message_history(self):
                return []

            def load_message_history(self, msgs):
                pass

        config = {
            "before_llm_call": f"{hook_file}:hook1",
            "after_llm_call": f"{hook_file}:hook2",
            "before_tool_call": f"{hook_file}:hook3",
            "after_tool_call": f"{hook_file}:hook4",
            "after_turn_complete": f"{hook_file}:hook5",
        }

        hooks = load_tool_runner_hooks(config, MockAgent())

        assert hooks is not None
        assert hooks.before_llm_call is not None
        assert hooks.after_llm_call is not None
        assert hooks.before_tool_call is not None
        assert hooks.after_tool_call is not None
        assert hooks.after_turn_complete is not None


@pytest.mark.unit
class TestValidHookTypes:
    """Tests for the VALID_HOOK_TYPES constant."""

    def test_valid_hook_types_contains_expected(self):
        """Test that VALID_HOOK_TYPES contains all expected hook types."""
        expected = {
            "before_llm_call",
            "after_llm_call",
            "before_tool_call",
            "after_tool_call",
            "after_turn_complete",
        }
        assert VALID_HOOK_TYPES == expected
