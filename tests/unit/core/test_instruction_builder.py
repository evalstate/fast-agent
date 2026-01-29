"""Unit tests for InstructionBuilder."""

import pytest

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.instruction import InstructionBuilder


class TestInstructionBuilder:
    """Tests for InstructionBuilder class."""

    @pytest.mark.asyncio
    async def test_build_with_no_placeholders(self):
        """Build should return template unchanged when no placeholders."""
        builder = InstructionBuilder("You are a helpful assistant.")
        result = await builder.build()
        assert result == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_build_with_static_placeholder(self):
        """Build should substitute static placeholders."""
        builder = InstructionBuilder("Today is {{currentDate}}.")
        builder.set("currentDate", "17 Dec 2025")
        result = await builder.build()
        assert result == "Today is 17 Dec 2025."

    @pytest.mark.asyncio
    async def test_build_with_escaped_placeholders(self):
        """Escaped placeholders should remain literal."""
        builder = InstructionBuilder(r"Literal: \{{currentDate}} and \{{file:missing.md}}")
        builder.set("currentDate", "17 Dec 2025")
        result = await builder.build()
        assert result == "Literal: {{currentDate}} and {{file:missing.md}}"

    @pytest.mark.asyncio
    async def test_build_with_multiple_static_placeholders(self):
        """Build should substitute multiple static placeholders."""
        builder = InstructionBuilder("Hello {{name}}, you are in {{location}}.")
        builder.set("name", "Alice")
        builder.set("location", "Wonderland")
        result = await builder.build()
        assert result == "Hello Alice, you are in Wonderland."

    @pytest.mark.asyncio
    async def test_build_with_resolver(self):
        """Build should call resolver for dynamic placeholders."""
        async def get_weather():
            return "sunny"

        builder = InstructionBuilder("The weather is {{weather}}.")
        builder.set_resolver("weather", get_weather)
        result = await builder.build()
        assert result == "The weather is sunny."

    @pytest.mark.asyncio
    async def test_resolver_called_each_build(self):
        """Resolver should be called on each build() invocation."""
        call_count = 0

        async def counter():
            nonlocal call_count
            call_count += 1
            return str(call_count)

        builder = InstructionBuilder("Count: {{count}}")
        builder.set_resolver("count", counter)

        result1 = await builder.build()
        result2 = await builder.build()

        assert result1 == "Count: 1"
        assert result2 == "Count: 2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_build_with_mixed_sources(self):
        """Build should handle both static and dynamic sources."""
        async def get_time():
            return "3:00 PM"

        builder = InstructionBuilder("Date: {{date}}, Time: {{time}}")
        builder.set("date", "Dec 17")
        builder.set_resolver("time", get_time)

        result = await builder.build()
        assert result == "Date: Dec 17, Time: 3:00 PM"

    @pytest.mark.asyncio
    async def test_unresolved_placeholder_left_as_is(self):
        """Unresolved placeholders should remain in output."""
        builder = InstructionBuilder("Hello {{name}}, {{unset}} world.")
        builder.set("name", "Alice")
        result = await builder.build()
        assert result == "Hello Alice, {{unset}} world."

    @pytest.mark.asyncio
    async def test_resolver_error_returns_empty(self):
        """Failed resolver should return empty string for that placeholder."""
        async def failing_resolver():
            raise ValueError("Something went wrong")

        builder = InstructionBuilder("Data: {{data}}")
        builder.set_resolver("data", failing_resolver)
        result = await builder.build()
        assert result == "Data: "

    @pytest.mark.asyncio
    async def test_set_many(self):
        """set_many should set multiple static values."""
        builder = InstructionBuilder("{{a}} {{b}} {{c}}")
        builder.set_many({"a": "1", "b": "2", "c": "3"})
        result = await builder.build()
        assert result == "1 2 3"

    def test_fluent_api(self):
        """set() and set_resolver() should return self for chaining."""
        async def resolver():
            return "x"

        builder = (
            InstructionBuilder("{{a}} {{b}}")
            .set("a", "1")
            .set_resolver("b", resolver)
        )
        assert isinstance(builder, InstructionBuilder)

    def test_get_placeholders(self):
        """get_placeholders should extract placeholder names."""
        builder = InstructionBuilder(
            "Hello {{name}}, {{greeting}}. File: {{file:test.md}}"
        )
        placeholders = builder.get_placeholders()
        # Should not include file: patterns
        assert placeholders == {"name", "greeting"}

    def test_get_placeholders_ignores_escaped(self):
        """Escaped placeholders should be ignored in placeholder extraction."""
        builder = InstructionBuilder(r"\{{ignored}} {{real}}")
        placeholders = builder.get_placeholders()
        assert placeholders == {"real"}

    def test_get_unresolved_placeholders(self):
        """get_unresolved_placeholders should return placeholders without sources."""
        builder = InstructionBuilder("{{a}} {{b}} {{c}}")
        builder.set("a", "1")
        unresolved = builder.get_unresolved_placeholders()
        assert unresolved == {"b", "c"}

    def test_copy(self):
        """copy() should create independent copy."""
        builder1 = InstructionBuilder("{{a}}")
        builder1.set("a", "original")

        builder2 = builder1.copy()
        builder2.set("a", "modified")

        # Original should be unchanged
        assert builder1._static["a"] == "original"
        assert builder2._static["a"] == "modified"

    def test_template_property(self):
        """template property should return original template."""
        template = "You are helpful. {{serverInstructions}}"
        builder = InstructionBuilder(template)
        assert builder.template == template

    def test_repr(self):
        """__repr__ should provide useful info."""
        builder = InstructionBuilder("Hello {{name}}")
        builder.set("name", "World")
        repr_str = repr(builder)
        assert "InstructionBuilder" in repr_str
        assert "static=1" in repr_str


class TestInstructionBuilderFilePatterns:
    """Tests for file pattern resolution."""

    @pytest.mark.asyncio
    async def test_file_pattern_with_workspace_root(self, tmp_path):
        """{{file:path}} should read file relative to workspaceRoot."""
        # Create a test file
        test_file = tmp_path / "instructions.md"
        test_file.write_text("# Instructions\nBe helpful.")

        builder = InstructionBuilder("Base. {{file:instructions.md}}")
        builder.set("workspaceRoot", str(tmp_path))

        result = await builder.build()
        assert result == "Base. # Instructions\nBe helpful."

    @pytest.mark.asyncio
    async def test_file_silent_pattern_missing_file(self, tmp_path):
        """{{file_silent:path}} should return empty for missing file."""
        builder = InstructionBuilder("Base.{{file_silent:missing.md}}")
        builder.set("workspaceRoot", str(tmp_path))

        result = await builder.build()
        assert result == "Base."

    @pytest.mark.asyncio
    async def test_file_pattern_missing_file_raises(self, tmp_path):
        """{{file:path}} should raise AgentConfigError when missing."""
        builder = InstructionBuilder("Base. {{file:missing.md}}")
        builder.set("workspaceRoot", str(tmp_path))

        with pytest.raises(AgentConfigError, match="Instruction file not found"):
            await builder.build()

    @pytest.mark.asyncio
    async def test_file_pattern_rejects_absolute_paths(self):
        """{{file:path}} should reject absolute paths."""
        builder = InstructionBuilder("{{file:/etc/passwd}}")
        with pytest.raises(ValueError, match="must be relative"):
            await builder.build()


class TestInstructionBuilderUrlPatterns:
    """Tests for URL pattern resolution."""

    @pytest.mark.asyncio
    async def test_url_pattern_resolution(self, monkeypatch):
        """{{url:...}} should fetch content from URL."""
        import requests

        class MockResponse:
            text = "Remote content"

            def raise_for_status(self):
                pass

        def mock_get(*args, **kwargs):
            return MockResponse()

        monkeypatch.setattr(requests, "get", mock_get)

        builder = InstructionBuilder("Data: {{url:https://example.com/data.txt}}")
        result = await builder.build()

        assert result == "Data: Remote content"

    @pytest.mark.asyncio
    async def test_url_pattern_error_returns_empty(self, monkeypatch):
        """Failed URL fetch should return empty string."""
        import requests

        def mock_get(*args, **kwargs):
            raise requests.RequestException("Network error")

        monkeypatch.setattr(requests, "get", mock_get)

        builder = InstructionBuilder("Data: {{url:https://example.com/fail.txt}}")
        result = await builder.build()

        assert result == "Data: "
