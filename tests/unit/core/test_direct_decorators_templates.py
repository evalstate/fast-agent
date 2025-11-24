from fast_agent.core.direct_decorators import _apply_templates


def test_apply_templates_preserves_file_placeholders():
    """File templates should be preserved for late binding resolution."""
    template = "Start {{file:relative/path.txt}} End"
    result = _apply_templates(template)

    # File templates should NOT be resolved early
    assert result == "Start {{file:relative/path.txt}} End"


def test_apply_templates_preserves_file_silent_placeholders():
    """File silent templates should be preserved for late binding resolution."""
    template = "Begin{{file_silent:docs/example.md}}Finish"
    result = _apply_templates(template)

    # File silent templates should NOT be resolved early
    assert result == "Begin{{file_silent:docs/example.md}}Finish"
