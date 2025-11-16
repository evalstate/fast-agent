
import pytest

from fast_agent.core.prompt_templates import (
    apply_template_variables,
    enrich_with_environment_context,
)


def test_apply_template_variables_is_noop_without_context():
    template = "Path: {{workspaceRoot}}"
    # First pass - no context yet
    assert apply_template_variables(template, {}) == template
    assert apply_template_variables(template, None) == template


@pytest.mark.parametrize(
    "variables,expected",
    [
        ({"workspaceRoot": "/workspace/project"}, "Path: /workspace/project"),
        ({"workspaceRoot": None}, "Path: {{workspaceRoot}}"),
    ],
)
def test_apply_template_variables_applies_when_context_available(variables, expected):
    template = "Path: {{workspaceRoot}}"
    assert apply_template_variables(template, variables) == expected


def test_enrich_with_environment_context_populates_env_block():
    context: dict[str, str] = {}
    client_info = {"name": "Zed", "version": "1.2.3"}

    enrich_with_environment_context(context, "/workspace/app", client_info)

    assert context["workspaceRoot"] == "/workspace/app"

    env_text = context["env"]
    assert "Environment:" in env_text
    assert "Workspace root: /workspace/app" in env_text
    assert "Client: Zed 1.2.3" in env_text
    assert "Host platform:" in env_text


def test_file_template_substitutes_contents_relative_to_workspace(tmp_path):
    """File templates should resolve relative to workspaceRoot."""
    # Create a file in the workspace
    file_path = tmp_path / "snippet.txt"
    file_path.write_text("Hello template", encoding="utf-8")

    template = "Start {{file:snippet.txt}} End"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "Start Hello template End"


def test_file_template_supports_nested_paths(tmp_path):
    """File templates should support nested relative paths."""
    # Create nested directory structure
    nested_dir = tmp_path / "docs" / "examples"
    nested_dir.mkdir(parents=True)
    file_path = nested_dir / "note.txt"
    file_path.write_text("Nested content", encoding="utf-8")

    template = "Content: {{file:docs/examples/note.txt}}"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "Content: Nested content"


def test_file_template_rejects_absolute_paths(tmp_path):
    """File templates must reject absolute paths."""
    absolute_path = tmp_path / "file.txt"
    absolute_path.write_text("content", encoding="utf-8")

    template = f"Start {{{{file:{absolute_path}}}}} End"
    variables = {"workspaceRoot": str(tmp_path)}

    with pytest.raises(ValueError, match="File template paths must be relative"):
        apply_template_variables(template, variables)


def test_file_silent_returns_empty_when_missing(tmp_path):
    """File silent templates should return empty string for missing files."""
    template = "Begin{{file_silent:missing.txt}}Finish"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "BeginFinish"


def test_file_silent_reads_when_present(tmp_path):
    """File silent templates should read file when present."""
    file_path = tmp_path / "note.txt"
    file_path.write_text("data", encoding="utf-8")

    template = "Value: {{file_silent:note.txt}}"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "Value: data"


def test_file_silent_rejects_absolute_paths(tmp_path):
    """File silent templates must reject absolute paths."""
    absolute_path = tmp_path / "file.txt"

    template = f"Start {{{{file_silent:{absolute_path}}}}} End"
    variables = {"workspaceRoot": str(tmp_path)}

    with pytest.raises(ValueError, match="File template paths must be relative"):
        apply_template_variables(template, variables)
