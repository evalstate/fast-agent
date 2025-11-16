import platform

import pytest

from fast_agent.core.prompt_templates import (
    apply_template_variables,
    enrich_with_environment_context,
)


def test_apply_template_variables_is_noop_without_context():
    template = "Path: {{sessionCwd}}"
    # First pass - no context yet
    assert apply_template_variables(template, {}) == template
    assert apply_template_variables(template, None) == template


@pytest.mark.parametrize(
    "variables,expected",
    [
        ({"sessionCwd": "/workspace/project"}, "Path: /workspace/project"),
        ({"sessionCwd": None}, "Path: {{sessionCwd}}"),
    ],
)
def test_apply_template_variables_applies_when_context_available(variables, expected):
    template = "Path: {{sessionCwd}}"
    assert apply_template_variables(template, variables) == expected


def test_enrich_with_environment_context_populates_env_block():
    context: dict[str, str] = {}
    client_info = {"name": "Zed", "version": "1.2.3"}

    enrich_with_environment_context(context, "/workspace/app", client_info)

    assert context["sessionCwd"] == "/workspace/app"

    env_text = context["env"]
    assert "Environment:" in env_text
    assert "Workspace root: /workspace/app" in env_text
    assert "Client: Zed 1.2.3" in env_text
    assert f"Python: {platform.python_version()}" in env_text
