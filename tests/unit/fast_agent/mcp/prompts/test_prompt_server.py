import asyncio

import pytest
from mcp.types import TextContent

from fast_agent.mcp.prompts import prompt_server


def test_register_prompt_preserves_non_identifier_template_variables(
    tmp_path,
    monkeypatch,
) -> None:
    template_path = tmp_path / "dynamic.txt"
    template_path.write_text(
        "Hello {{file:snippet.txt}} from {{user-name}} and {{user_name}}.",
        encoding="utf-8",
    )

    added_prompts = []
    monkeypatch.setattr(prompt_server, "prompt_registry", {})
    monkeypatch.setattr(prompt_server, "exposed_resources", {})
    monkeypatch.setattr(prompt_server.mcp, "add_prompt", added_prompts.append)
    monkeypatch.setattr(prompt_server.mcp, "add_resource", lambda _resource: None)

    prompt_server.register_prompt(template_path)

    assert len(added_prompts) == 1
    prompt = added_prompts[0]
    assert [arg.name for arg in prompt.arguments or []] == [
        "file:snippet.txt",
        "user-name",
        "user_name",
    ]

    result = asyncio.run(
        prompt.render(
            {
                "file:snippet.txt": "README.md",
                "user-name": "alice",
                "user_name": "bob",
            }
        )
    )

    assert isinstance(result.messages[0].content, TextContent)
    assert result.messages[0].content.text == "Hello README.md from alice and bob."


def test_template_function_prompt_rejects_non_dict_arguments(tmp_path, monkeypatch) -> None:
    template_path = tmp_path / "dynamic.txt"
    template_path.write_text("Hello {{name}}.", encoding="utf-8")

    added_prompts = []
    monkeypatch.setattr(prompt_server, "prompt_registry", {})
    monkeypatch.setattr(prompt_server, "exposed_resources", {})
    monkeypatch.setattr(prompt_server.mcp, "add_prompt", added_prompts.append)
    monkeypatch.setattr(prompt_server.mcp, "add_resource", lambda _resource: None)

    prompt_server.register_prompt(template_path)

    with pytest.raises(ValueError, match="Prompt arguments must be a dict"):
        asyncio.run(added_prompts[0].render("name=alice"))


def test_register_prompt_normalizes_json_suffix(tmp_path, monkeypatch) -> None:
    prompt_path = tmp_path / "dynamic.JSON"
    prompt_path.write_text(
        """
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Hello {{name}}."}
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    added_prompts = []
    monkeypatch.setattr(prompt_server, "prompt_registry", {})
    monkeypatch.setattr(prompt_server, "exposed_resources", {})
    monkeypatch.setattr(prompt_server.mcp, "add_prompt", added_prompts.append)
    monkeypatch.setattr(prompt_server.mcp, "add_resource", lambda _resource: None)

    prompt_server.register_prompt(prompt_path)

    assert len(added_prompts) == 1
    result = asyncio.run(added_prompts[0].render({"name": "Ada"}))

    assert isinstance(result.messages[0].content, TextContent)
    assert result.messages[0].content.text == "Hello Ada."
