from click.utils import strip_ansi
from mcp.types import TextContent

from fast_agent.cli.command_support import get_settings_or_exit
from fast_agent.llm.structured_schema import load_json_schema_file
from fast_agent.mcp.prompts.prompt_load import load_prompt


def test_cli_config_path_accepts_file_uri(tmp_path):
    config = tmp_path / "fast-agent.yaml"
    config.write_text("default_model: passthrough\n", encoding="utf-8")

    settings = get_settings_or_exit(config.as_uri(), no_home=True)

    assert settings._config_file == str(config)
    assert settings.default_model == "passthrough"


def test_prompt_file_accepts_file_uri(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hello from uri", encoding="utf-8")

    messages = load_prompt(prompt.as_uri())

    assert len(messages) == 1
    content = messages[0].content[0]
    assert isinstance(content, TextContent)
    assert content.text == "hello from uri"


def test_json_schema_accepts_file_uri(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object","properties":{"ok":{"type":"boolean"}}}', encoding="utf-8")

    loaded = load_json_schema_file(schema.as_uri())

    assert loaded["properties"]["ok"]["type"] == "boolean"


def test_go_help_shows_path_or_url_metavars():
    from typer.testing import CliRunner

    from fast_agent.cli.commands import go

    result = CliRunner().invoke(go.app, ["--help"], terminal_width=160)

    assert result.exit_code == 0
    output = strip_ansi(result.output)
    assert "--config-path" in output
    assert "--prompt-file" in output
    assert "--json-schema" in output
    assert output.count("<path-or-uri>") >= 3


def test_go_help_does_not_show_completion_options():
    from typer.testing import CliRunner

    from fast_agent.cli.commands import go

    result = CliRunner().invoke(go.app, ["--help"], terminal_width=160)

    assert result.exit_code == 0
    output = strip_ansi(result.output)
    assert "--install-completion" not in output
    assert "--show-completion" not in output
