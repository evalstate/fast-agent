from rich.syntax import Syntax

from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.tool_display import ToolDisplay


def test_code_tool_call_syntax_uses_code_arg_and_collects_other_args() -> None:
    tool_display = ToolDisplay(ConsoleDisplay())

    syntax, footer_items = tool_display._build_code_tool_call_syntax(
        {
            "code": "def run():\n    return 1\n",
            "limit": 3,
            "raw": True,
        },
        {
            "variant": "code",
            "code_arg": "code",
            "language": "python",
        },
    )

    assert isinstance(syntax, Syntax)
    assert syntax.code == "def run():\n    return 1"
    assert footer_items == ["limit: 3", "raw: true"]


def test_write_text_file_call_uses_path_derived_syntax() -> None:
    tool_display = ToolDisplay(ConsoleDisplay())

    prepared = tool_display._prepare_tool_call_display(
        tool_name="write_text_file",
        tool_args={
            "content": "# Boeing 747-400\n\nA highly detailed model\n",
            "path": "/tmp/plane/README.md",
        },
        bottom_items=None,
        highlight_indexes=[],
        max_item_length=None,
        metadata={},
        tool_call_id="call-write-1",
        type_label="agent tool",
    )

    assert isinstance(prepared.content, Syntax)
    assert prepared.content.code == "# Boeing 747-400\n\nA highly detailed model"
    lexer = prepared.content.lexer
    assert lexer is not None
    assert lexer.name == "Markdown"
    assert prepared.bottom_items == ["path: /tmp/plane/README.md"]
    assert prepared.truncate_content is False
