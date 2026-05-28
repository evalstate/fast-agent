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
