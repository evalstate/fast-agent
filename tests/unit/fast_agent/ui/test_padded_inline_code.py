import io

import pytest
from rich.console import Console

from fast_agent.ui.markdown.renderables import build_markdown_renderable


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        ("`<tag> & value`", "<tag> & value"),
        ("` <tag> & value `", "<tag> & value"),
        ("`` `<tag>` & value ``", "`<tag>` & value"),
        ("```` <tag> `` value ````", "<tag> `` value"),
    ],
)
def test_xml_escaping_preserves_padded_inline_code(span: str, expected: str) -> None:
    renderable = build_markdown_renderable(
        f"Code: {span}; outside <other>",
        code_theme="monokai",
        escape_xml=True,
    )
    output = io.StringIO()
    Console(file=output, force_terminal=False, width=120).print(renderable)

    assert f"Code: {expected}; outside <other>" in output.getvalue()
