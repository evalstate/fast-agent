from __future__ import annotations

import pytest

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.tools.function_tool_loader import load_function_from_spec
from fast_agent.tools.hook_loader import load_hook_function
from fast_agent.tools.python_file_loader import parse_callable_file_spec


def test_parse_callable_file_spec_splits_on_final_colon() -> None:
    parsed = parse_callable_file_spec(
        "package:module.py:run",
        invalid_message="Invalid spec: {spec}",
    )

    assert parsed.raw == "package:module.py:run"
    assert parsed.module_path_text == "package:module.py"
    assert parsed.callable_name == "run"


def test_parse_callable_file_spec_rejects_missing_callable_separator() -> None:
    with pytest.raises(AgentConfigError, match="Invalid spec: module.py"):
        parse_callable_file_spec("module.py", invalid_message="Invalid spec: {spec}")


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", [load_function_from_spec, load_hook_function])
async def test_file_callables_can_use_dataclasses_with_postponed_annotations(
    tmp_path, loader
) -> None:
    module_path = tmp_path / "callables.py"
    module_path.write_text(
        """from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class Result:
    prefix: ClassVar[str] = "processed"
    value: str

async def run(value: str) -> str:
    result = Result(value)
    return f"{result.prefix}: {result.value}"
""",
        encoding="utf-8",
    )

    function = loader("callables.py:run", base_path=tmp_path)

    assert await function("tool output") == "processed: tool output"
