from __future__ import annotations

import pytest

from fast_agent.core.exceptions import AgentConfigError
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
