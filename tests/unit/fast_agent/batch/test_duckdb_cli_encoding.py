"""Check the optional DuckDB CLI's UTF-8 pipe contract."""

import json
import shutil
import subprocess
import sys

import pytest

_QUERY_SCRIPT = """
import json
import locale
import sys

from fast_agent.batch.input import _run_duckdb_cli_json
from fast_agent.utils import huggingface_hub

locale.getencoding = lambda: sys.argv[1]
huggingface_hub.get_huggingface_hub_token = lambda: None
print(json.dumps(_run_duckdb_cli_json(sys.argv[2])))
"""


@pytest.mark.skipif(shutil.which("duckdb") is None, reason="DuckDB CLI is not installed")
@pytest.mark.parametrize("encoding", ["cp1252", "cp936"])
@pytest.mark.parametrize(
    "query, expected",
    [
        ("SELECT chr(128512) AS message", [{"message": "😀"}]),
        ("SELECT '你好😀' AS message", [{"message": "你好😀"}]),
    ],
    ids=["unicode-output", "unicode-query"],
)
def test_duckdb_cli_preserves_unicode_with_non_utf8_locale(encoding, query, expected):
    completed = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", _QUERY_SCRIPT, encoding, query],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == expected
