import os
from pathlib import Path

import pytest

from fast_agent.io.path_uri import local_file_url_path

AWKWARD_NAMES = ["sample.pdf", "a b.pdf", "report%20.pdf", "résumé #100%.pdf"]
if os.name != "nt":  # "?" is not a legal Windows filename character
    AWKWARD_NAMES.append("q?x.pdf")


@pytest.mark.parametrize("name", AWKWARD_NAMES)
def test_local_file_url_path_resolves_encoded_uri(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"data")

    assert local_file_url_path(path.as_uri()) == path


@pytest.mark.parametrize("name", AWKWARD_NAMES)
def test_local_file_url_path_resolves_unencoded_url(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"data")

    assert local_file_url_path(f"file://{path}") == path


@pytest.mark.parametrize(
    "url",
    ["https://example.com/a.pdf", "data:application/pdf;base64,AAAA", "file:///no/such/file.pdf"],
)
def test_local_file_url_path_rejects_non_local_or_missing(url: str) -> None:
    assert local_file_url_path(url) is None


def test_local_file_url_path_rejects_directories(tmp_path: Path) -> None:
    assert local_file_url_path(tmp_path.as_uri()) is None
