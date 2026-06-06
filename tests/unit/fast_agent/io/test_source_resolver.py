import nturl2path
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

import fast_agent.io.source_resolver as source_resolver
from fast_agent.io.path_uri import file_uri_to_path
from fast_agent.io.source_resolver import (
    _read_hf_text_source,
    materialize_text_source,
    materialized_text_source,
    read_text_source,
)


def test_read_text_source_supports_file_uri(tmp_path):
    source = tmp_path / "prompt.txt"
    source.write_text("hello", encoding="utf-8")

    assert read_text_source(source.as_uri(), label="prompt file") == "hello"


def test_materialize_text_source_returns_path_for_file_uri(tmp_path):
    source = tmp_path / "fast-agent.yaml"
    source.write_text("default_model: test\n", encoding="utf-8")

    assert materialize_text_source(source.as_uri(), label="config file") == source


def test_read_text_source_expands_user_home(tmp_path, monkeypatch):
    source = tmp_path / "schema.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert read_text_source("~/schema.json", label="JSON schema file") == "{}"


def test_read_text_source_preserves_local_file_not_found(tmp_path):
    missing = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError):
        read_text_source(missing, label="instruction")


class _FakeHfFile:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self._data


class _FakeHfFileSystem:
    def __init__(self, data: bytes = b"hello from hf") -> None:
        self.data = data
        self.opened: list[tuple[str, str]] = []

    def open(self, path: str, mode: str = "rb"):
        self.opened.append((path, mode))
        return _FakeHfFile(self.data)


def test_read_hf_text_source_uses_hf_filesystem():
    filesystem = _FakeHfFileSystem(b"hello from hf")

    result = _read_hf_text_source(
        "hf://buckets/evalstate/home/demo.md",
        label="prompt file",
        filesystem=filesystem,
    )

    assert result == "hello from hf"
    assert filesystem.opened == [("hf://buckets/evalstate/home/demo.md", "rb")]


def test_read_text_source_delegates_hf_scheme(monkeypatch):
    calls = []

    def fake_read_hf_text_source(source: str, *, label: str) -> str:
        calls.append((source, label))
        return "delegated"

    monkeypatch.setattr(source_resolver, "_read_hf_text_source", fake_read_hf_text_source)

    assert read_text_source("hf://buckets/evalstate/home/demo.md", label="prompt file") == "delegated"
    assert calls == [("hf://buckets/evalstate/home/demo.md", "prompt file")]


def test_read_text_source_wraps_http_request_errors(monkeypatch):
    def fake_get(source: str, *, timeout: int):
        assert source == "https://example.com/missing.md"
        assert timeout == 30
        raise requests.ConnectionError("connection failed")

    monkeypatch.setattr(source_resolver.requests, "get", fake_get)

    with pytest.raises(
        ValueError,
        match="Could not read prompt file https://example.com/missing.md: connection failed",
    ):
        read_text_source("https://example.com/missing.md", label="prompt file")


def test_read_text_source_wraps_http_status_errors(monkeypatch):
    class FakeResponse:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("404 Client Error")

    def fake_get(source: str, *, timeout: int) -> FakeResponse:
        assert source == "https://example.com/missing.md"
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr(source_resolver.requests, "get", fake_get)

    with pytest.raises(
        ValueError,
        match="Could not read prompt file https://example.com/missing.md: 404 Client Error",
    ):
        read_text_source("https://example.com/missing.md", label="prompt file")


def test_materialize_text_source_writes_temp_file_for_hf_uri(monkeypatch):
    def fake_read_text_source(source: str, *, label: str) -> str:
        assert source == "hf://buckets/evalstate/home/demo.md"
        assert label == "prompt file"
        return "hf prompt"

    monkeypatch.setattr(source_resolver, "read_text_source", fake_read_text_source)

    materialized = materialize_text_source(
        "hf://buckets/evalstate/home/demo.md",
        label="prompt file",
    )

    assert materialized.suffix == ".md"
    assert materialized.read_text(encoding="utf-8") == "hf prompt"


def test_materialized_text_source_removes_remote_temp_file(monkeypatch):
    def fake_read_text_source(source: str, *, label: str) -> str:
        assert source == "hf://buckets/evalstate/home/demo.md"
        assert label == "prompt file"
        return "hf prompt"

    monkeypatch.setattr(source_resolver, "read_text_source", fake_read_text_source)

    with materialized_text_source("hf://buckets/evalstate/home/demo.md", label="prompt file") as path:
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "hf prompt"

    assert not path.exists()


def test_file_uri_to_path_supports_windows_drive_uri():
    parsed = urlparse("file:///C:/Users/alice/fast-agent.yaml")

    path = file_uri_to_path(parsed, pathname_decoder=nturl2path.url2pathname)

    assert str(path) == r"C:\Users\alice\fast-agent.yaml"


def test_file_uri_to_path_preserves_unc_host():
    parsed = urlparse("file://server/share/fast-agent.yaml")

    path = file_uri_to_path(parsed, pathname_decoder=nturl2path.url2pathname)

    assert str(path) == r"\\server\share\fast-agent.yaml"


def test_file_uri_to_path_preserves_nonlocal_authority_by_default():
    parsed = urlparse("file://server/share/fast-agent%20config.yaml")

    path = file_uri_to_path(parsed)

    assert path == Path("//server/share/fast-agent config.yaml")


def test_file_uri_to_path_treats_localhost_as_local():
    parsed = urlparse("file://localhost/tmp/fast-agent.yaml")

    path = file_uri_to_path(parsed)

    assert path == Path("/tmp/fast-agent.yaml")


def test_file_uri_to_path_normalizes_scheme_and_localhost():
    parsed = urlparse("FILE://LOCALHOST/tmp/fast-agent.yaml")

    path = file_uri_to_path(parsed)

    assert path == Path("/tmp/fast-agent.yaml")


def test_file_uri_to_path_rejects_non_file_scheme():
    parsed = urlparse("https://example.test/fast-agent.yaml")

    with pytest.raises(ValueError, match="Expected file URI, got https"):
        file_uri_to_path(parsed)


def test_file_uri_to_path_rejects_missing_scheme():
    parsed = urlparse("/tmp/fast-agent.yaml")

    with pytest.raises(ValueError, match="Expected file URI, got <missing>"):
        file_uri_to_path(parsed)
