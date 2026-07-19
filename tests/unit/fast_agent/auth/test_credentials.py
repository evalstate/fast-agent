import json
import stat
from pathlib import Path

from fast_agent.auth.credentials import (
    OAuthCredential,
    load_oauth_credential,
    save_oauth_credential,
)


def test_auth_file_can_be_read_without_creating_a_sibling_lock(
    monkeypatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / "mounted-secret" / "auth.json"
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(auth_path))
    save_oauth_credential("codex", OAuthCredential(access_token="mounted-secret-token"))
    auth_path.parent.chmod(0o500)

    try:
        stored = load_oauth_credential("codex")
    finally:
        auth_path.parent.chmod(0o700)

    assert stored is not None
    assert stored.credential.access_token == "mounted-secret-token"
    assert not auth_path.with_suffix(".json.lock").exists()


def test_explicit_auth_file_preserves_existing_file_and_directory_modes(
    monkeypatch, tmp_path: Path
) -> None:
    shared_directory = tmp_path / "shared"
    shared_directory.mkdir()
    shared_directory.chmod(0o755)
    auth_path = shared_directory / "auth.json"
    auth_path.write_text(json.dumps({"version": 1, "providers": {}}))
    auth_path.chmod(0o640)
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(auth_path))

    save_oauth_credential("xai", OAuthCredential(access_token="xai-token"))

    assert stat.S_IMODE(shared_directory.stat().st_mode) == 0o755
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o640


def test_new_explicit_auth_file_preserves_existing_directory_mode(
    monkeypatch, tmp_path: Path
) -> None:
    shared_directory = tmp_path / "shared"
    shared_directory.mkdir()
    shared_directory.chmod(0o775)
    auth_path = shared_directory / "auth.json"
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(auth_path))

    save_oauth_credential("xai", OAuthCredential(access_token="xai-token"))

    assert stat.S_IMODE(shared_directory.stat().st_mode) == 0o775
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600


def test_new_default_auth_directory_is_private(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FAST_AGENT_AUTH_FILE", raising=False)
    monkeypatch.setattr("fast_agent.auth.credentials.Path.home", lambda: tmp_path)

    save_oauth_credential(
        "xai",
        OAuthCredential(access_token="xai-token"),
        source="file",
    )

    auth_path = tmp_path / ".fast-agent" / "auth.json"
    assert stat.S_IMODE(auth_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
