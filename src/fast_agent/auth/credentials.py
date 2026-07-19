from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, Field

from fast_agent.constants import FAST_AGENT_AUTH_FILE
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.keyring_utils import get_keyring_status, maybe_print_keyring_access_notice

AUTH_KEYRING_SERVICE = "fast-agent-provider-auth"
AuthSource = Literal["file", "keyring"]


class OAuthCredential(BaseModel):
    type: Literal["oauth"] = "oauth"
    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    token_type: str = "Bearer"
    scope: str | None = None


class AuthDocument(BaseModel):
    version: Literal[1] = 1
    providers: dict[str, OAuthCredential] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StoredCredential:
    credential: OAuthCredential
    source: AuthSource


def default_auth_path() -> Path:
    return Path.home() / ".fast-agent" / "auth.json"


def configured_auth_path() -> Path | None:
    value = os.getenv(FAST_AGENT_AUTH_FILE, "").strip()
    return Path(value).expanduser() if value else None


def is_codex_cli_auth_path(path: Path) -> bool:
    """Return whether path is reserved for a Codex CLI credential file."""
    protected_paths = {Path.home() / ".codex" / "auth.json"}
    codex_home = os.getenv("CODEX_HOME", "").strip()
    if codex_home:
        protected_paths.add(Path(codex_home).expanduser() / "auth.json")
    explicit = os.getenv("CODEX_AUTH_JSON_PATH", "").strip()
    if explicit:
        protected_paths.add(Path(explicit).expanduser())

    normalized = os.path.normcase(str(path.expanduser().resolve()))
    return any(
        normalized == os.path.normcase(str(protected_path.resolve()))
        for protected_path in protected_paths
    )


def _reject_codex_cli_auth_path(provider: str, path: Path) -> None:
    if provider == "codex" and is_codex_cli_auth_path(path):
        raise ProviderKeyError(
            "Codex CLI auth file is read-only",
            "Choose a separate fast-agent provider credential file. Codex CLI "
            "auth files are external sources and are never modified.",
        )


def _load_document(path: Path) -> AuthDocument:
    if not path.exists():
        return AuthDocument()
    return AuthDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _ensure_parent_directory(path: Path, *, private: bool) -> None:
    parent_exists = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if private and not parent_exists:
        with suppress(OSError):
            path.parent.chmod(0o700)


def _write_document(
    path: Path,
    document: AuthDocument,
    *,
    private_parent: bool = False,
) -> None:
    _ensure_parent_directory(path, private=private_parent)
    existing_mode: int | None = None
    try:
        existing_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        pass
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(document.model_dump_json(indent=2))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        with suppress(OSError):
            temporary_path.chmod(existing_mode if existing_mode is not None else 0o600)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _file_lock(path: Path) -> FileLock:
    return FileLock(path.with_suffix(f"{path.suffix}.lock"))


@contextmanager
def credential_refresh_lock(provider: str):
    configured_path = configured_auth_path()
    path = configured_path or default_auth_path()
    _ensure_parent_directory(path, private=configured_path is None)
    with FileLock(path.parent / f".{provider}.refresh.lock"):
        yield


def _read_file_credential(path: Path, provider: str) -> OAuthCredential | None:
    return _load_document(path).providers.get(provider)


def _modify_file_credential(
    path: Path,
    provider: str,
    credential: OAuthCredential | None,
    *,
    private_parent: bool = False,
) -> None:
    _ensure_parent_directory(path, private=private_parent)
    with _file_lock(path):
        document = _load_document(path)
        providers = dict(document.providers)
        if credential is None:
            providers.pop(provider, None)
        else:
            providers[provider] = credential
        _write_document(path, document.model_copy(update={"providers": providers}))


def _read_keyring_credential(provider: str) -> OAuthCredential | None:
    try:
        maybe_print_keyring_access_notice(purpose="loading provider OAuth credentials")
        import keyring

        payload = keyring.get_password(AUTH_KEYRING_SERVICE, provider)
        return OAuthCredential.model_validate_json(payload) if payload else None
    except Exception:
        return None


def _write_keyring_credential(provider: str, credential: OAuthCredential) -> None:
    maybe_print_keyring_access_notice(purpose="saving provider OAuth credentials")
    import keyring

    keyring.set_password(AUTH_KEYRING_SERVICE, provider, credential.model_dump_json())


def _delete_keyring_credential(provider: str) -> bool:
    try:
        maybe_print_keyring_access_notice(purpose="clearing provider OAuth credentials")
        import keyring

        keyring.delete_password(AUTH_KEYRING_SERVICE, provider)
        return True
    except Exception:
        return False


def load_oauth_credential(provider: str) -> StoredCredential | None:
    explicit_path = configured_auth_path()
    if explicit_path is not None:
        credential = _read_file_credential(explicit_path, provider)
        return StoredCredential(credential, "file") if credential else None

    credential = _read_keyring_credential(provider)
    if credential:
        return StoredCredential(credential, "keyring")

    credential = _read_file_credential(default_auth_path(), provider)
    return StoredCredential(credential, "file") if credential else None


def save_oauth_credential(
    provider: str,
    credential: OAuthCredential,
    *,
    source: AuthSource | None = None,
) -> AuthSource:
    explicit_path = configured_auth_path()
    if explicit_path is not None:
        _reject_codex_cli_auth_path(provider, explicit_path)
        _modify_file_credential(explicit_path, provider, credential)
        return "file"

    if source == "file":
        _modify_file_credential(
            default_auth_path(), provider, credential, private_parent=True
        )
        return "file"

    status = get_keyring_status()
    if source == "keyring" or status.writable:
        try:
            _write_keyring_credential(provider, credential)
            return "keyring"
        except Exception:
            if source == "keyring":
                raise

    _modify_file_credential(default_auth_path(), provider, credential, private_parent=True)
    return "file"


def delete_oauth_credential(provider: str) -> bool:
    explicit_path = configured_auth_path()
    if explicit_path is not None:
        if provider == "codex" and is_codex_cli_auth_path(explicit_path):
            return False
        if _read_file_credential(explicit_path, provider) is None:
            return False
        _modify_file_credential(explicit_path, provider, None)
        return True

    removed = _delete_keyring_credential(provider)
    path = default_auth_path()
    if _read_file_credential(path, provider) is not None:
        _modify_file_credential(path, provider, None, private_parent=True)
        removed = True
    return removed


def export_oauth_credential(provider: str, credential: OAuthCredential, path: Path) -> None:
    expanded_path = path.expanduser()
    _reject_codex_cli_auth_path(provider, expanded_path)
    _write_document(expanded_path, AuthDocument(providers={provider: credential}))
