"""Utilities for detecting keyring availability and write access."""

from __future__ import annotations

import secrets
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.utils.env import env_flag

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@dataclass(frozen=True)
class KeyringStatus:
    name: str
    available: bool
    writable: bool


_KEYRING_ACCESS_NOTICE_LOCK = threading.Lock()
_KEYRING_ACCESS_NOTICE_SHOWN = False


def format_keyring_access_notice(*, purpose: str | None = None) -> str:
    """Return the standard one-time keyring access notice."""
    message = (
        "fast-agent is accessing the OS keyring for stored tokens. "
        "Some platforms may pause and show a prompt."
    )
    if purpose:
        return f"{message} ({purpose})"
    return message


def _keyring_access_notice_enabled() -> bool:
    return env_flag("FAST_AGENT_KEYRING_NOTICE", default=True)


def emit_keyring_access_notice(
    *,
    purpose: str | None = None,
    emitter: Callable[[str], None] | None = None,
) -> bool:
    """Emit the one-time keyring access notice through the supplied emitter."""
    global _KEYRING_ACCESS_NOTICE_SHOWN

    if _KEYRING_ACCESS_NOTICE_SHOWN or not _keyring_access_notice_enabled():
        return False

    with _KEYRING_ACCESS_NOTICE_LOCK:
        if _KEYRING_ACCESS_NOTICE_SHOWN or not _keyring_access_notice_enabled():
            return False

        message = format_keyring_access_notice(purpose=purpose)

        if emitter is None:
            if not _stderr_is_tty():
                return False

            def _stderr_emitter(text: str) -> None:
                sys.stderr.write(f"{text}\n")
                sys.stderr.flush()

            emitter = _stderr_emitter

        with suppress(Exception):
            emitter(message)
            _KEYRING_ACCESS_NOTICE_SHOWN = True
            return True

        return False


def _stderr_is_tty() -> bool:
    with suppress(Exception):
        return sys.stderr.isatty()
    return False


def maybe_print_keyring_access_notice(*, purpose: str | None = None) -> None:
    """Print a one-time note before first keyring access in interactive sessions."""
    emit_keyring_access_notice(purpose=purpose)


def _probe_keyring_write(service: str) -> bool:
    try:
        maybe_print_keyring_access_notice(purpose="checking keyring availability")
        import keyring

        probe_key = f"probe:{secrets.token_urlsafe(8)}"
        keyring.set_password(service, probe_key, "probe")
        # If deletion fails but set succeeded, still treat the backend as writable.
        with suppress(Exception):
            keyring.delete_password(service, probe_key)
        return True
    except Exception:
        return False


def _is_fail_keyring_backend(backend: object) -> bool:
    with suppress(Exception):
        from keyring.backends.fail import Keyring as FailKeyring

        return isinstance(backend, FailKeyring)
    return False


def _keyring_backend_name(backend: object) -> str:
    name = getattr(backend, "name", None)
    return name if isinstance(name, str) else backend.__class__.__name__


def _load_keyring_module() -> ModuleType | None:
    with suppress(Exception):
        import keyring

        return keyring
    return None


def get_keyring_status() -> KeyringStatus:
    maybe_print_keyring_access_notice(purpose="checking keyring backend")
    keyring = _load_keyring_module()
    if keyring is None:
        return KeyringStatus(name="unavailable", available=False, writable=False)

    try:
        backend = keyring.get_keyring()
        name = _keyring_backend_name(backend)
        available = not _is_fail_keyring_backend(backend)
        writable = _probe_keyring_write("fast-agent-keyring-probe") if available else False
        return KeyringStatus(name=name, available=available, writable=writable)
    except Exception:
        return KeyringStatus(name="unavailable", available=False, writable=False)
