"""Utilities for detecting keyring availability and write access."""

from __future__ import annotations

import os
import secrets
import sys
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyringStatus:
    name: str
    available: bool
    writable: bool


_KEYRING_ACCESS_NOTICE_LOCK = threading.Lock()
_KEYRING_ACCESS_NOTICE_SHOWN = False


def maybe_print_keyring_access_notice(*, purpose: str | None = None) -> None:
    """Print a one-time note before first keyring access in interactive sessions.

    Some keyring backends may block while showing an OS keychain prompt.
    This hint makes startup pauses easier to understand for users.
    """

    global _KEYRING_ACCESS_NOTICE_SHOWN

    if _KEYRING_ACCESS_NOTICE_SHOWN:
        return

    suppress_notice = os.getenv("FAST_AGENT_KEYRING_NOTICE", "1").strip().lower()
    if suppress_notice in {"0", "false", "no", "off"}:
        return

    try:
        if not sys.stderr.isatty():
            return
    except Exception:
        return

    with _KEYRING_ACCESS_NOTICE_LOCK:
        if _KEYRING_ACCESS_NOTICE_SHOWN:
            return

        message = (
            "fast-agent is accessing your OS keyring for stored OAuth/API tokens. "
            "Some platforms may show a keychain prompt and pause startup until "
            "access is allowed."
        )
        if purpose:
            message = f"{message} ({purpose})"

        try:
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except Exception:
            return

        _KEYRING_ACCESS_NOTICE_SHOWN = True


def _probe_keyring_write(service: str) -> bool:
    try:
        maybe_print_keyring_access_notice(purpose="checking keyring availability")
        import keyring

        probe_key = f"probe:{secrets.token_urlsafe(8)}"
        keyring.set_password(service, probe_key, "probe")
        try:
            keyring.delete_password(service, probe_key)
        except Exception:
            # If deletion fails but set succeeded, still treat as writable.
            pass
        return True
    except Exception:
        return False


def get_keyring_status() -> KeyringStatus:
    try:
        maybe_print_keyring_access_notice(purpose="checking keyring backend")
        import keyring

        backend = keyring.get_keyring()
        name = getattr(backend, "name", backend.__class__.__name__)
        available = True
        try:
            from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

            available = not isinstance(backend, FailKeyring)
        except Exception:
            available = True
        writable = _probe_keyring_write("fast-agent-keyring-probe") if available else False
        return KeyringStatus(name=name, available=available, writable=writable)
    except Exception:
        return KeyringStatus(name="unavailable", available=False, writable=False)
