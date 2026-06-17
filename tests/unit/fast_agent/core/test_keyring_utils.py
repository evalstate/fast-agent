from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING

from fast_agent.core import keyring_utils

if TYPE_CHECKING:
    import pytest


def _reset_keyring_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keyring_utils, "_KEYRING_ACCESS_NOTICE_SHOWN", False)


def test_emit_keyring_access_notice_returns_false_when_emitter_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_keyring_notice(monkeypatch)

    def _raise_emit(_message: str) -> None:
        raise RuntimeError("no output")

    assert keyring_utils.emit_keyring_access_notice(emitter=_raise_emit) is False
    assert keyring_utils._KEYRING_ACCESS_NOTICE_SHOWN is False


def test_emit_keyring_access_notice_marks_notice_shown_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []
    _reset_keyring_notice(monkeypatch)

    assert keyring_utils.emit_keyring_access_notice(emitter=emitted.append) is True

    assert len(emitted) == 1
    assert keyring_utils._KEYRING_ACCESS_NOTICE_SHOWN is True


def test_get_keyring_status_returns_unavailable_when_keyring_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    _reset_keyring_notice(monkeypatch)

    assert keyring_utils.get_keyring_status() == keyring_utils.KeyringStatus(
        name="unavailable",
        available=False,
        writable=False,
    )


def test_get_keyring_status_reports_writable_non_fail_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keyring_module = ModuleType("keyring")

    class WorkingBackend:
        name = "working backend"

    def _get_keyring() -> WorkingBackend:
        return WorkingBackend()

    def _set_password(_service: str, _key: str, _value: str) -> None:
        return None

    def _delete_password(_service: str, _key: str) -> None:
        return None

    setattr(keyring_module, "get_keyring", _get_keyring)
    setattr(keyring_module, "set_password", _set_password)
    setattr(keyring_module, "delete_password", _delete_password)
    monkeypatch.setitem(sys.modules, "keyring", keyring_module)
    _reset_keyring_notice(monkeypatch)

    assert keyring_utils.get_keyring_status() == keyring_utils.KeyringStatus(
        name="working backend",
        available=True,
        writable=True,
    )


def test_get_keyring_status_treats_fail_backend_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keyring_module = ModuleType("keyring")
    backends_module = ModuleType("keyring.backends")
    fail_module = ModuleType("keyring.backends.fail")

    class FailKeyring:
        name = "fail backend"

    setattr(fail_module, "Keyring", FailKeyring)

    def _get_keyring() -> FailKeyring:
        return FailKeyring()

    setattr(keyring_module, "get_keyring", _get_keyring)
    setattr(keyring_module, "backends", backends_module)
    setattr(backends_module, "fail", fail_module)
    monkeypatch.setitem(sys.modules, "keyring", keyring_module)
    monkeypatch.setitem(sys.modules, "keyring.backends", backends_module)
    monkeypatch.setitem(sys.modules, "keyring.backends.fail", fail_module)
    _reset_keyring_notice(monkeypatch)

    assert keyring_utils.get_keyring_status() == keyring_utils.KeyringStatus(
        name="fail backend",
        available=False,
        writable=False,
    )
