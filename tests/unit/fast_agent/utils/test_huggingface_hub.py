from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from fast_agent.utils import huggingface_hub

if TYPE_CHECKING:
    import pytest


def test_get_huggingface_hub_token_returns_none_when_module_import_fails(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    def failing_import(_name: str) -> object:
        raise ModuleNotFoundError("huggingface_hub")

    monkeypatch.setattr(huggingface_hub, "import_module", failing_import)

    assert huggingface_hub.get_huggingface_hub_token() is None


def test_get_huggingface_hub_token_returns_none_without_callable_get_token(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    monkeypatch.setattr(
        huggingface_hub,
        "import_module",
        lambda _name: SimpleNamespace(get_token="not-callable"),
    )

    assert huggingface_hub.get_huggingface_hub_token() is None


def test_get_huggingface_hub_token_normalizes_token_string(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    monkeypatch.setattr(
        huggingface_hub,
        "import_module",
        lambda _name: SimpleNamespace(get_token=lambda: " hf_token "),
    )

    assert huggingface_hub.get_huggingface_hub_token() == "hf_token"


def test_get_huggingface_hub_token_returns_none_for_blank_token(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    monkeypatch.setattr(
        huggingface_hub,
        "import_module",
        lambda _name: SimpleNamespace(get_token=lambda: "   "),
    )

    assert huggingface_hub.get_huggingface_hub_token() is None


def test_get_huggingface_hub_token_returns_none_when_provider_fails(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    def failing_get_token() -> object:
        raise RuntimeError("token store unavailable")

    monkeypatch.setattr(
        huggingface_hub,
        "import_module",
        lambda _name: SimpleNamespace(get_token=failing_get_token),
    )

    assert huggingface_hub.get_huggingface_hub_token() is None
