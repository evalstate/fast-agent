from __future__ import annotations

import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageGrab

from fast_agent.ui.prompt import clipboard_image
from fast_agent.ui.prompt.clipboard_image import (
    ClipboardImagePasteError,
    paste_clipboard_image_to_temp_png,
)


def test_paste_clipboard_image_to_temp_png_saves_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    target_path = tmp_path / "pasted.png"
    image = Image.new("RGB", (2, 3), color=(255, 0, 0))

    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda: image)
    monkeypatch.setattr(clipboard_image, "_new_clipboard_image_path", lambda: target_path)

    pasted = paste_clipboard_image_to_temp_png()

    assert pasted.path == target_path
    assert pasted.width == 2
    assert pasted.height == 3
    assert target_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_clipboard_image_temp_path_uses_private_permissions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    target_path = clipboard_image._new_clipboard_image_path()
    image = Image.new("RGB", (2, 3), color=(255, 0, 0))
    pasted = clipboard_image._save_image_as_png(image)

    assert pasted.path.parent == target_path.parent
    assert stat.S_IMODE(pasted.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(pasted.path.stat().st_mode) == 0o600


def test_paste_clipboard_image_to_temp_png_raises_without_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda: None)
    monkeypatch.setattr(clipboard_image, "_is_probably_wsl", lambda: False)

    with pytest.raises(ClipboardImagePasteError, match="no image"):
        paste_clipboard_image_to_temp_png()


def test_wsl_powershell_fallback_maps_windows_temp_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image_path = tmp_path / "windows-temp.png"
    secure_path = tmp_path / "secure.png"
    Image.new("RGB", (4, 5), color=(0, 0, 255)).save(image_path, format="PNG")

    def fake_run_powershell_clipboard_script(_script: str):
        return SimpleNamespace(stdout="C:\\Users\\me\\AppData\\Local\\Temp\\clip.png\n")

    monkeypatch.setattr(
        clipboard_image,
        "_run_powershell_clipboard_script",
        fake_run_powershell_clipboard_script,
    )
    monkeypatch.setattr(
        clipboard_image,
        "_wslpath_to_linux",
        lambda _windows_path: Path(image_path),
    )
    monkeypatch.setattr(clipboard_image, "_new_clipboard_image_path", lambda: secure_path)

    pasted = clipboard_image._paste_clipboard_image_with_wsl_powershell()

    assert pasted.path == secure_path
    assert pasted.width == 4
    assert pasted.height == 5
    assert not image_path.exists()
