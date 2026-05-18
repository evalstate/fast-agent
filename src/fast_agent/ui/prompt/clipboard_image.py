"""Clipboard image paste helpers for prompt input."""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.mcp.mime_utils import guess_mime_type, is_image_mime_type

if TYPE_CHECKING:
    from PIL.Image import Image


class ClipboardImagePasteError(RuntimeError):
    """Raised when an image cannot be read from the clipboard."""


@dataclass(frozen=True, slots=True)
class PastedClipboardImage:
    path: Path
    width: int | None = None
    height: int | None = None


def paste_clipboard_image_to_temp_png() -> PastedClipboardImage:
    """Save the current clipboard image to a temporary PNG file."""
    try:
        return _paste_clipboard_image_with_pillow()
    except ClipboardImagePasteError as pillow_error:
        if _is_probably_wsl():
            try:
                return _paste_clipboard_image_with_wsl_powershell()
            except ClipboardImagePasteError as powershell_error:
                raise ClipboardImagePasteError(
                    f"{pillow_error}; WSL fallback failed: {powershell_error}"
                ) from powershell_error
        raise


def _paste_clipboard_image_with_pillow() -> PastedClipboardImage:
    try:
        from PIL import Image, ImageGrab
    except Exception as exc:  # noqa: BLE001
        raise ClipboardImagePasteError(f"Pillow clipboard support is unavailable: {exc}") from exc

    try:
        clipboard_data = ImageGrab.grabclipboard()
    except Exception as exc:  # noqa: BLE001
        raise ClipboardImagePasteError(f"clipboard unavailable: {exc}") from exc

    if isinstance(clipboard_data, Image.Image):
        return _save_image_as_png(clipboard_data)

    if isinstance(clipboard_data, list):
        image_path = _first_image_path(clipboard_data)
        if image_path is None:
            raise ClipboardImagePasteError("clipboard contains files, but none are images")
        try:
            with Image.open(image_path) as image:
                return _save_image_as_png(image)
        except Exception as exc:  # noqa: BLE001
            raise ClipboardImagePasteError(f"failed to open clipboard image file: {exc}") from exc

    raise ClipboardImagePasteError("no image on clipboard")


def _save_image_as_png(image: "Image") -> PastedClipboardImage:
    target_path = _new_clipboard_image_path()
    normalized = image.convert("RGBA") if image.mode not in ("RGB", "RGBA") else image.copy()
    fd = os.open(target_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as image_file:
            normalized.save(image_file, format="PNG")
    except Exception:
        try:
            target_path.unlink()
        except OSError:
            pass
        raise
    target_path.chmod(0o600)
    return PastedClipboardImage(path=target_path, width=normalized.width, height=normalized.height)


def _first_image_path(paths: list[str]) -> Path | None:
    for value in paths:
        path = Path(value).expanduser()
        if path.is_file() and is_image_mime_type(guess_mime_type(str(path))):
            return path
    return None


def _new_clipboard_image_path() -> Path:
    user_id: int | str = "user" if platform.system() == "Windows" else os.getuid()
    image_dir = Path(tempfile.gettempdir()) / f"fast-agent-clipboard-{user_id}"
    image_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    image_dir.chmod(0o700)
    return image_dir / f"fast-agent-clipboard-{uuid.uuid4().hex}.png"


def _is_probably_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in release or "wsl" in release


def _paste_clipboard_image_with_wsl_powershell() -> PastedClipboardImage:
    script = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "$img = Get-Clipboard -Format Image; "
        "if ($img -eq $null) { exit 1 }; "
        "$p = [System.IO.Path]::GetTempFileName(); "
        "$p = [System.IO.Path]::ChangeExtension($p, 'png'); "
        "$img.Save($p, [System.Drawing.Imaging.ImageFormat]::Png); "
        "Write-Output $p"
    )
    completed = _run_powershell_clipboard_script(script)
    windows_path = completed.stdout.strip()
    if not windows_path:
        raise ClipboardImagePasteError("PowerShell did not return an image path")

    image_path = _wslpath_to_linux(windows_path)
    if not image_path.is_file():
        raise ClipboardImagePasteError("PowerShell did not create an image file")

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            pasted = _save_image_as_png(image)
        try:
            image_path.unlink()
        except OSError:
            pass
        return pasted
    except Exception as exc:  # noqa: BLE001
        raise ClipboardImagePasteError(f"failed to inspect PowerShell image file: {exc}") from exc


def _wslpath_to_linux(windows_path: str) -> Path:
    try:
        completed = subprocess.run(
            ["wslpath", "-u", windows_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise ClipboardImagePasteError(f"failed to convert Windows path: {exc}") from exc
    return Path(completed.stdout.strip())


def _run_powershell_clipboard_script(script: str) -> subprocess.CompletedProcess[str]:
    commands = (["powershell.exe"], ["pwsh"], ["powershell"])
    last_error = "PowerShell unavailable"
    for command in commands:
        try:
            return subprocess.run(
                [*command, "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
    raise ClipboardImagePasteError(last_error)
