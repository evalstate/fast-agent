from __future__ import annotations

from fast_agent.utils import shell_detection


def test_default_shell_command_prefers_windows_candidate(monkeypatch) -> None:
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Windows")
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    def fake_which(name: str) -> str | None:
        return r"C:\Program Files\PowerShell\7\pwsh.exe" if name == "pwsh" else None

    monkeypatch.setattr(shell_detection.shutil, "which", fake_which)

    assert shell_detection.default_shell_command() == r"C:\Program Files\PowerShell\7\pwsh.exe"


def test_default_shell_command_uses_comspec_when_no_windows_candidate(monkeypatch) -> None:
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Windows")
    monkeypatch.setattr(shell_detection.shutil, "which", lambda _name: None)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    assert shell_detection.default_shell_command() == r"C:\Windows\System32\cmd.exe"


def test_shell_runtime_info_uses_shell_env_on_posix(monkeypatch, tmp_path) -> None:
    shell_path = tmp_path / "custom-sh"
    shell_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SHELL", str(shell_path))

    assert shell_detection.shell_runtime_info() == {
        "name": "custom-sh",
        "path": str(shell_path),
    }


def test_shell_runtime_info_strips_shell_env_on_posix(monkeypatch, tmp_path) -> None:
    shell_path = tmp_path / "custom-sh"
    shell_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SHELL", f" {shell_path} ")

    assert shell_detection.shell_runtime_info() == {
        "name": "custom-sh",
        "path": str(shell_path),
    }


def test_shell_runtime_info_strips_quoted_shell_env_on_posix(monkeypatch, tmp_path) -> None:
    shell_path = tmp_path / "custom sh"
    shell_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Linux")
    monkeypatch.setenv("SHELL", f' " {shell_path} " ')

    assert shell_detection.shell_runtime_info() == {
        "name": "custom sh",
        "path": str(shell_path),
    }


def test_posix_shell_detection_uses_first_available_candidate(monkeypatch) -> None:
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Linux")
    monkeypatch.delenv("SHELL", raising=False)

    def fake_which(name: str) -> str | None:
        return "/usr/bin/zsh" if name == "zsh" else None

    monkeypatch.setattr(shell_detection.shutil, "which", fake_which)

    assert shell_detection.default_shell_command() == "/usr/bin/zsh"
    assert shell_detection.shell_runtime_info() == {"name": "zsh", "path": "/usr/bin/zsh"}


def test_shell_runtime_info_falls_back_to_generic_sh(monkeypatch) -> None:
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Linux")
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.setattr(shell_detection.shutil, "which", lambda _name: None)

    assert shell_detection.shell_runtime_info() == {"name": "sh", "path": None}


def test_shell_runtime_info_uses_windows_basename_for_comspec(monkeypatch) -> None:
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Windows")
    monkeypatch.setattr(shell_detection.shutil, "which", lambda _name: None)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    assert shell_detection.shell_runtime_info() == {
        "name": "cmd.exe",
        "path": r"C:\Windows\System32\cmd.exe",
    }


def test_windows_shell_detection_uses_default_cmd_when_comspec_is_blank(monkeypatch) -> None:
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Windows")
    monkeypatch.setattr(shell_detection.shutil, "which", lambda _name: None)
    monkeypatch.setenv("COMSPEC", " ")

    assert shell_detection.shell_runtime_info() == {"name": "cmd.exe", "path": "cmd.exe"}


def test_windows_shell_detection_strips_quoted_comspec(monkeypatch) -> None:
    monkeypatch.setattr(shell_detection.platform, "system", lambda: "Windows")
    monkeypatch.setattr(shell_detection.shutil, "which", lambda _name: None)
    monkeypatch.setenv("COMSPEC", r'"C:\Program Files\Shell\cmd.exe"')

    assert shell_detection.shell_runtime_info() == {
        "name": "cmd.exe",
        "path": r"C:\Program Files\Shell\cmd.exe",
    }
