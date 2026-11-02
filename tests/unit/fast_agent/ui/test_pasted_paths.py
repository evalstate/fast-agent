from __future__ import annotations

import os
import shlex
from types import SimpleNamespace

import pytest
from PIL import Image
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

from fast_agent.ui.prompt.attachment_tokens import build_local_attachment_token
from fast_agent.ui.prompt.keybindings import create_keybindings
from fast_agent.ui.prompt.pasted_paths import (
    is_herdr_upload,
    pasted_image_tokens,
    unsupported_image_paste_message,
)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "uploaded image (1)'s.png"
    Image.new("RGB", (2, 2), "red").save(path)
    return path


@pytest.mark.parametrize("style", ["raw", "single", "double", "uri", "escaped"])
def test_pasted_uploaded_image_becomes_attachment(image_path, style) -> None:
    if os.name == "nt" and style in ("single", "escaped"):
        pytest.skip("POSIX shell quoting")
    raw = str(image_path)
    representations = {
        "raw": raw,
        "single": shlex.quote(raw),
        "double": f'"{raw}"',
        "uri": image_path.as_uri(),
        "escaped": raw.replace(" ", r"\ ")
        .replace("'", r"\'")
        .replace("(", r"\(")
        .replace(")", r"\)"),
    }
    assert pasted_image_tokens(representations[style]) == [build_local_attachment_token(image_path)]


def test_multiple_uploaded_images(image_path, tmp_path) -> None:
    second = tmp_path / "second.png"
    Image.new("RGB", (1, 1)).save(second)
    expected = [build_local_attachment_token(image_path), build_local_attachment_token(second)]
    assert pasted_image_tokens(f"{image_path}\r\n{second}\r\n") == expected
    assert pasted_image_tokens(f'"{image_path}" "{second}"') == expected


@pytest.mark.parametrize(
    "text",
    [
        "ordinary pasted text",
        "/missing/file.png",
        "https://example.com/image.png",
        "relative.png",
        "^file:/missing/image.png",
        "print('image.png')",
        "\x00",
        "",
    ],
)
def test_non_image_path_pastes_are_not_reinterpreted(text) -> None:
    assert pasted_image_tokens(text) == []


def test_mixed_text_or_non_image_file_stays_literal(image_path, tmp_path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not an image")
    assert pasted_image_tokens(str(text_file)) == []
    assert pasted_image_tokens(str(tmp_path)) == []
    assert pasted_image_tokens(f"look at {image_path}") == []
    assert pasted_image_tokens(f"{shlex.quote(str(image_path))} {text_file}") == []


def paste(buffer: Buffer, data: str, *, images: bool = True) -> None:
    kb = create_keybindings(enable_clipboard_image_paste=images)
    binding = next(b for b in kb.bindings if b.keys == (Keys.BracketedPaste,))
    # Minimal event stub; the real prompt-toolkit path is exercised below.
    binding.handler(SimpleNamespace(current_buffer=buffer, data=data))  # ty: ignore[invalid-argument-type]


def test_terminal_paste_attaches_at_cursor_without_changing_other_text(image_path) -> None:
    buffer = Buffer(document=Document("beforeafter", cursor_position=6))
    paste(buffer, str(image_path))
    token = build_local_attachment_token(image_path)
    assert buffer.text == f"before {token} after"
    assert buffer.document.text_after_cursor == "after"


@pytest.mark.parametrize("prefix", ["/attach ", "!cat ", "  /save ", "#agent "])
def test_command_paths_remain_literal(image_path, prefix) -> None:
    buffer = Buffer(document=Document(prefix, cursor_position=len(prefix)))
    paste(buffer, str(image_path))
    assert buffer.text == prefix + str(image_path)


def test_normal_paste_preserves_prompt_toolkit_line_ending_behavior() -> None:
    buffer = Buffer()
    paste(buffer, "first\r\nsecond\rthird")
    assert buffer.text == "first\nsecond\nthird"


def test_non_vision_models_keep_image_paths_literal_and_explain(image_path, monkeypatch) -> None:
    from fast_agent.ui.prompt import keybindings

    printed: list[str] = []
    monkeypatch.setattr(
        keybindings, "_print_styled", lambda message, _style: printed.append(message)
    )
    buffer = Buffer()
    paste(buffer, str(image_path), images=False)
    assert buffer.text == str(image_path)
    assert len(printed) == 1
    assert "cannot read images" in printed[0]
    assert "/model" in printed[0]


def test_non_vision_models_paste_ordinary_text_silently(monkeypatch) -> None:
    from fast_agent.ui.prompt import keybindings

    printed: list[str] = []
    monkeypatch.setattr(
        keybindings, "_print_styled", lambda message, _style: printed.append(message)
    )
    buffer = Buffer()
    paste(buffer, "first\r\nsecond", images=False)
    assert buffer.text == "first\nsecond"
    assert printed == []


def test_herdr_uploads_are_named_in_the_explanation(tmp_path) -> None:
    upload = tmp_path / "state" / "herdr-core" / "default" / "files" / "t7" / "shot.png"
    upload.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1)).save(upload)
    assert is_herdr_upload(upload)
    assert not is_herdr_upload(tmp_path / "shot.png")
    message = unsupported_image_paste_message(str(upload))
    assert message is not None and "from Herdr" in message
    assert unsupported_image_paste_message("plain text") is None


def test_unknown_command_that_is_a_file_explains_attach(image_path, tmp_path) -> None:
    from fast_agent.ui.interactive.command_dispatch import _unknown_command_message

    assert _unknown_command_message("/nope") == "Command not found: /nope"
    image_message = _unknown_command_message(str(image_path))
    assert "is an image file, not a command" in image_message
    assert f"/attach {shlex.quote(str(image_path))}" in image_message
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    assert _unknown_command_message(str(notes)) == (
        f"{notes} is a file, not a command. Use /attach {shlex.quote(str(notes))} to attach it."
    )


@pytest.mark.asyncio
async def test_terminal_bracketed_paste_through_prompt_toolkit(image_path) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=create_keybindings(enable_clipboard_image_paste=True),
        )
        pipe.send_text(f"\x1b[200~{image_path}\x1b[201~\r")
        result = await session.prompt_async()
    assert result.strip() == build_local_attachment_token(image_path)
