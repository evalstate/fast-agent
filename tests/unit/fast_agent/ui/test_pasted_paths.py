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
from fast_agent.ui.prompt.pasted_paths import pasted_image_tokens


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


def paste(buffer: Buffer, data: str) -> None:
    kb = create_keybindings(enable_clipboard_image_paste=True)
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


def test_non_vision_models_keep_default_paste_binding() -> None:
    kb = create_keybindings(enable_clipboard_image_paste=False)
    assert not any(b.keys == (Keys.BracketedPaste,) for b in kb.bindings)


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
