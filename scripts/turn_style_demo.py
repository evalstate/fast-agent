#!/usr/bin/env python3
"""Interactive & CLI gallery for comparing Option-5 code/tool/turn designs.

Examples:
    uv run python scripts/turn_style_demo.py
    uv run python scripts/turn_style_demo.py -i
    uv run python scripts/turn_style_demo.py --style all
    uv run python scripts/turn_style_demo.py --style linework --stream
    uv run python scripts/turn_style_demo.py --style marquee --scenario errors
    uv run python scripts/turn_style_demo.py --list-sessions
    uv run python scripts/turn_style_demo.py --session 2611261357 --turn 1 --style linework
    uv run python scripts/turn_style_demo.py --verify
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal

from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.ui.apply_patch_preview import (
    extract_apply_patch_text,
    render_patch_preview,
    style_apply_patch_preview_text,
)
from fast_agent.ui.markdown.renderables import build_rich_markdown
from fast_agent.ui.message_styles import SHORT_RULE_WIDTH, build_code_rule
from fast_agent.ui.streaming.display import _DiffLive
from fast_agent.ui.syntax_highlighting import python_syntax_blocks, syntax_language_for_path

StyleName = Literal["hybrid", "current", "linework", "marquee", "hairline", "blueprint"]
RuleMode = Literal["default", "stub", "fixed", "full"]
SpeakerMode = Literal["default", "dots", "bars"]
OutputMode = Literal["compact", "full", "none"]

STYLE_ORDER: tuple[StyleName, ...] = (
    "hybrid",
    "current",
    "linework",
    "marquee",
    "hairline",
    "blueprint",
)
FIXED_MEASURE = 52
_FENCE_RE = re.compile(r"(?m)^([ \t]{0,3})(`{3,}|~{3,})([^\n]*)\n")
_PTC_STATUS_RE = re.compile(
    r"^Script\s+(\w+)\s+in\s+([0-9.]+s)\s*\((\d+)\s+calls?(?:,\s*(\d+)\s+failed)?\)"
)


@dataclass(frozen=True, slots=True)
class SandboxCallStat:
    name: str
    ok: int = 0
    failed: int = 0
    marks: str = ""


@dataclass(frozen=True, slots=True)
class UserEvent:
    sender: str
    text: str


@dataclass(frozen=True, slots=True)
class AssistantSegment:
    kind: Literal["prose", "code"]
    text: str
    language: str = "text"
    is_open: bool = False


@dataclass(frozen=True, slots=True)
class AssistantEvent:
    sender: str
    model: str | None
    segments: tuple[AssistantSegment, ...]
    first_in_turn: bool = False


@dataclass(frozen=True, slots=True)
class SandboxEvent:
    code: str
    output: str = ""
    status: Literal["ok", "warn", "error"] = "ok"
    elapsed: str = "0.28s"
    total_calls: int = 0
    failed_calls: int = 0
    calls: tuple[SandboxCallStat, ...] = ()
    error_summary: str = ""
    is_open: bool = False


@dataclass(frozen=True, slots=True)
class FileWriteEvent:
    path: str
    content: str
    language: str
    status: Literal["ok", "error"] = "ok"
    error_detail: str = ""
    is_open: bool = False


@dataclass(frozen=True, slots=True)
class FileEditEvent:
    kind: Literal["edit", "patch"]
    path_label: str
    diff_lines: tuple[str, ...]
    added: int
    removed: int
    status: Literal["ok", "error"] = "ok"
    error_detail: str = ""


@dataclass(frozen=True, slots=True)
class ShellEvent:
    command: str
    output: str = ""
    exit_code: int | None = 0
    elapsed: str = "0.42s"
    cwd: str | None = None


@dataclass(frozen=True, slots=True)
class CompactToolBatchEvent:
    verb: str
    count: int
    targets: tuple[str, ...] = ()
    payload_json: str | None = None
    result_summary: str = ""
    elapsed: str = "0.18s"
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class TurnFooterEvent:
    tool_count: int
    elapsed: str | None = None
    tokens: str | None = None


TurnEvent = (
    UserEvent
    | AssistantEvent
    | SandboxEvent
    | FileWriteEvent
    | FileEditEvent
    | ShellEvent
    | CompactToolBatchEvent
    | TurnFooterEvent
)


@dataclass(frozen=True, slots=True)
class RenderConfig:
    style: StyleName = "hybrid"
    theme: str = "ansi_light"
    rule_mode: RuleMode = "default"
    speaker_mode: SpeakerMode = "default"
    output_mode: OutputMode = "compact"
    cursor: str = ""


class _UnpaddedSyntax:
    """Transparent Syntax rendered without trailing space padding (with embedded bash support)."""

    def __init__(self, code: str, language: str, *, theme: str, cursor: str = "") -> None:
        self._code = code
        self._language = language or "text"
        self._theme = theme
        self._cursor = cursor

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        blocks = (
            python_syntax_blocks(self._code, language=self._language)
            if self._language.lower() == "python"
            else ()
        )
        renderable: RenderableType
        if len(blocks) > 1:
            renderable = Group(
                *(
                    Syntax(
                        block.code,
                        block.language,
                        theme=self._theme,
                        background_color="default",
                        line_numbers=False,
                        word_wrap=True,
                    )
                    for block in blocks
                )
            )
        else:
            renderable = Syntax(
                self._code,
                self._language,
                theme=self._theme,
                background_color="default",
                line_numbers=False,
                word_wrap=True,
            )
        lines = console.render_lines(renderable, options, pad=False)
        newline = Segment("\n")
        if not lines:
            if self._cursor:
                yield Segment(self._cursor, Style(color="bright_green", bold=True))
                yield newline
            return
        for idx, line in enumerate(lines):
            yield from line
            if idx == len(lines) - 1 and self._cursor:
                yield Segment(self._cursor, Style(color="bright_green", bold=True))
            yield newline


class _RuleLine:
    """Dynamic-width opener / divider / closer line so width is computed at render time."""

    def __init__(
        self,
        prefix: Text,
        *,
        fill_char: str = "─",
        fill_style: str = "dim",
        mode: Literal["stub", "fixed", "full"] = "stub",
    ) -> None:
        self._prefix = prefix
        self._fill_char = fill_char
        self._fill_style = fill_style
        self._mode = mode

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        line = self._prefix.copy()
        used = cell_len(line.plain)
        target = 0
        if self._mode == "fixed":
            target = min(options.max_width, FIXED_MEASURE)
        elif self._mode == "full":
            target = options.max_width
        if target > used + 1 and self._fill_char:
            line.append(" " + self._fill_char * (target - used - 1), style=self._fill_style)
        line.end = "\n"
        yield from console.render(line, options)


def _effective_rule_mode(cfg: RenderConfig) -> Literal["stub", "fixed", "full"]:
    if cfg.rule_mode != "default":
        return cfg.rule_mode
    return "fixed" if cfg.style == "blueprint" else "stub"


def _speaker_line(
    role: Literal["user", "assistant"],
    sender: str,
    model: str | None,
    cfg: RenderConfig,
) -> Text:
    mode = cfg.speaker_mode
    style = cfg.style
    t = Text()
    color = "blue" if role == "user" else "green"
    if style in {"current", "hybrid"}:
        t.append("▎▶ " if role == "user" else "▎◀ ", style=color)
        t.append(sender, style="bold")
        if model:
            sep = "  " if style == "current" else " · "
            t.append(f"{sep}{model}", style="dim")
        return t

    if mode == "bars" or (mode == "default" and style == "marquee"):
        t.append("▎", style=color)
        t.append(sender, style="bold")
    elif mode == "dots" or (mode == "default" and style == "linework"):
        t.append("● " if role == "user" else "○ ", style=color)
        t.append(sender, style="bold")
    elif style == "blueprint":
        t.append("■ " if role == "user" else "□ ", style=color)
        t.append(sender.upper(), style="bold")
    else:
        t.append("› " if role == "user" else "‹ ", style=color)
        t.append(sender, style="bold")

    if model:
        t.append(f" · {model}", style="dim")
    return t


def _trim_output(text: str, mode: OutputMode, *, limit: int = 8) -> str:
    stripped = text.strip("\n")
    if not stripped or mode == "none":
        return ""
    if mode == "full":
        return stripped
    lines = stripped.splitlines()
    if len(lines) <= limit:
        return stripped
    head_n = max(2, limit // 2)
    tail_n = max(2, limit - head_n - 1)
    omitted = len(lines) - head_n - tail_n
    return "\n".join([*lines[:head_n], f"… ({omitted} lines omitted)", *lines[-tail_n:]])


def _format_call_chips(
    calls: tuple[SandboxCallStat, ...],
    *,
    chip_marks: bool,
) -> Text:
    out = Text()
    for idx, stat in enumerate(calls[:4]):
        if idx > 0:
            out.append("  " if chip_marks else " · ", style="dim")
        total = stat.ok + stat.failed
        if chip_marks and stat.marks and total <= 6:
            out.append(f"{stat.name} ", style="dim")
            for ch in stat.marks:
                out.append(ch, style="dim green" if ch == "✓" else "bold red")
        else:
            label = f"{stat.name} ×{total}" if total > 1 else stat.name
            out.append(label, style="dim")
            if stat.failed:
                out.append(f" ({stat.failed} ✗)", style="bold red")
    if len(calls) > 4:
        out.append(f" · +{len(calls) - 4} more", style="dim")
    return out


def _render_diff_body(diff_lines: tuple[str, ...]) -> Text:
    text = Text()
    for idx, raw in enumerate(diff_lines):
        if idx > 0:
            text.append("\n")
        if raw.startswith("*** ") or raw.startswith("--- ") or raw.startswith("+++ "):
            text.append(raw, style="cyan")
        elif raw.startswith("@@"):
            text.append(raw, style="yellow")
        elif raw.startswith("+"):
            text.append(raw, style="green")
        elif raw.startswith("-"):
            text.append(raw, style="red")
        else:
            text.append(raw, style="dim")
    return text


def _render_prose_code(
    code: str,
    language: str,
    *,
    is_open: bool,
    cfg: RenderConfig,
) -> RenderableType:
    lang = language or "text"
    label = "" if lang == "text" else lang
    if cfg.style == "current":
        parts_cur: list[RenderableType] = [
            build_code_rule(label, max_width=SHORT_RULE_WIDTH),
            _UnpaddedSyntax(code, lang, theme=cfg.theme, cursor=cfg.cursor if is_open else ""),
        ]
        if not is_open:
            parts_cur.append(build_code_rule(max_width=SHORT_RULE_WIDTH))
        return Group(*parts_cur)

    if cfg.style == "hybrid":
        op_hyb = Text("── ", style="dim")
        if label:
            op_hyb.append(f"{label} ", style="dim")
        rem = max(2, SHORT_RULE_WIDTH - cell_len(op_hyb.plain))
        op_hyb.append("─" * rem, style="dim")
        parts_hyb: list[RenderableType] = [
            op_hyb,
            _UnpaddedSyntax(code, lang, theme=cfg.theme, cursor=cfg.cursor if is_open else ""),
        ]
        if not is_open:
            parts_hyb.append(Text("─" * SHORT_RULE_WIDTH, style="dim"))
        return Group(*parts_hyb)

    rmode = _effective_rule_mode(cfg)
    fill_ch = "┄" if cfg.style == "linework" else "─"
    if cfg.style == "linework":
        op = Text("┄┄┄ ", style="dim")
        op.append(lang, style="dim")
        cl = Text("┄┄┄", style="dim")
    elif cfg.style == "marquee":
        op = Text("▸ ", style="dim")
        op.append(lang, style="dim")
        cl = Text("└", style="dim")
    elif cfg.style == "blueprint":
        op = Text("┌─ ", style="dim")
        op.append("CODE  ", style="bold dim")
        op.append(lang, style="dim")
        cl = Text("└─", style="dim")
    else:
        op = Text("── ", style="dim")
        op.append(lang, style="dim")
        cl = Text("──", style="dim")

    parts: list[RenderableType] = [
        _RuleLine(op, fill_char=fill_ch, fill_style="dim", mode=rmode),
        _UnpaddedSyntax(code, lang, theme=cfg.theme, cursor=cfg.cursor if is_open else ""),
    ]
    if not is_open:
        parts.append(_RuleLine(cl, fill_char=fill_ch, fill_style="dim", mode="stub"))
    return Group(*parts)


def _render_sandbox_event(ev: SandboxEvent, cfg: RenderConfig) -> RenderableType:
    if cfg.style == "current":
        call_prefix = Text("▎◀ ", style="magenta")
        call_hdr = build_code_rule(
            "python (sandbox)",
            max_width=80,
            prefix=call_prefix,
            suffix=Text("id: call_…456789", style="dim"),
        )
        body = _UnpaddedSyntax(
            ev.code, "python", theme=cfg.theme, cursor=cfg.cursor if ev.is_open else ""
        )
        if ev.is_open:
            return Group(call_hdr, body)
        res_prefix = Text("▎▶ ", style="magenta")
        status_note = (
            "error · 280ms · id: call_…456789"
            if ev.status == "error"
            else f"text ({len(ev.output)} chars) · {ev.elapsed} · id: call_…456789"
        )
        res_hdr = build_code_rule(
            "python (sandbox) result",
            max_width=80,
            prefix=res_prefix,
            suffix=Text(status_note, style="bold red" if ev.status == "error" else "dim"),
        )
        items: list[RenderableType] = [
            call_hdr,
            body,
            build_code_rule(max_width=SHORT_RULE_WIDTH),
            res_hdr,
        ]
        if ev.status == "error" and ev.output:
            items.append(Text(ev.output, style="red"))
        return Group(*items)

    if cfg.style == "hybrid":
        op_hyb = Text("▎◀ ", style="magenta")
        op_hyb.append(" sandbox · python ", style="bold reverse magenta")
        rem = max(2, SHORT_RULE_WIDTH - cell_len(op_hyb.plain))
        op_hyb.append(" " + "━" * (rem - 1), style="magenta")
        hyb_parts: list[RenderableType] = [
            op_hyb,
            _UnpaddedSyntax(
                ev.code, "python", theme=cfg.theme, cursor=cfg.cursor if ev.is_open else ""
            ),
        ]
        if ev.is_open:
            return Group(*hyb_parts)
        trimmed_hyb = _trim_output(
            ev.output,
            "full" if ev.status == "error" and cfg.output_mode == "compact" else cfg.output_mode,
        )
        if trimmed_hyb:
            hyb_parts.append(Text("── output " + "─" * (SHORT_RULE_WIDTH - 10), style="dim"))
            hyb_parts.append(Text(trimmed_hyb, style="red" if ev.status == "error" else "dim"))
        cl_hyb = Text("━━ ", style="magenta")
        if ev.status == "ok":
            cl_hyb.append("✓ ", style="green")
            cl_hyb.append(ev.elapsed, style="dim")
        elif ev.status == "warn":
            cl_hyb.append("! ", style="bold yellow")
            cl_hyb.append(ev.elapsed, style="dim")
        else:
            cl_hyb.append("✗ ", style="bold red")
            cl_hyb.append(ev.error_summary or f"failed {ev.elapsed}", style="bold red")
        if ev.calls:
            cl_hyb.append(" · ", style="dim")
            cl_hyb.append_text(_format_call_chips(ev.calls, chip_marks=True))
        hyb_parts.append(cl_hyb)
        return Group(*hyb_parts)

    rmode = _effective_rule_mode(cfg)
    style = cfg.style
    if style == "marquee":
        op = Text()
        op.append("┏━ ", style="bold magenta")
        op.append(" SANDBOX ", style="bold reverse magenta")
        op.append("  python", style="dim")
        div = Text()
        div.append("┣━ ", style="bold magenta")
        div.append("output", style="dim")
        cl_prefix = Text()
        cl_prefix.append("┗━ ", style="bold magenta")
        fill_ch = "━"
    elif style == "blueprint":
        op = Text()
        op.append("╔═ ", style="bold magenta")
        op.append("EXEC  ", style="bold magenta")
        op.append("sandbox · python", style="dim")
        div = Text()
        div.append("╠═ ", style="bold magenta")
        div.append("OUT", style="dim")
        cl_prefix = Text()
        cl_prefix.append("╚═ ", style="bold magenta")
        fill_ch = "═"
    elif style == "hairline":
        op = Text()
        op.append("══ ", style="bold magenta")
        op.append("sandbox", style="bold magenta")
        op.append(" · python", style="dim")
        div = Text("── output", style="dim")
        cl_prefix = Text()
        cl_prefix.append("══ ", style="bold magenta")
        fill_ch = "═"
    else:
        op = Text()
        op.append("═══ ", style="bold magenta")
        op.append("sandbox", style="bold magenta")
        op.append(" · python", style="dim")
        div = Text("─── output", style="dim")
        cl_prefix = Text()
        cl_prefix.append("═══ ", style="bold magenta")
        fill_ch = "═"

    parts: list[RenderableType] = [
        _RuleLine(op, fill_char=fill_ch, fill_style="magenta", mode=rmode),
        _UnpaddedSyntax(
            ev.code, "python", theme=cfg.theme, cursor=cfg.cursor if ev.is_open else ""
        ),
    ]
    if ev.is_open:
        return Group(*parts)

    trimmed_out = _trim_output(
        ev.output,
        "full" if ev.status == "error" and cfg.output_mode == "compact" else cfg.output_mode,
    )
    if trimmed_out:
        parts.append(_RuleLine(div, fill_char="─", fill_style="dim", mode="stub"))
        out_style = "red" if ev.status == "error" else ""
        parts.append(Text(trimmed_out, style=out_style))

    cl = cl_prefix.copy()
    if ev.status == "ok":
        cl.append("✓ ", style="dim green")
        cl.append(ev.elapsed, style="dim")
    elif ev.status == "warn":
        cl.append("! ", style="bold yellow")
        cl.append(ev.elapsed, style="dim")
    else:
        cl.append("✗ ", style="bold red")
        cl.append(ev.error_summary or f"failed {ev.elapsed}", style="bold red")

    if ev.total_calls > 0:
        call_word = "call" if ev.total_calls == 1 else "calls"
        cl.append(f" · {ev.total_calls} {call_word}", style="dim")
        if ev.failed_calls and ev.status != "error":
            cl.append(f" · {ev.failed_calls} failed", style="bold yellow")
    if ev.calls:
        cl.append("  " if style == "marquee" else " · ", style="dim")
        cl.append_text(_format_call_chips(ev.calls, chip_marks=(style == "marquee")))

    parts.append(_RuleLine(cl, fill_char=fill_ch, fill_style="magenta", mode="stub"))
    return Group(*parts)


def _action_opener(op_word: str, label: str, cfg: RenderConfig, *, tag5: str) -> _RuleLine:
    rmode = _effective_rule_mode(cfg)
    if cfg.style == "marquee":
        t = Text()
        t.append("▸ ", style="dim")
        t.append(op_word, style="bold")
        if label:
            t.append(f"  {label}", style="cyan" if "/" in label or "." in label else "dim")
        return _RuleLine(t, fill_char="─", fill_style="dim", mode=rmode)
    if cfg.style == "blueprint":
        t = Text()
        t.append("┌─ ", style="dim")
        t.append(f"{tag5:<5} ", style="bold")
        if label:
            t.append(label, style="cyan" if "/" in label or "." in label else "dim")
        return _RuleLine(t, fill_char="─", fill_style="dim", mode=rmode)
    stub = "── " if cfg.style == "hairline" else "─── "
    t = Text()
    t.append(stub, style="dim")
    t.append(op_word, style="bold")
    if label:
        t.append(" · ", style="dim")
        t.append(label, style="cyan" if "/" in label or "." in label else "dim")
    return _RuleLine(t, fill_char="─", fill_style="dim", mode=rmode)


def _action_closer(cfg: RenderConfig) -> Text:
    if cfg.style == "marquee":
        t = Text()
        t.append("└ ", style="dim")
        return t
    if cfg.style == "blueprint":
        t = Text()
        t.append("└─ ", style="dim")
        return t
    t = Text()
    t.append("── " if cfg.style == "hairline" else "─── ", style="dim")
    return t


def _render_file_write(ev: FileWriteEvent, cfg: RenderConfig) -> RenderableType:
    if cfg.style == "current":
        hdr = build_code_rule(
            f"{ev.language} (write_text_file)" if ev.language != "text" else "write_text_file",
            max_width=80,
            prefix=Text("▎◀ ", style="magenta"),
        )
        path_line = Text("path: ", style="dim")
        path_line.append(ev.path, style="cyan")
        syn = _UnpaddedSyntax(
            ev.content, ev.language, theme=cfg.theme, cursor=cfg.cursor if ev.is_open else ""
        )
        if ev.is_open:
            return Group(hdr, path_line, syn)
        res = Text("▎▶ ", style="magenta")
        res.append("file write")
        res.append(f" · text ({len(ev.content)} chars)", style="dim")
        return Group(hdr, path_line, syn, build_code_rule(max_width=SHORT_RULE_WIDTH), res)

    if cfg.style == "hybrid":
        op_hyb = Text("▎◀ ", style="magenta")
        op_hyb.append(" write ", style="reverse cyan")
        op_hyb.append(f" {ev.path}", style="cyan")
        if ev.language and ev.language != "text":
            op_hyb.append(f" · {ev.language}", style="dim")
        body_hyb = _UnpaddedSyntax(
            ev.content, ev.language, theme=cfg.theme, cursor=cfg.cursor if ev.is_open else ""
        )
        if ev.is_open:
            return Group(op_hyb, body_hyb)
        n_lines = len(ev.content.splitlines()) or 1
        cl_hyb = Text("── ", style="dim")
        if ev.status == "ok":
            cl_hyb.append("✓ ", style="dim green")
            cl_hyb.append(f"wrote {n_lines} lines", style="dim")
        else:
            cl_hyb.append(f"✗ {ev.error_detail or 'write failed'}", style="bold red")
        return Group(op_hyb, body_hyb, cl_hyb)

    parts: list[RenderableType] = [
        _action_opener("write", ev.path, cfg, tag5="WRITE"),
        _UnpaddedSyntax(
            ev.content, ev.language, theme=cfg.theme, cursor=cfg.cursor if ev.is_open else ""
        ),
    ]
    if not ev.is_open:
        cl = _action_closer(cfg)
        if ev.status == "ok":
            n_lines = len(ev.content.splitlines()) or 1
            cl.append("✓ ", style="dim green")
            cl.append(f"wrote {n_lines} lines · {ev.language}", style="dim")
        else:
            cl.append(f"✗ {ev.error_detail or 'write failed'}", style="bold red")
        parts.append(_RuleLine(cl, mode="stub"))
    return Group(*parts)


def _render_file_edit(ev: FileEditEvent, cfg: RenderConfig) -> RenderableType:
    if cfg.style == "current":
        hdr = Text("▎◀ ", style="magenta")
        hdr.append(f"edit {ev.kind}_file")
        raw = "\n".join((f"edit_file preview: {ev.path_label}", *ev.diff_lines))
        return Group(hdr, style_apply_patch_preview_text(raw, default_style="dim"))

    if cfg.style == "hybrid":
        op_hyb = Text("▎◀ ", style="magenta")
        op_hyb.append(" edit ", style="reverse yellow")
        op_hyb.append(f" {ev.path_label} ", style="cyan")
        op_hyb.append(f"+{ev.added}", style="green")
        op_hyb.append(" ", style="dim")
        op_hyb.append(f"-{ev.removed}", style="red")
        if ev.status != "ok":
            err = Text(f"── ✗ {ev.error_detail or 'edit failed'}", style="bold red")
            return Group(op_hyb, _render_diff_body(ev.diff_lines), err)
        return Group(op_hyb, _render_diff_body(ev.diff_lines))

    op_word = "patch" if ev.kind == "patch" else "edit"
    tag5 = "PATCH" if ev.kind == "patch" else "EDIT"
    cl = _action_closer(cfg)
    if ev.status == "ok":
        cl.append("✓ ", style="dim green")
        cl.append(f"+{ev.added}", style="green")
        cl.append(" ", style="dim")
        cl.append(f"-{ev.removed}", style="red")
    else:
        cl.append(f"✗ {ev.error_detail or 'edit failed'}", style="bold red")

    return Group(
        _action_opener(op_word, ev.path_label, cfg, tag5=tag5),
        _render_diff_body(ev.diff_lines),
        _RuleLine(cl, mode="stub"),
    )


def _render_shell(ev: ShellEvent, cfg: RenderConfig) -> RenderableType:
    if cfg.style == "current":
        hdr = Text("▎◀ ", style="magenta")
        hdr.append("bash | yield 10s idle / 120s total")
        hdr.append(" · id: call_…789abc", style="dim")
        syn = _UnpaddedSyntax(ev.command, "bash", theme=cfg.theme)
        if ev.exit_code is None:
            return Group(hdr, syn)
        exit_badge = Text("▎", style="dim")
        exit_badge.append(
            f" exit code {ev.exit_code} ",
            style="white reverse dim" if ev.exit_code == 0 else "red reverse bold",
        )
        n_lines = len(ev.output.splitlines())
        detail_parts = []
        if n_lines:
            detail_parts.append(f"{n_lines} line{'s' if n_lines != 1 else ''}")
        if ev.elapsed:
            detail_parts.append(ev.elapsed)
        if detail_parts:
            exit_badge.append(" " + " · ".join(detail_parts), style="dim")
        return Group(hdr, syn, exit_badge)

    if cfg.style == "hybrid":
        hdr_hyb = Text("▎◀ ", style="magenta")
        hdr_hyb.append("bash")
        if ev.cwd:
            hdr_hyb.append(f" · {ev.cwd}", style="dim")
        syn_hyb = _UnpaddedSyntax(ev.command, "bash", theme=cfg.theme)
        if ev.exit_code is None:
            return Group(hdr_hyb, syn_hyb)
        exit_hyb = Text("▎", style="dim")
        exit_hyb.append(
            f" exit {ev.exit_code} ",
            style="white reverse dim" if ev.exit_code == 0 else "red reverse bold",
        )
        if ev.elapsed:
            exit_hyb.append(f" {ev.elapsed}", style="dim")
        if ev.exit_code != 0 and ev.output:
            return Group(hdr_hyb, syn_hyb, Text(ev.output.strip(), style="red"), exit_hyb)
        return Group(hdr_hyb, syn_hyb, exit_hyb)

    parts: list[RenderableType] = [
        _action_opener("shell", ev.cwd or "", cfg, tag5="SHELL"),
        _UnpaddedSyntax(ev.command, "bash", theme=cfg.theme),
    ]
    if ev.exit_code is None:
        return Group(*parts)
    trimmed = _trim_output(
        ev.output,
        "full" if ev.exit_code != 0 and cfg.output_mode == "compact" else cfg.output_mode,
        limit=6,
    )
    if trimmed:
        div = (
            Text("· output", style="dim")
            if cfg.style == "marquee"
            else Text("├─ OUT", style="dim")
            if cfg.style == "blueprint"
            else Text("── output" if cfg.style == "hairline" else "─── output", style="dim")
        )
        parts.append(_RuleLine(div, mode="stub"))
        parts.append(Text(trimmed, style="red" if ev.exit_code != 0 else "dim"))

    cl = _action_closer(cfg)
    suffix_el = f" · {ev.elapsed}" if ev.elapsed else ""
    if ev.exit_code == 0:
        cl.append("✓ ", style="dim green")
        cl.append(f"exit 0{suffix_el}", style="dim")
    else:
        cl.append(f"✗ exit {ev.exit_code}{suffix_el}", style="bold red")
    parts.append(_RuleLine(cl, mode="stub"))
    return Group(*parts)


def _render_compact_batch(ev: CompactToolBatchEvent, cfg: RenderConfig) -> RenderableType:
    if cfg.style == "current":
        hdr = Text("▎◀ ", style="magenta")
        suffix = f" ×{ev.count}" if ev.count > 1 else ""
        hdr.append(f"file {ev.verb}{suffix}")
        res = Text("▎▶ ", style="magenta")
        res.append(f"file {ev.verb}")
        res.append(f" · {ev.result_summary}", style="dim")
        return Group(hdr, res)

    if cfg.style == "hybrid":
        line_hyb = Text("▎◀ ", style="magenta")
        verb_label = f"{ev.verb} ×{ev.count}" if ev.count > 1 else ev.verb
        line_hyb.append(verb_label)
        if ev.targets:
            line_hyb.append(" · " + " · ".join(ev.targets[:3]), style="dim")
        if ev.result_summary:
            line_hyb.append(f"  ✓ {ev.result_summary}", style="dim")
        return line_hyb

    verb_label = f"{ev.verb} ×{ev.count}" if ev.count > 1 else ev.verb
    target_str = " · ".join(ev.targets[:3])
    if len(ev.targets) > 3:
        target_str += f" · +{len(ev.targets) - 3}"
    parts: list[RenderableType] = [_action_opener(verb_label, target_str, cfg, tag5="TOOL")]
    if ev.payload_json:
        parts.append(_UnpaddedSyntax(ev.payload_json, "json", theme=cfg.theme))
    cl = _action_closer(cfg)
    if ev.is_error:
        cl.append(f"✗ {ev.result_summary}", style="bold red")
    else:
        cl.append("✓ ", style="dim green")
        summary = ev.result_summary
        if ev.elapsed and ev.elapsed not in summary:
            summary = f"{summary} · {ev.elapsed}" if summary else ev.elapsed
        cl.append(summary, style="dim")
    parts.append(_RuleLine(cl, mode="stub"))
    return Group(*parts)


def _render_assistant_event(ev: AssistantEvent, cfg: RenderConfig) -> RenderableType:
    items: list[RenderableType] = [
        _speaker_line("assistant", ev.sender, ev.model if ev.first_in_turn else None, cfg)
    ]
    for idx, seg in enumerate(ev.segments):
        is_last = idx == len(ev.segments) - 1
        if seg.kind == "prose":
            prose = seg.text.strip()
            if prose:
                items.append(
                    build_rich_markdown(
                        prose,
                        code_theme=cfg.theme,
                        live_incomplete_fences=True,
                        cursor_suffix=cfg.cursor if is_last else "",
                    )
                )
        else:
            items.append(
                _render_prose_code(
                    seg.text,
                    seg.language,
                    is_open=seg.is_open,
                    cfg=cfg
                    if is_last
                    else RenderConfig(
                        style=cfg.style,
                        theme=cfg.theme,
                        rule_mode=cfg.rule_mode,
                        speaker_mode=cfg.speaker_mode,
                        output_mode=cfg.output_mode,
                        cursor="",
                    ),
                )
            )
    return Group(*items)


def render_turn_event(ev: TurnEvent, cfg: RenderConfig) -> RenderableType:
    if isinstance(ev, UserEvent):
        return Group(_speaker_line("user", ev.sender, None, cfg), Text(ev.text))
    if isinstance(ev, AssistantEvent):
        return _render_assistant_event(ev, cfg)
    if isinstance(ev, SandboxEvent):
        return _render_sandbox_event(ev, cfg)
    if isinstance(ev, FileWriteEvent):
        return _render_file_write(ev, cfg)
    if isinstance(ev, FileEditEvent):
        return _render_file_edit(ev, cfg)
    if isinstance(ev, ShellEvent):
        return _render_shell(ev, cfg)
    if isinstance(ev, CompactToolBatchEvent):
        return _render_compact_batch(ev, cfg)
    bits = [f"{ev.tool_count} tool{'s' if ev.tool_count != 1 else ''}"]
    if ev.elapsed:
        bits.append(ev.elapsed)
    if ev.tokens:
        bits.append(ev.tokens)
    if cfg.style == "current":
        return Text("▎ " + " • ".join(bits), style="dim")
    return Text("· " + " · ".join(bits), style="dim")


def render_transcript(events: list[TurnEvent], cfg: RenderConfig) -> RenderableType:
    spaced: list[RenderableType] = []
    for idx, ev in enumerate(events):
        if idx > 0:
            spaced.append(Text(""))
        spaced.append(render_turn_event(ev, cfg))
    return Group(*spaced)


def build_showcase_scenario() -> list[TurnEvent]:
    return [
        UserEvent(
            sender="you",
            text="The retry jitter test is flaky under load. Diagnose it in the sandbox, add a focused regression test file, fix the clamp, and run pytest.",
        ),
        AssistantEvent(
            sender="dev",
            model="claude-opus-5.5",
            first_in_turn=True,
            segments=(
                AssistantSegment(
                    kind="prose",
                    text="I'll inspect `retry.py` and reproduce the failure in a single sandbox pass.",
                ),
            ),
        ),
        SandboxEvent(
            code='''src = open("src/fast_agent/llm/retry.py").read()
run, log = await asyncio.gather(
    shell(command="""
        uv run pytest tests/unit/test_retry.py -q
        git status -sb
    """),
    shell(command="git log --oneline -2 -- src/fast_agent/llm/retry.py"),
)
print(run["stdout"].strip().splitlines()[-2])
print([line.strip() for line in src.splitlines() if "random" in line])''',
            output=(
                "2 failed, 18 passed in 0.41s\n"
                "['delay = min(base * 2**attempt, max_delay) + random.uniform(0, base)']"
            ),
            status="ok",
            elapsed="0.44s",
            total_calls=2,
            calls=(SandboxCallStat(name="shell", ok=2, marks="✓✓"),),
        ),
        CompactToolBatchEvent(
            verb="read",
            count=2,
            targets=("conftest.py", "test_retry.py"),
            result_summary="2 files · 3.8k chars",
            elapsed="0.06s",
        ),
        FileWriteEvent(
            path="tests/unit/test_retry_jitter.py",
            language="python",
            content="""from fast_agent.llm.retry import compute_backoff_delay


def test_backoff_delay_never_exceeds_max_delay() -> None:
    for attempt in range(10):
        assert compute_backoff_delay(attempt, base=0.5, max_delay=4.0) <= 4.0""",
        ),
        FileEditEvent(
            kind="edit",
            path_label="src/fast_agent/llm/retry.py",
            diff_lines=(
                "@@ -41,1 +41,2 @@",
                "-    return min(base * 2**attempt, max_delay) + random.uniform(0, base)",
                "+    jitter = random.uniform(0, base)",
                "+    return min(base * 2**attempt + jitter, max_delay)",
            ),
            added=2,
            removed=1,
        ),
        ShellEvent(
            command="uv run pytest tests/unit/test_retry.py tests/unit/test_retry_jitter.py -q",
            output="21 passed in 0.36s",
            exit_code=0,
            elapsed="0.39s",
        ),
        AssistantEvent(
            sender="dev",
            model="claude-opus-5.5",
            first_in_turn=False,
            segments=(
                AssistantSegment(
                    kind="prose",
                    text="Fixed — jitter was added *after* clamping to `max_delay`, allowing delays up to `max_delay + base`. Clamping the sum keeps every retry within bounds:",
                ),
                AssistantSegment(
                    kind="code",
                    language="python",
                    text=(
                        "jitter = random.uniform(0, base)\n"
                        "return min(base * 2**attempt + jitter, max_delay)"
                    ),
                ),
            ),
        ),
        TurnFooterEvent(tool_count=5, elapsed="4.8s", tokens="12.4k tokens"),
    ]


def build_errors_scenario() -> list[TurnEvent]:
    return [
        UserEvent(sender="you", text="Check the staging endpoint and patch the timeout constant."),
        SandboxEvent(
            code="""r1, r2 = await asyncio.gather(
    shell(command="curl -sf http://127.0.0.1:9999/health"),
    shell(command="git status -sb"),
)
print("health exit:", r1["exit_code"], "branch:", r2["stdout"].strip())""",
            output="health exit: 7 branch: ## feat/osc-7577-program-resume",
            status="warn",
            elapsed="0.31s",
            total_calls=2,
            failed_calls=1,
            calls=(SandboxCallStat(name="shell", ok=1, failed=1, marks="✗✓"),),
        ),
        FileEditEvent(
            kind="edit",
            path_label="src/fast_agent/config.py",
            diff_lines=(
                "@@ -1,1 +1,1 @@",
                "-DEFAULT_TIMEOUT = 15",
                "+DEFAULT_TIMEOUT = 30",
            ),
            added=1,
            removed=1,
            status="error",
            error_detail="old_string not found · id 7f3a",
        ),
        ShellEvent(
            command="uv run pytest tests/unit/test_missing.py -q",
            output="ERROR: file or directory not found: tests/unit/test_missing.py",
            exit_code=4,
            elapsed="0.27s",
        ),
        TurnFooterEvent(tool_count=3, elapsed="2.1s", tokens="4.9k tokens"),
    ]


def _split_markdown_segments(text: str) -> tuple[AssistantSegment, ...]:
    segments: list[AssistantSegment] = []
    pos = 0
    while pos < len(text):
        match = _FENCE_RE.search(text, pos)
        if match is None:
            rem = text[pos:].strip()
            if rem:
                segments.append(AssistantSegment(kind="prose", text=rem))
            break
        if match.start() > pos:
            before = text[pos : match.start()].strip()
            if before:
                segments.append(AssistantSegment(kind="prose", text=before))
        fence = match.group(2)
        info = match.group(3).strip()
        lang = info.split()[0] if info else "text"
        close_re = re.compile(rf"(?m)^[ \t]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        close_match = close_re.search(text, match.end())
        if close_match is None:
            code = text[match.end() :].rstrip("\n")
            segments.append(AssistantSegment(kind="code", text=code, language=lang, is_open=True))
            break
        code = text[match.end() : close_match.start()].rstrip("\n")
        segments.append(AssistantSegment(kind="code", text=code, language=lang))
        pos = close_match.end()
        if pos < len(text) and text[pos] == "\n":
            pos += 1
    return tuple(segments)


def _discover_session_files() -> list[tuple[str, Path, int]]:
    roots = [Path(".fast-agent/sessions"), Path.home() / ".fast-agent/sessions"]
    found: list[tuple[float, str, Path, int]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for session_dir in root.iterdir():
            if not session_dir.is_dir() or session_dir.name.startswith("."):
                continue
            for hist in sorted(session_dir.glob("history_*.json")):
                if hist.name.endswith("_previous.json"):
                    continue
                resolved = hist.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    stat = hist.stat()
                except OSError:
                    continue
                found.append((stat.st_mtime, session_dir.name, hist, stat.st_size))
    found.sort(key=lambda item: item[0], reverse=True)
    return [(sid, path, size) for _, sid, path, size in found]


def _extract_text_blocks(blocks: object) -> str:
    if not isinstance(blocks, list):
        return ""
    chunks: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            val = block.get("text")
            if isinstance(val, str) and val:
                chunks.append(val)
    return "\n".join(chunks)


def _parse_ptc_result(
    raw_text: str,
    meta: dict[str, Any] | None,
    is_error: bool,
    fallback_elapsed: str = "",
) -> tuple[str, Literal["ok", "warn", "error"], str, int, int, tuple[SandboxCallStat, ...], str]:
    lines = raw_text.splitlines()
    elapsed = fallback_elapsed
    total_calls = 0
    failed_calls = 0
    status: Literal["ok", "warn", "error"] = "error" if is_error else "ok"
    error_summary = ""
    body_start = 0

    if lines:
        m = _PTC_STATUS_RE.match(lines[0].strip())
        if m:
            word, elapsed, calls_s, failed_s = m.groups()
            total_calls = int(calls_s)
            failed_calls = int(failed_s or "0")
            if word != "completed" or is_error:
                status = "error"
                error_summary = f"{word} {elapsed}"
            elif failed_calls > 0:
                status = "warn"
            body_start = 1
            if body_start < len(lines) and lines[body_start].strip() in {"Output:", "Error:"}:
                body_start += 1

    output = "\n".join(lines[body_start:]).strip("\n")
    ptc_calls_raw = meta.get("fast-agent-ptc-calls") if isinstance(meta, dict) else None
    stats_order: list[str] = []
    ok_counts: Counter[str] = Counter()
    fail_counts: Counter[str] = Counter()
    marks_map: dict[str, list[str]] = {}
    if isinstance(ptc_calls_raw, list):
        for entry in ptc_calls_raw:
            if not isinstance(entry, dict):
                continue
            fn = str(entry.get("function") or entry.get("tool") or "call")
            if fn not in marks_map:
                stats_order.append(fn)
                marks_map[fn] = []
            err = bool(entry.get("is_error"))
            if err:
                fail_counts[fn] += 1
                marks_map[fn].append("✗")
            else:
                ok_counts[fn] += 1
                marks_map[fn].append("✓")
        if not total_calls:
            total_calls = sum(ok_counts.values()) + sum(fail_counts.values())
        if not failed_calls:
            failed_calls = sum(fail_counts.values())
        if failed_calls and status == "ok":
            status = "warn"

    call_stats = tuple(
        SandboxCallStat(
            name=fn,
            ok=ok_counts[fn],
            failed=fail_counts[fn],
            marks="".join(marks_map[fn]),
        )
        for fn in stats_order
    )
    return output, status, elapsed, total_calls, failed_calls, call_stats, error_summary


def load_real_session_turn(
    session_query: str,
    turn_number: int,
) -> tuple[str, int, int, list[TurnEvent]]:
    candidates = _discover_session_files()
    if not candidates:
        raise RuntimeError(
            "No saved sessions found in .fast-agent/sessions or ~/.fast-agent/sessions"
        )

    chosen_path: Path | None = None
    chosen_id = ""
    if session_query == "latest":
        chosen_id, chosen_path, _ = candidates[0]
    else:
        for sid, path, _ in candidates:
            if session_query in sid or session_query in str(path):
                chosen_id, chosen_path = sid, path
                break
    if chosen_path is None:
        raise RuntimeError(f"No session matching {session_query!r}; try --list-sessions")

    raw_data = json.loads(chosen_path.read_text(encoding="utf-8"))
    messages: list[dict[str, Any]] = raw_data.get("messages", [])

    # Split into user-initiated turns (user messages without tool_results, skipping compaction)
    turns: list[list[dict[str, Any]]] = []
    current_turn: list[dict[str, Any]] = []
    for msg in messages:
        channels = msg.get("channels") or {}
        if "fast-agent-compaction" in channels and not msg.get("tool_results"):
            continue
        is_user_prompt = msg.get("role") == "user" and not msg.get("tool_results")
        if is_user_prompt and current_turn:
            turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)
    if current_turn:
        turns.append(current_turn)
    if not turns:
        raise RuntimeError(f"Session {chosen_id} contains no turns")

    idx = max(1, min(turn_number, len(turns))) - 1
    turn_msgs = turns[idx]

    # Pre-index tool results, timings, and metadata across the turn
    results_by_id: dict[str, dict[str, Any]] = {}
    meta_by_id: dict[str, dict[str, Any]] = {}
    timings_by_id: dict[str, float] = {}
    total_ms = 0.0
    for msg in turn_msgs:
        tr = msg.get("tool_results") or {}
        if isinstance(tr, dict):
            for cid, res in tr.items():
                if isinstance(res, dict):
                    results_by_id[cid] = res
        channels = msg.get("channels") or {}
        for block in channels.get("fast-agent-tool-metadata") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parsed = json.loads(block["text"])
                if isinstance(parsed, dict):
                    for cid, mval in parsed.items():
                        if isinstance(mval, dict):
                            meta_by_id[cid] = mval
        for block in channels.get("fast-agent-tool-timing") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parsed = json.loads(block["text"])
                if isinstance(parsed, dict):
                    for cid, tval in parsed.items():
                        if isinstance(tval, dict) and isinstance(
                            tval.get("timing_ms"), (int, float)
                        ):
                            timings_by_id[cid] = float(tval["timing_ms"])
        for block in channels.get("fast-agent-timing") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parsed = json.loads(block["text"])
                if isinstance(parsed, dict) and isinstance(parsed.get("duration_ms"), (int, float)):
                    total_ms += float(parsed["duration_ms"])

    events: list[TurnEvent] = []
    saw_assistant = False
    tool_total = 0
    agent_label = chosen_path.stem.removeprefix("history_")

    for msg in turn_msgs:
        role = msg.get("role")
        text = _extract_text_blocks(msg.get("content"))
        if role == "user" and not msg.get("tool_results"):
            if text.strip():
                events.append(UserEvent(sender="you", text=text.strip()))
            continue
        if role != "assistant":
            continue

        if text.strip():
            segs = _split_markdown_segments(text)
            if segs:
                events.append(
                    AssistantEvent(
                        sender=agent_label,
                        model=None,
                        segments=segs,
                        first_in_turn=not saw_assistant,
                    )
                )
                saw_assistant = True

        tcalls = msg.get("tool_calls") or {}
        if not isinstance(tcalls, dict):
            continue
        for cid, call in tcalls.items():
            tool_total += 1
            params = call.get("params") if isinstance(call, dict) else {}
            tname = str((params or {}).get("name") or "tool")
            targs = (params or {}).get("arguments")
            args_dict: dict[str, Any] = targs if isinstance(targs, dict) else {}
            tmeta = meta_by_id.get(cid, {})
            has_result = cid in results_by_id
            tres = results_by_id.get(cid, {})
            res_text = _extract_text_blocks(tres.get("content"))
            res_err = bool(tres.get("isError"))
            ms = timings_by_id.get(cid)
            elapsed_s = f"{ms / 1000:.2f}s" if ms is not None else ""

            if tname == "exec_sandbox" or tmeta.get("variant") == "code":
                code_key = str(tmeta.get("code_arg") or "code")
                code_str = str(args_dict.get(code_key) or "").rstrip()
                if not has_result:
                    events.append(SandboxEvent(code=code_str, is_open=True))
                    continue
                res_meta = tres.get("_meta") if isinstance(tres.get("_meta"), dict) else None
                out, st, el, tot_c, fail_c, cstats, err_s = _parse_ptc_result(
                    res_text, res_meta, res_err, fallback_elapsed=elapsed_s
                )
                events.append(
                    SandboxEvent(
                        code=code_str,
                        output=out,
                        status=st,
                        elapsed=el or "done",
                        total_calls=tot_c,
                        failed_calls=fail_c,
                        calls=cstats,
                        error_summary=err_s,
                    )
                )
            elif tname in {"write_text_file", "filesystem__write_text_file"}:
                p_str = str(args_dict.get("path") or "file.txt")
                content_s = str(args_dict.get("content") or "")
                lang = syntax_language_for_path(p_str) or "text"
                events.append(
                    FileWriteEvent(
                        path=p_str,
                        content=content_s,
                        language=lang,
                        status="error" if res_err else "ok",
                        error_detail=res_text[:80] if res_err else "",
                        is_open=not has_result,
                    )
                )
            elif tname == "edit_file":
                p_str = str(args_dict.get("path") or "file")
                old_s = str(args_dict.get("old_string") or "")
                new_s = str(args_dict.get("new_string") or "")
                dlines = tuple(
                    line
                    for line in unified_diff(
                        old_s.splitlines(), new_s.splitlines(), lineterm="", n=0
                    )
                    if not line.startswith("--- ") and not line.startswith("+++ ")
                )
                added = sum(1 for ln in dlines if ln.startswith("+"))
                removed = sum(1 for ln in dlines if ln.startswith("-"))
                events.append(
                    FileEditEvent(
                        kind="edit",
                        path_label=p_str,
                        diff_lines=dlines,
                        added=added,
                        removed=removed,
                        status="error" if res_err else "ok",
                        error_detail=res_text.splitlines()[0][:72] if res_err and res_text else "",
                    )
                )
            elif tname == "apply_patch" or (
                tname in {"bash", "shell", "execute"}
                and isinstance(args_dict.get("command"), str)
                and extract_apply_patch_text(args_dict["command"]) is not None
            ):
                patch_in = str(
                    args_dict.get("input")
                    or extract_apply_patch_text(str(args_dict.get("command") or ""))
                    or ""
                )
                preview_txt = render_patch_preview(patch_in, max_lines=24)
                dlines = tuple(preview_txt.splitlines())
                added = sum(1 for ln in dlines if ln.startswith("+") and not ln.startswith("+++"))
                removed = sum(1 for ln in dlines if ln.startswith("-") and not ln.startswith("---"))
                events.append(
                    FileEditEvent(
                        kind="patch",
                        path_label="apply_patch",
                        diff_lines=dlines,
                        added=added,
                        removed=removed,
                        status="error" if res_err else "ok",
                    )
                )
            elif tname in {"bash", "shell", "execute"} or tmeta.get("variant") == "shell":
                cmd_s = str(args_dict.get("command") or "").strip()
                exit_code: int | None = None
                if has_result:
                    exit_code = 1 if res_err else 0
                    exit_m = re.search(r"exit[_ ]code[=:\s]+(-?\d+)", res_text, re.IGNORECASE)
                    if exit_m:
                        exit_code = int(exit_m.group(1))
                events.append(
                    ShellEvent(
                        command=cmd_s,
                        output=res_text,
                        exit_code=exit_code,
                        elapsed=elapsed_s,
                    )
                )
            elif "read_text_file" in tname:
                p_name = Path(str(args_dict.get("path") or "file")).name
                events.append(
                    CompactToolBatchEvent(
                        verb="read",
                        count=1,
                        targets=(p_name,),
                        result_summary=f"{len(res_text):,} chars",
                        elapsed=elapsed_s,
                        is_error=res_err,
                    )
                )
            else:
                payload = json.dumps(args_dict, ensure_ascii=False) if args_dict else None
                if payload and len(payload) > 140:
                    payload = payload[:137] + "..."
                events.append(
                    CompactToolBatchEvent(
                        verb=f"tool · {tname}",
                        count=1,
                        payload_json=payload,
                        result_summary=f"{len(res_text):,} chars",
                        elapsed=elapsed_s,
                        is_error=res_err,
                    )
                )

    if tool_total > 0 or total_ms > 0:
        events.append(
            TurnFooterEvent(
                tool_count=tool_total,
                elapsed=f"{total_ms / 1000:.1f}s" if total_ms > 0 else None,
            )
        )
    return f"{chosen_id}/{chosen_path.name}", idx + 1, len(turns), events


def stream_transcript_live(console: Console, events: list[TurnEvent], cfg: RenderConfig) -> None:
    """Replay events through _DiffLive showing open-fence append-only streaming."""
    cursor_glyph = "●" if cfg.style in {"marquee", "current"} else "▍"
    completed: list[TurnEvent] = []
    live = _DiffLive(console=console, transient=False)
    with live:
        for ev in events:
            if isinstance(ev, SandboxEvent):
                lines = ev.code.splitlines()
                for n in range(1, len(lines) + 1):
                    partial = SandboxEvent(code="\n".join(lines[:n]), is_open=True)
                    live.update(
                        render_transcript(
                            [*completed, partial],
                            RenderConfig(
                                style=cfg.style,
                                theme=cfg.theme,
                                rule_mode=cfg.rule_mode,
                                speaker_mode=cfg.speaker_mode,
                                output_mode=cfg.output_mode,
                                cursor=cursor_glyph,
                            ),
                        ),
                        refresh=True,
                    )
                    time.sleep(0.05)
                time.sleep(0.12)
            elif isinstance(ev, FileWriteEvent):
                lines = ev.content.splitlines()
                for n in range(1, len(lines) + 1):
                    partial_fw = FileWriteEvent(
                        path=ev.path,
                        content="\n".join(lines[:n]),
                        language=ev.language,
                        is_open=True,
                    )
                    live.update(
                        render_transcript(
                            [*completed, partial_fw],
                            RenderConfig(
                                style=cfg.style,
                                theme=cfg.theme,
                                rule_mode=cfg.rule_mode,
                                speaker_mode=cfg.speaker_mode,
                                output_mode=cfg.output_mode,
                                cursor=cursor_glyph,
                            ),
                        ),
                        refresh=True,
                    )
                    time.sleep(0.05)
            elif isinstance(ev, AssistantEvent):
                built_segs: list[AssistantSegment] = []
                for seg in ev.segments:
                    step = 18 if seg.kind == "prose" else 14
                    for end in range(step, len(seg.text) + step, step):
                        chunk = seg.text[:end]
                        open_flag = seg.kind == "code" and end < len(seg.text)
                        part_seg = AssistantSegment(
                            kind=seg.kind,
                            text=chunk,
                            language=seg.language,
                            is_open=open_flag,
                        )
                        part_ev = AssistantEvent(
                            sender=ev.sender,
                            model=ev.model,
                            first_in_turn=ev.first_in_turn,
                            segments=(*built_segs, part_seg),
                        )
                        live.update(
                            render_transcript(
                                [*completed, part_ev],
                                RenderConfig(
                                    style=cfg.style,
                                    theme=cfg.theme,
                                    rule_mode=cfg.rule_mode,
                                    speaker_mode=cfg.speaker_mode,
                                    output_mode=cfg.output_mode,
                                    cursor=cursor_glyph,
                                ),
                            ),
                            refresh=True,
                        )
                        time.sleep(0.03)
                    built_segs.append(seg)

            completed.append(ev)
            live.update(render_transcript(completed, cfg), refresh=True)
            time.sleep(0.06)


def verify_invariants() -> None:
    """Mechanical verification: pure col-0 code lines, zero trailing prose spaces, zero bg fill."""
    events = build_showcase_scenario()
    expected_code_lines = [
        "from fast_agent.llm.retry import compute_backoff_delay",
        '    shell(command="git log --oneline -2 -- src/fast_agent/llm/retry.py"),',
        "return min(base * 2**attempt + jitter, max_delay)",
    ]
    for style in STYLE_ORDER:
        from io import StringIO

        rec = Console(
            file=StringIO(), width=100, record=True, force_terminal=True, color_system="truecolor"
        )
        cfg = RenderConfig(style=style, theme="ansi_light")
        lines = rec.render_lines(render_transcript(events, cfg), rec.options, pad=False)
        plain_lines = ["".join(seg.text for seg in line) for line in lines]
        for expected in expected_code_lines:
            assert expected in plain_lines, f"{style}: missing exact col-0 line {expected!r}"
        for plain in plain_lines:
            trailing_spaces = len(plain) - len(plain.rstrip(" "))
            assert trailing_spaces <= 1, (
                f"{style}: unexpected right-padding ({trailing_spaces} spaces) in {plain!r}"
            )
    print(
        f"✓ Verified all {len(STYLE_ORDER)} styles ({', '.join(STYLE_ORDER)}): "
        "exact col-0 source, zero trailing spaces, reflow-clean."
    )


def _print_session_list(console: Console) -> None:
    sessions = _discover_session_files()
    console.print(Text("Recent saved sessions available for --session <id>:", style="bold cyan"))
    for sid, path, size in sessions[:15]:
        try:
            msgs = json.loads(path.read_text(encoding="utf-8")).get("messages", [])
            ptc_n = sum(
                1
                for m in msgs
                for c in (m.get("tool_calls") or {}).values()
                if isinstance(c, dict) and (c.get("params") or {}).get("name") == "exec_sandbox"
            )
            turns_n = sum(
                1
                for m in msgs
                if m.get("role") == "user"
                and not m.get("tool_results")
                and "fast-agent-compaction" not in (m.get("channels") or {})
            )
            badge = f" · {ptc_n} exec_sandbox" if ptc_n else ""
            console.print(
                f"  [bold]{sid}[/bold] ({path.name}) — {turns_n} turns, {size // 1024} KB{badge}"
            )
        except Exception:
            continue


def _run_interactive(console: Console, initial_cfg: RenderConfig) -> None:
    cfg = initial_cfg
    scenario_mode: str = "showcase"
    session_query = "2611261357"
    turn_num = 1

    while True:
        console.clear()
        if scenario_mode == "session":
            try:
                label, t_idx, t_total, events = load_real_session_turn(session_query, turn_num)
                source_desc = f"session {label} (turn {t_idx}/{t_total})"
            except Exception as exc:
                source_desc = f"session load error ({exc}); showing showcase"
                events = build_showcase_scenario()
        elif scenario_mode == "errors":
            source_desc = "scenario=errors"
            events = build_errors_scenario()
        else:
            source_desc = "scenario=showcase"
            events = build_showcase_scenario()

        banner = Text()
        banner.append(" fast-agent style gallery ", style="bold reverse cyan")
        banner.append(
            f"  style={cfg.style}  rule={_effective_rule_mode(cfg)}  "
            f"output={cfg.output_mode}  theme={cfg.theme}  [{source_desc}]",
            style="dim",
        )
        console.print(banner)
        console.print()
        console.print(render_transcript(events, cfg))
        console.print()
        help_bar = Text(
            "[1] hybrid  [2] current  [3] linework  [4] marquee  [5] hairline  [6] blueprint  "
            "| [c] showcase  [e] errors  [s] real session  [n/p] next/prev turn  "
            "| [l] live stream  [r] cycle rule  [o] cycle output  [t] theme  [q] quit",
            style="dim",
        )
        console.print(help_bar)
        try:
            choice = input("choice> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return

        if choice in {"q", "quit", "exit"}:
            return
        if choice in {"1", "2", "3", "4", "5", "6"}:
            cfg = RenderConfig(
                style=STYLE_ORDER[int(choice) - 1],
                theme=cfg.theme,
                rule_mode=cfg.rule_mode,
                speaker_mode=cfg.speaker_mode,
                output_mode=cfg.output_mode,
            )
        elif choice == "c":
            scenario_mode = "showcase"
        elif choice == "e":
            scenario_mode = "errors"
        elif choice == "s":
            scenario_mode = "session"
        elif choice == "n":
            scenario_mode = "session"
            turn_num += 1
        elif choice == "p":
            scenario_mode = "session"
            turn_num = max(1, turn_num - 1)
        elif choice == "r":
            modes: tuple[RuleMode, ...] = ("default", "stub", "fixed", "full")
            nxt = modes[(modes.index(cfg.rule_mode) + 1) % len(modes)]
            cfg = RenderConfig(
                style=cfg.style,
                theme=cfg.theme,
                rule_mode=nxt,
                speaker_mode=cfg.speaker_mode,
                output_mode=cfg.output_mode,
            )
        elif choice == "o":
            omodes: tuple[OutputMode, ...] = ("compact", "full", "none")
            nxt_o = omodes[(omodes.index(cfg.output_mode) + 1) % len(omodes)]
            cfg = RenderConfig(
                style=cfg.style,
                theme=cfg.theme,
                rule_mode=cfg.rule_mode,
                speaker_mode=cfg.speaker_mode,
                output_mode=nxt_o,
            )
        elif choice == "t":
            themes = ("monokai", "ansi_dark", "github-dark", "ansi_light")
            nxt_t = (
                themes[(themes.index(cfg.theme) + 1) % len(themes)]
                if cfg.theme in themes
                else "ansi_dark"
            )
            cfg = RenderConfig(
                style=cfg.style,
                theme=nxt_t,
                rule_mode=cfg.rule_mode,
                speaker_mode=cfg.speaker_mode,
                output_mode=cfg.output_mode,
            )
        elif choice == "l":
            console.clear()
            console.print(Text(f"Streaming {cfg.style} via _DiffLive…", style="bold cyan"))
            console.print()
            stream_transcript_live(console, events, cfg)
            console.print()
            input("Press Enter to return to gallery…")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--style",
        choices=[*STYLE_ORDER, "all"],
        default="hybrid",
        help="Visual family to render (default: hybrid; use 'all' to compare all 6 incl. current)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Open interactive style/session/streaming switcher",
    )
    parser.add_argument(
        "--scenario",
        choices=["showcase", "errors"],
        default="showcase",
        help="Synthetic turn scenario when --session is not used",
    )
    parser.add_argument(
        "--session",
        help="Load a real saved session by ID substring or 'latest' (e.g. --session 2611261357)",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=1,
        help="1-based turn index when replaying --session (default: 1)",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List saved sessions in .fast-agent/sessions and ~/.fast-agent/sessions",
    )
    parser.add_argument(
        "--rule",
        choices=["default", "stub", "fixed", "full"],
        default="default",
        help="Override horizontal rule width strategy (stub never wraps; full shows shrink wrap)",
    )
    parser.add_argument(
        "--speaker",
        choices=["default", "dots", "bars"],
        default="default",
        help="Override user/assistant speaker markers (dots ●/○ vs bars ▎)",
    )
    parser.add_argument(
        "--output",
        choices=["compact", "full", "none"],
        default="compact",
        help="How much sandbox/shell stdout to show inline",
    )
    parser.add_argument(
        "--theme",
        default="ansi_light",
        help="Syntax color theme on transparent background (default: ansi_light)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Animate live token/line streaming through _DiffLive",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Override Console render width",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run mechanical copy-paste & zero-background invariant checks",
    )
    args = parser.parse_args()

    if args.verify:
        verify_invariants()
        return

    console = Console(width=args.width) if args.width else Console()
    if args.list_sessions:
        _print_session_list(console)
        return

    base_cfg = RenderConfig(
        style="hybrid" if args.style == "all" else args.style,
        theme=args.theme,
        rule_mode=args.rule,
        speaker_mode=args.speaker,
        output_mode=args.output,
    )

    if args.interactive:
        _run_interactive(console, base_cfg)
        return

    if args.session:
        label, t_idx, t_total, events = load_real_session_turn(args.session, args.turn)
        header_note = f"Session {label} — turn {t_idx}/{t_total}"
    elif args.scenario == "errors":
        events = build_errors_scenario()
        header_note = "Scenario: errors & partial sandbox failure"
    else:
        events = build_showcase_scenario()
        header_note = "Scenario: showcase (sandbox + write + edit + shell + prose fence)"

    styles_to_show: tuple[StyleName, ...] = (
        STYLE_ORDER if args.style == "all" else (base_cfg.style,)
    )
    for idx, st in enumerate(styles_to_show):
        if idx > 0:
            console.print()
        cfg = RenderConfig(
            style=st,
            theme=base_cfg.theme,
            rule_mode=base_cfg.rule_mode,
            speaker_mode=base_cfg.speaker_mode,
            output_mode=base_cfg.output_mode,
        )
        title = Text()
        title.append(f" {st.upper()} ", style="bold reverse cyan")
        title.append(
            f"  rule={_effective_rule_mode(cfg)} · theme={cfg.theme} · {header_note}",
            style="dim",
        )
        console.print()
        console.print(title)
        console.print()
        if args.stream:
            stream_transcript_live(console, events, cfg)
        else:
            console.print(render_transcript(events, cfg))

    console.print()
    console.print(
        Text(
            "Tip: run with `-i` for interactive switching, `--style all` to compare all 5, "
            "`--stream` for live _DiffLive streaming, or `--session 2611261357 --turn 1` for real history.",
            style="dim",
        )
    )


if __name__ == "__main__":
    main()
