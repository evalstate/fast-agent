#!/usr/bin/env python3
"""Render the Terminal-Bench Sol repricing social animation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image

DOCS_DIR = Path(__file__).resolve().parent
STORYBOARD_PATH = DOCS_DIR / "social_cards" / "benchmark" / "storyboard.html"
OUTPUT_DIR = DOCS_DIR / "social_cards" / "benchmark" / "renders"
DATA_PATH = DOCS_DIR / "docs" / "javascripts" / "homepage-benchmark-data.js"
STORYBOARD_SCRIPT_PATH = DOCS_DIR / "social_cards" / "benchmark" / "storyboard.js"
STYLES_PATH = DOCS_DIR / "social_cards" / "benchmark" / "styles.css"


@dataclass(frozen=True, slots=True)
class AnimationSpec:
    width: int = 1200
    height: int = 900
    fps: int = 30
    pulse_seconds: float = 0.6
    motion_seconds: float = 3.5
    hold_seconds: float = 4.0
    scale: int = 2
    preview_fps: int = 10

    @property
    def motion_frames(self) -> int:
        return round(self.motion_seconds * self.fps)

    @property
    def pulse_frames(self) -> int:
        return round(self.pulse_seconds * self.fps)

    @property
    def hold_frames(self) -> int:
        return round(self.hold_seconds * self.fps)

    @property
    def motion_end_frame(self) -> int:
        return self.pulse_frames + self.motion_frames

    @property
    def frame_count(self) -> int:
        return self.motion_end_frame + self.hold_frames

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True, slots=True)
class WebPMetadata:
    nominal_fps: int
    encoded_frames: int
    duration_ms: int
    final_frame_duration_ms: int


def executable_path(name: str) -> str | None:
    return shutil.which(name)


def chrome_path() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = executable_path(name)
        if path:
            return path
    return None


def ffmpeg_path(explicit: str | None) -> str | None:
    candidate = explicit or os.environ.get("FAST_AGENT_FFMPEG")
    if candidate:
        path = Path(candidate).expanduser()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return executable_path("ffmpeg")


def command_version(command: str, argument: str) -> str:
    result = subprocess.run(
        [command, argument],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = result.stdout or result.stderr
    return output.splitlines()[0] if output else "unknown"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def webp_metadata(path: Path, nominal_fps: int) -> WebPMetadata:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise RuntimeError("Animated WebP output has an invalid container")
    offset = 12
    durations: list[int] = []
    while offset + 8 <= len(data):
        chunk_type = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        payload = data[offset + 8 : offset + 8 + chunk_size]
        if chunk_type == b"ANMF":
            durations.append(int.from_bytes(payload[12:15], "little"))
        offset += 8 + chunk_size + chunk_size % 2
    if not durations:
        raise RuntimeError("Animated WebP output contains no frames")
    return WebPMetadata(
        nominal_fps=nominal_fps,
        encoded_frames=len(durations),
        duration_ms=sum(durations),
        final_frame_duration_ms=durations[-1],
    )


def frame_url(frame: int, spec: AnimationSpec) -> str:
    query = urlencode(
        {
            "variant": "sol-price-animation",
            "frame": frame,
            "fps": spec.fps,
        }
    )
    return f"{STORYBOARD_PATH.resolve().as_uri()}?{query}"


def render_frame(
    chrome: str,
    frame: int,
    spec: AnimationSpec,
    output: Path,
    capture: Path,
) -> None:
    result = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1200",
            f"--window-size={spec.width},{spec.height}",
            f"--force-device-scale-factor={spec.scale}",
            f"--screenshot={capture}",
            frame_url(frame, spec),
        ],
        cwd=DOCS_DIR,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Chrome failed while rendering frame {frame}")
    expected_capture_size = (spec.width * spec.scale, spec.height * spec.scale)
    with Image.open(capture) as image:
        if image.size != expected_capture_size:
            raise RuntimeError(
                f"Unexpected frame {frame} capture size: {image.size}, "
                f"expected {expected_capture_size}"
            )
        image.resize(
            (spec.width, spec.height),
            Image.Resampling.LANCZOS,
        ).convert("RGB").save(output)


def render_frames(chrome: str, spec: AnimationSpec, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = frames_dir / "capture@2x.png"
    frames: list[Path] = []
    for frame in range(spec.frame_count):
        output = frames_dir / f"terminal-bench-sol-price-{frame:03d}.png"
        if frame > spec.motion_end_frame:
            shutil.copy2(frames[spec.motion_end_frame], output)
        else:
            render_frame(chrome, frame, spec, output, capture)
        frames.append(output)
        if frame % spec.fps == 0 or frame == spec.frame_count - 1:
            print(f"Rendered frame {frame + 1}/{spec.frame_count}", flush=True)
    capture.unlink(missing_ok=True)
    return frames


def existing_frames(spec: AnimationSpec, frames_dir: Path) -> list[Path]:
    frames = [
        frames_dir / f"terminal-bench-sol-price-{frame:03d}.png"
        for frame in range(spec.frame_count)
    ]
    missing = [frame for frame in frames if not frame.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} retained animation frames")
    return frames


def write_contact_sheet(frames: list[Path], spec: AnimationSpec, output: Path) -> list[int]:
    keyframes = [
        0,
        spec.pulse_frames,
        spec.pulse_frames + spec.motion_frames // 2,
        spec.motion_end_frame,
    ]
    sheet = Image.new("RGB", (spec.width, spec.height), "#090a0d")
    cell_size = (spec.width // 2, spec.height // 2)
    for index, frame in enumerate(keyframes):
        with Image.open(frames[frame]) as image:
            thumbnail = image.resize(cell_size, Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, ((index % 2) * cell_size[0], (index // 2) * cell_size[1]))
    sheet.save(output, optimize=True)
    return keyframes


def write_mp4(
    ffmpeg: str,
    frames_dir: Path,
    spec: AnimationSpec,
    output: Path,
) -> None:
    pattern = frames_dir / "terminal-bench-sol-price-%03d.png"
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(spec.fps),
            "-start_number",
            "0",
            "-i",
            str(pattern),
            "-frames:v",
            str(spec.frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed to encode the animation")


def write_animated_webp(
    ffmpeg: str,
    frames_dir: Path,
    spec: AnimationSpec,
    output: Path,
) -> None:
    pattern = frames_dir / "terminal-bench-sol-price-%03d.png"
    preview_frames = round(spec.duration_seconds * spec.preview_fps)
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(spec.fps),
            "-start_number",
            "0",
            "-i",
            str(pattern),
            "-vf",
            f"fps={spec.preview_fps}",
            "-frames:v",
            str(preview_frames),
            "-loop",
            "0",
            "-an",
            "-c:v",
            "libwebp_anim",
            "-quality",
            "85",
            "-compression_level",
            "4",
            str(output),
        ],
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed to encode the animated WebP")


def write_manifest(
    spec: AnimationSpec,
    chrome: str,
    ffmpeg: str,
    outputs: list[Path],
    keyframes: list[int],
    preview: WebPMetadata,
    output: Path,
) -> None:
    manifest = {
        "animation": "sol-price-animation",
        "spec": asdict(spec),
        "pulse_frames": spec.pulse_frames,
        "motion_frames": spec.motion_frames,
        "hold_frames": spec.hold_frames,
        "motion_end_frame": spec.motion_end_frame,
        "frame_count": spec.frame_count,
        "duration_seconds": spec.duration_seconds,
        "keyframes": keyframes,
        "webp_preview": asdict(preview),
        "versions": {
            "chrome": command_version(chrome, "--version"),
            "ffmpeg": command_version(ffmpeg, "-version"),
        },
        "sources": {
            str(path.relative_to(DOCS_DIR)): file_hash(path)
            for path in (DATA_PATH, STORYBOARD_SCRIPT_PATH, STYLES_PATH)
        },
        "outputs": {path.name: file_hash(path) for path in outputs},
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", help="path to ffmpeg; defaults to FAST_AGENT_FFMPEG or PATH")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="retain the rendered PNG frame sequence under the output directory",
    )
    parser.add_argument(
        "--reuse-frames",
        action="store_true",
        help="skip Chrome rendering and encode the retained PNG frame sequence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chrome = chrome_path()
    if not chrome:
        print("google-chrome/chromium is required", file=sys.stderr)
        return 1
    ffmpeg = ffmpeg_path(args.ffmpeg)
    if not ffmpeg:
        print(
            "ffmpeg is required; use --ffmpeg, FAST_AGENT_FFMPEG, or add it to PATH",
            file=sys.stderr,
        )
        return 1

    spec = AnimationSpec()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_output = OUTPUT_DIR / "terminal-bench-sol-price-animation-frames"
    temporary_frames: tempfile.TemporaryDirectory[str] | None = None
    if args.reuse_frames:
        frames_dir = frames_output
    elif args.keep_frames:
        if frames_output.exists():
            shutil.rmtree(frames_output)
        frames_dir = frames_output
    else:
        temporary_frames = tempfile.TemporaryDirectory(prefix="fast-agent-benchmark-animation-")
        frames_dir = Path(temporary_frames.name)

    try:
        frames = (
            existing_frames(spec, frames_dir)
            if args.reuse_frames
            else render_frames(chrome, spec, frames_dir)
        )
        poster = OUTPUT_DIR / "terminal-bench-sol-price-animation-poster.png"
        webp = OUTPUT_DIR / "terminal-bench-sol-price-animation.webp"
        mp4 = OUTPUT_DIR / "terminal-bench-sol-price-animation.mp4"
        contact_sheet = OUTPUT_DIR / "terminal-bench-sol-price-animation-contact-sheet.png"
        manifest = OUTPUT_DIR / "terminal-bench-sol-price-animation.json"

        shutil.copy2(frames[-1], poster)
        keyframes = write_contact_sheet(frames, spec, contact_sheet)
        write_mp4(ffmpeg, frames_dir, spec, mp4)
        write_animated_webp(ffmpeg, frames_dir, spec, webp)
        preview = webp_metadata(webp, spec.preview_fps)
        if preview.duration_ms != round(spec.duration_seconds * 1000):
            raise RuntimeError("Animated WebP duration does not match the animation spec")
        if preview.final_frame_duration_ms != round(spec.hold_seconds * 1000):
            raise RuntimeError("Animated WebP does not preserve the final hold")
        write_manifest(
            spec,
            chrome,
            ffmpeg,
            [mp4, webp, poster, contact_sheet],
            keyframes,
            preview,
            manifest,
        )
    finally:
        if temporary_frames:
            temporary_frames.cleanup()

    print(f"Generated {mp4.relative_to(DOCS_DIR)}")
    print(f"Generated {webp.relative_to(DOCS_DIR)}")
    print(f"Generated {poster.relative_to(DOCS_DIR)}")
    print(f"Generated {contact_sheet.relative_to(DOCS_DIR)}")
    print(f"Generated {manifest.relative_to(DOCS_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
