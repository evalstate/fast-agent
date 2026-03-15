from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm_traces"
RAW_ROOT = TRACE_FIXTURES_DIR / "raw"
SANITIZED_ROOT = TRACE_FIXTURES_DIR / "sanitized"
FIXTURE_FORMAT_VERSION = 1

_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("resp", re.compile(r"\bresp_[A-Za-z0-9_-]+\b")),
    ("msg", re.compile(r"\bmsg_[A-Za-z0-9_-]+\b")),
    ("call", re.compile(r"\bcall_[A-Za-z0-9_-]+\b")),
    ("fc", re.compile(r"\bfc_[A-Za-z0-9_-]+\b")),
    ("rs", re.compile(r"\brs_[A-Za-z0-9_-]+\b")),
    ("srvtoolu", re.compile(r"\bsrvtoolu_[A-Za-z0-9_-]+\b")),
    ("toolu", re.compile(r"\btoolu_[A-Za-z0-9_-]+\b")),
    ("srv", re.compile(r"\bsrv_[A-Za-z0-9_-]+\b")),
    ("ws", re.compile(r"\bws_[A-Za-z0-9_-]+\b")),
    ("container", re.compile(r"\bcontainer_[A-Za-z0-9_-]+\b")),
)
_TIMESTAMP_PATTERN = re.compile(r"\b\d{8}T\d{6}Z\b")
_RFC1123_TIMESTAMP_PATTERN = re.compile(
    r"\b[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} GMT\b"
)

_NUMERIC_TIME_KEYS = {
    "created_at",
    "completed_at",
    "timestamp",
    "start_time",
    "end_time",
    "duration_ms",
    "ttft_ms",
    "time_to_response_ms",
}
_STRING_TIME_KEYS = {
    "started_at",
    "completed_at",
}
_OPAQUE_ID_KEYS = {
    "response_id",
}

_FAMILY_BY_PROVIDER = {
    "anthropic": "anthropic",
    "google": "google",
    "hf": "openai-chat",
    "openai": "openai-chat",
    "openresponses": "openresponses",
    "responses": "responses",
}


@dataclass(slots=True)
class TraceNormalizer:
    repo_root: Path
    counters: dict[str, int] = field(default_factory=dict)
    replacements: dict[str, str] = field(default_factory=dict)

    def normalize(self, value: Any, *, key: str | None = None) -> Any:
        if key in _NUMERIC_TIME_KEYS and isinstance(value, (int, float)):
            return 0
        if key in _STRING_TIME_KEYS and isinstance(value, str):
            return "<TIMESTAMP>"
        if key in _OPAQUE_ID_KEYS and isinstance(value, str):
            return self._replacement("response", value)

        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return self._normalize_string(value)
        if isinstance(value, Path):
            return self._normalize_string(str(value))
        if isinstance(value, dict):
            return {str(item_key): self.normalize(item, key=str(item_key)) for item_key, item in value.items()}
        if isinstance(value, list):
            return [self.normalize(item) for item in value]
        return self._normalize_string(str(value))

    def _normalize_string(self, value: str) -> str:
        normalized = value.replace(str(self.repo_root), "<REPO_ROOT>")
        normalized = _TIMESTAMP_PATTERN.sub("<TIMESTAMP>", normalized)
        normalized = _RFC1123_TIMESTAMP_PATTERN.sub("<RFC1123_TIMESTAMP>", normalized)
        for prefix, pattern in _TOKEN_PATTERNS:
            normalized = pattern.sub(
                lambda match: self._replacement(prefix, match.group(0)),
                normalized,
            )
        return normalized

    def _replacement(self, prefix: str, value: str) -> str:
        existing = self.replacements.get(value)
        if existing is not None:
            return existing
        next_index = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = next_index
        replacement = f"{prefix}_{next_index}"
        self.replacements[value] = replacement
        return replacement


@dataclass(frozen=True, slots=True)
class FixtureIdentity:
    family: str
    model_label: str
    scenario: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a raw LLM trace run into a stable replay fixture."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="A raw run directory to normalize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination fixture directory. Defaults under tests/fixtures/llm_traces/sanitized.",
    )
    parser.add_argument(
        "--model-label",
        help="Override the promoted model label used in the destination path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _family_from_provider(provider: str) -> str:
    family = _FAMILY_BY_PROVIDER.get(provider)
    if family is None:
        raise ValueError(f"Unsupported provider for replay fixture normalization: {provider}")
    return family


def _target_slug_from_source(source: Path) -> str:
    try:
        return source.resolve().relative_to(RAW_ROOT.resolve()).parts[0]
    except ValueError as exc:
        raise ValueError(f"Source must live under {RAW_ROOT}") from exc


def _default_model_label(meta: dict[str, Any], source: Path, *, family: str) -> str:
    requested_model = str(meta.get("requested_model") or "").strip()
    resolved_model = str(meta.get("resolved_model") or "").strip()

    if family == "anthropic" and requested_model:
        return requested_model
    if family in {"responses", "openresponses", "google"} and resolved_model:
        return resolved_model
    if family == "openai-chat" and requested_model:
        return requested_model

    target_slug = _target_slug_from_source(source)
    if target_slug.startswith(f"{family}."):
        return target_slug[len(f"{family}.") :]
    return target_slug


def _fixture_identity(source: Path, *, model_label_override: str | None = None) -> FixtureIdentity:
    meta_path = source / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Missing meta.json in {source}")
    meta = _read_json(meta_path)
    family = _family_from_provider(str(meta.get("provider") or ""))
    scenario = str(meta.get("scenario") or "").strip()
    if not scenario:
        raise ValueError(f"Missing scenario in {meta_path}")
    model_label = model_label_override or _default_model_label(meta, source, family=family)
    model_label = model_label.strip().replace("/", "-")
    if not model_label:
        raise ValueError(f"Unable to infer model label for {source}")
    return FixtureIdentity(family=family, model_label=model_label, scenario=scenario)


def _default_output_dir(source: Path, *, model_label_override: str | None = None) -> Path:
    identity = _fixture_identity(source, model_label_override=model_label_override)
    return SANITIZED_ROOT / identity.family / identity.model_label / identity.scenario


def _standard_name(path: Path) -> str:
    name = path.name
    if name == "meta.json":
        return "meta.json"
    if name == "result.json":
        return "result.json"
    if name == "listener_stream.jsonl":
        return "listener_stream.jsonl"
    if name == "listener_tools.jsonl":
        return "listener_tools.jsonl"
    if name == "google_stream_chunks.jsonl" or name.endswith("_chunks.jsonl") or name.endswith(".jsonl"):
        return "stream.jsonl"
    if (
        name == "google_stream_request.json"
        or name.endswith("_request.json")
        or name.endswith(".request.json")
    ):
        return "request.json"
    return name


def _classify_files(source: Path) -> dict[str, Path]:
    classified: dict[str, Path] = {}
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        standard_name = _standard_name(path)
        existing = classified.get(standard_name)
        if existing is not None:
            raise ValueError(
                f"Ambiguous fixture mapping for {source}: "
                f"{existing.name} and {path.name} both map to {standard_name}"
            )
        classified[standard_name] = path
    return classified


def _normalize_meta(
    *,
    meta: dict[str, Any],
    source: Path,
    identity: FixtureIdentity,
    files: dict[str, Path],
    normalizer: TraceNormalizer,
) -> dict[str, Any]:
    normalized = normalizer.normalize(meta)
    source_run = source.resolve().relative_to(REPO_ROOT.resolve())
    normalized["family"] = identity.family
    normalized["fixture_format_version"] = FIXTURE_FORMAT_VERSION
    normalized["fixture_files"] = sorted(files)
    normalized["model_label"] = identity.model_label
    normalized["scenario"] = identity.scenario
    normalized["source_run"] = normalizer.normalize(str(source_run))
    normalized["trace_files"] = [
        name for name in ("stream.jsonl", "request.json") if name in files
    ]
    return normalized


def _normalize_standard_file(
    *,
    standard_name: str,
    source_path: Path,
    output_dir: Path,
    normalizer: TraceNormalizer,
    normalized_meta: dict[str, Any] | None,
) -> None:
    output_path = output_dir / standard_name
    if standard_name == "meta.json":
        if normalized_meta is None:
            raise ValueError("normalized_meta must be provided for meta.json")
        _write_json(output_path, normalized_meta)
        return

    suffix = source_path.suffix.lower()
    if suffix == ".json":
        payload = _read_json(source_path)
        _write_json(output_path, normalizer.normalize(payload))
        return

    if suffix == ".jsonl":
        payloads = [normalizer.normalize(payload) for payload in _read_jsonl(source_path)]
        _write_jsonl(output_path, payloads)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def _validate_fixture_contract(output_dir: Path) -> None:
    required = ("meta.json", "result.json", "stream.jsonl")
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Normalized fixture missing required files: {missing_text}")

    meta = _read_json(output_dir / "meta.json")
    expected = output_dir.relative_to(SANITIZED_ROOT).parts
    if len(expected) != 3:
        raise ValueError(f"Unexpected fixture path shape: {output_dir}")

    family, model_label, scenario = expected
    if meta.get("family") != family:
        raise ValueError(f"meta.json family mismatch: expected {family}, got {meta.get('family')}")
    if meta.get("model_label") != model_label:
        raise ValueError(
            f"meta.json model_label mismatch: expected {model_label}, got {meta.get('model_label')}"
        )
    if meta.get("scenario") != scenario:
        raise ValueError(
            f"meta.json scenario mismatch: expected {scenario}, got {meta.get('scenario')}"
        )
    if meta.get("fixture_format_version") != FIXTURE_FORMAT_VERSION:
        raise ValueError(
            "meta.json fixture_format_version mismatch: "
            f"expected {FIXTURE_FORMAT_VERSION}, got {meta.get('fixture_format_version')}"
        )


def main() -> None:
    args = _parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"Not a directory: {source}")

    identity = _fixture_identity(source, model_label_override=args.model_label)
    output_dir = (
        args.output.resolve()
        if args.output
        else _default_output_dir(source, model_label_override=args.model_label)
    )

    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)

    classified_files = _classify_files(source)
    normalizer = TraceNormalizer(repo_root=REPO_ROOT)
    normalized_meta = _normalize_meta(
        meta=_read_json(source / "meta.json"),
        source=source,
        identity=identity,
        files=classified_files,
        normalizer=normalizer,
    )

    for standard_name, source_path in classified_files.items():
        _normalize_standard_file(
            standard_name=standard_name,
            source_path=source_path,
            output_dir=output_dir,
            normalizer=normalizer,
            normalized_meta=normalized_meta,
        )

    _validate_fixture_contract(output_dir)

    print(
        "Normalized "
        f"{source.relative_to(REPO_ROOT)} -> {output_dir.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
