from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_VERTEX_PROJECT_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_PROJECT_ID",
    "GCLOUD_PROJECT",
    "GCP_PROJECT",
)
_VERTEX_LOCATION_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_VERTEX_LOCATION",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_CLOUD_REGION",
    "CLOUD_ML_REGION",
    "VERTEX_REGION",
)
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@dataclass(frozen=True, slots=True)
class AnthropicVertexConfig:
    enabled: bool = False
    project_id: str | None = None
    location: str | None = None
    base_url: str | None = None


@dataclass(frozen=True, slots=True)
class GoogleAdcStatus:
    available: bool
    project_id: str | None = None
    error: Exception | None = None
    credentials: object | None = None


def _get_value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _env_value(env_vars: tuple[str, ...]) -> str | None:
    for env_var in env_vars:
        value = _clean_str(os.getenv(env_var))
        if value is not None:
            return value
    return None


def anthropic_vertex_source(config: Any) -> Any:
    anthropic_cfg = _get_value(config, "anthropic")
    return _get_value(anthropic_cfg, "vertex_ai")


def anthropic_vertex_config(config: Any) -> AnthropicVertexConfig:
    source = anthropic_vertex_source(config)
    if source is None:
        return AnthropicVertexConfig()

    return AnthropicVertexConfig(
        enabled=bool(_get_value(source, "enabled")),
        project_id=_clean_str(_get_value(source, "project_id")),
        location=_clean_str(_get_value(source, "location")),
        base_url=_clean_str(_get_value(source, "base_url")),
    )


def anthropic_vertex_intent(config: Any) -> bool:
    cfg = anthropic_vertex_config(config)
    return bool(
        cfg.enabled
        or cfg.project_id
        or cfg.location
        or cfg.base_url
        or _env_value(_VERTEX_PROJECT_ENV_VARS)
        or _env_value(_VERTEX_LOCATION_ENV_VARS)
    )


def anthropic_vertex_enabled(config: Any) -> bool:
    return anthropic_vertex_config(config).enabled


def detect_google_adc() -> GoogleAdcStatus:
    try:
        import google.auth

        credentials, project_id = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    except Exception as exc:  # pragma: no cover - exercised via callers
        return GoogleAdcStatus(available=False, error=exc)

    return GoogleAdcStatus(
        available=True,
        project_id=_clean_str(project_id),
        credentials=credentials,
    )


def resolve_anthropic_vertex_project_id(
    config: Any,
    *,
    adc_status: GoogleAdcStatus | None = None,
) -> str | None:
    cfg = anthropic_vertex_config(config)
    if cfg.project_id is not None:
        return cfg.project_id

    env_project = _env_value(_VERTEX_PROJECT_ENV_VARS)
    if env_project is not None:
        return env_project

    if adc_status is None:
        adc_status = detect_google_adc()
    return adc_status.project_id


def resolve_anthropic_vertex_location(config: Any) -> str | None:
    cfg = anthropic_vertex_config(config)
    if cfg.location is not None:
        return cfg.location

    env_location = _env_value(_VERTEX_LOCATION_ENV_VARS)
    if env_location is not None:
        return env_location

    return "global"
def anthropic_vertex_ready(
    config: Any,
    *,
    adc_status: GoogleAdcStatus | None = None,
) -> tuple[bool, str | None]:
    if not anthropic_vertex_intent(config):
        return (False, None)

    if adc_status is None:
        adc_status = detect_google_adc()

    project_id = resolve_anthropic_vertex_project_id(config, adc_status=adc_status)
    if project_id is None:
        return (
            False,
            "Google Cloud project not found",
        )
    if not adc_status.available:
        return (
            False,
            "Google ADC not found",
        )
    return (True, None)
