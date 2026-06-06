"""Helpers for parsing URL elicitation required error payloads."""

from __future__ import annotations

from dataclasses import dataclass

from mcp.types import ElicitRequestURLParams
from pydantic import ValidationError

from fast_agent.utils.text import strip_str_to_none
from fast_agent.utils.type_narrowing import is_str_object_dict


@dataclass(slots=True)
class ParsedURLElicitationErrorData:
    """Parsed URL elicitation required payload with validation issues."""

    elicitations: list[ElicitRequestURLParams]
    issues: list[str]


@dataclass(slots=True)
class URLElicitationDisplayItem:
    """Display-friendly URL elicitation entry."""

    message: str
    url: str
    elicitation_id: str


@dataclass(slots=True)
class URLElicitationRequiredDisplayPayload:
    """Normalized URL elicitation payload attached to request failures."""

    server_name: str
    request_method: str
    elicitations: list[URLElicitationDisplayItem]
    issues: list[str]


@dataclass(frozen=True, slots=True)
class _NormalizedElicitationPayload:
    payload: dict[str, object]
    non_compliant_issue: str | None = None


def parse_url_elicitation_required_data(data: object) -> ParsedURLElicitationErrorData:
    """Parse and validate ``error.data`` for URL elicitation required errors.

    Returns both successfully parsed URL elicitations and human-readable issues
    for malformed payload content.
    """

    elicitations: list[ElicitRequestURLParams] = []
    issues: list[str] = []

    if data is None:
        issues.append("error.data is missing")
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    if not is_str_object_dict(data):
        issues.append(f"error.data must be an object, got {type(data).__name__}")
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    raw_elicitations = data.get("elicitations")
    if raw_elicitations is None:
        issues.append("error.data.elicitations is missing")
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    if not isinstance(raw_elicitations, list):
        issues.append(
            f"error.data.elicitations must be a list, got {type(raw_elicitations).__name__}"
        )
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    if not raw_elicitations:
        issues.append("error.data.elicitations is empty")
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    for index, raw_elicitation in enumerate(raw_elicitations):
        if not is_str_object_dict(raw_elicitation):
            issues.append(
                f"error.data.elicitations[{index}] must be an object, "
                f"got {type(raw_elicitation).__name__}"
            )
            continue

        normalized_elicitation = _normalize_elicitation_payload(raw_elicitation)
        if normalized_elicitation.non_compliant_issue is not None:
            issues.append(
                f"error.data.elicitations[{index}] is non-compliant: "
                f"{normalized_elicitation.non_compliant_issue}"
            )

        try:
            elicitation = ElicitRequestURLParams.model_validate(normalized_elicitation.payload)
        except ValidationError as exc:
            details = _format_validation_error(exc)
            issues.append(f"error.data.elicitations[{index}] is invalid: {details}")
            continue

        elicitations.append(elicitation)

    return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)


def build_url_elicitation_required_display_payload(
    data: object,
    *,
    server_name: str,
    request_method: str,
) -> URLElicitationRequiredDisplayPayload:
    """Build normalized display payload from URL elicitation required error data."""
    parsed = parse_url_elicitation_required_data(data)
    items = [
        URLElicitationDisplayItem(
            message=item.message,
            url=item.url,
            elicitation_id=item.elicitationId,
        )
        for item in parsed.elicitations
    ]
    return URLElicitationRequiredDisplayPayload(
        server_name=server_name,
        request_method=request_method,
        elicitations=items,
        issues=parsed.issues,
    )


def _format_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "validation error"

    first_error = errors[0]
    loc_items = first_error.get("loc", ())
    loc = ".".join(str(item) for item in loc_items) if loc_items else "field"
    message = str(first_error.get("msg", "validation error"))
    return f"{loc}: {message}"


def _normalize_elicitation_payload(
    raw_elicitation: dict[str, object],
) -> _NormalizedElicitationPayload:
    """Normalize provider variants while recording MCP wire-format non-compliance."""
    if "elicitationId" in raw_elicitation:
        return _NormalizedElicitationPayload(payload=raw_elicitation)

    elicitation_id = strip_str_to_none(raw_elicitation.get("elicitation_id"))
    if elicitation_id is not None:
        normalized = dict(raw_elicitation)
        normalized["elicitationId"] = elicitation_id
        return _NormalizedElicitationPayload(
            payload=normalized,
            non_compliant_issue="uses 'elicitation_id'; expected MCP field 'elicitationId'",
        )

    return _NormalizedElicitationPayload(payload=raw_elicitation)
