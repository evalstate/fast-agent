"""Helpers for parsing URL elicitation required error payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from mcp.types import ElicitRequestURLParams
from pydantic import ValidationError


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

    if not isinstance(data, dict):
        issues.append(f"error.data must be an object, got {type(data).__name__}")
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    data_dict = cast("dict[str, object]", data)
    raw_elicitations = data_dict.get("elicitations")
    if raw_elicitations is None:
        issues.append("error.data.elicitations is missing")
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    if not isinstance(raw_elicitations, list):
        issues.append(
            "error.data.elicitations must be a list, "
            f"got {type(raw_elicitations).__name__}"
        )
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    if not raw_elicitations:
        issues.append("error.data.elicitations is empty")
        return ParsedURLElicitationErrorData(elicitations=elicitations, issues=issues)

    for index, raw_elicitation in enumerate(raw_elicitations):
        if not isinstance(raw_elicitation, dict):
            issues.append(
                f"error.data.elicitations[{index}] must be an object, "
                f"got {type(raw_elicitation).__name__}"
            )
            continue

        normalized_elicitation, non_compliant_issue = _normalize_elicitation_payload(raw_elicitation)
        if non_compliant_issue is not None:
            issues.append(f"error.data.elicitations[{index}] is non-compliant: {non_compliant_issue}")

        try:
            elicitation = ElicitRequestURLParams.model_validate(normalized_elicitation)
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


def _normalize_elicitation_payload(raw_elicitation: dict[str, object]) -> tuple[dict[str, object], str | None]:
    """Normalize provider variants while recording MCP wire-format non-compliance."""
    if "elicitationId" in raw_elicitation:
        return raw_elicitation, None

    snake_case_value = raw_elicitation.get("elicitation_id")
    if isinstance(snake_case_value, str) and snake_case_value:
        normalized = dict(raw_elicitation)
        normalized["elicitationId"] = snake_case_value
        return (
            normalized,
            "uses 'elicitation_id'; expected MCP field 'elicitationId'",
        )

    return raw_elicitation, None
