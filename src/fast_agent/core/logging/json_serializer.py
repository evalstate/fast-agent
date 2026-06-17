from __future__ import annotations

import dataclasses
import inspect
import os
import warnings
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

import httpx

from fast_agent.core.logging import logger
from fast_agent.utils.text import strip_casefold

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

_TEXTLIKE_BYTES_TYPES = (bytes, bytearray, memoryview)
_JSON_NATIVE_SCALAR_TYPES = (str, bool, int, float)
_ISOFORMAT_SCALAR_TYPES = (datetime, date)
_STRINGIFIED_SCALAR_TYPES = (Decimal, UUID, Path)
_MODEL_LIKE_MISSING = object()


def snapshot_json_value(obj: object | None) -> JsonValue:
    """Capture a JSON-safe snapshot for persistence/debugging."""
    return _snapshot_json_value(obj, seen=set())


def _snapshot_scalar(obj: object) -> JsonValue | None:
    if isinstance(obj, _JSON_NATIVE_SCALAR_TYPES):
        return obj
    if isinstance(obj, _ISOFORMAT_SCALAR_TYPES):
        return obj.isoformat()
    if isinstance(obj, _STRINGIFIED_SCALAR_TYPES):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, _TEXTLIKE_BYTES_TYPES):
        return str(obj)
    return None


def _snapshot_circular_reference(obj: object) -> str:
    return f"<circular-reference:{type(obj).__name__}>"


def _snapshot_mapping(obj: Mapping[Any, object], *, seen: set[int]) -> JsonValue:
    obj_id = id(obj)
    if obj_id in seen:
        return _snapshot_circular_reference(obj)

    seen.add(obj_id)
    try:
        return {str(key): _snapshot_json_value(value, seen=seen) for key, value in obj.items()}
    finally:
        seen.remove(obj_id)


def _snapshot_iterable(obj: Iterable[object], *, seen: set[int]) -> JsonValue:
    obj_id = id(obj)
    if obj_id in seen:
        return _snapshot_circular_reference(obj)

    seen.add(obj_id)
    try:
        return [_snapshot_json_value(item, seen=seen) for item in obj]
    finally:
        seen.remove(obj_id)


def _snapshot_extracted_object(obj: object, extracted: object, *, seen: set[int]) -> JsonValue:
    obj_id = id(obj)
    if obj_id in seen:
        return _snapshot_circular_reference(obj)

    seen.add(obj_id)
    try:
        return _snapshot_json_value(extracted, seen=seen)
    finally:
        seen.remove(obj_id)


def _snapshot_json_value(obj: object | None, *, seen: set[int]) -> JsonValue:
    if obj is None:
        return None

    scalar = _snapshot_scalar(obj)
    if scalar is not None:
        return scalar

    if isinstance(obj, Mapping):
        return _snapshot_mapping(obj, seen=seen)

    extracted = _extract_snapshot_source(obj)
    if extracted is None or extracted is obj:
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray, memoryview)):
            return _snapshot_iterable(obj, seen=seen)
        return str(obj)

    return _snapshot_extracted_object(obj, extracted, seen=seen)


def _extract_snapshot_source(obj: object) -> object | None:
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            with suppress(Exception):
                return model_dump()
        except Exception:
            return _extract_mapping_snapshot_source(obj)

    return _extract_mapping_snapshot_source(obj)


def _extract_mapping_snapshot_source(obj: object) -> object | None:
    dict_method = getattr(obj, "dict", None)
    if callable(dict_method):
        with suppress(Exception):
            return dict_method()

    raw_dict = getattr(obj, "__dict__", None)
    if isinstance(raw_dict, Mapping) and raw_dict:
        return raw_dict

    return None


def _serialize_external_special_object(obj: object) -> str | None:
    if isinstance(obj, httpx.Response):
        return f"<httpx.Response [{obj.status_code}] {obj.url}>"
    if isinstance(obj, logger.Logger):
        return "<logging: logger>"
    return None


class JSONSerializer:
    """
    A robust JSON serializer that handles various Python objects by attempting
    different serialization strategies recursively.
    """

    MAX_DEPTH = 99  # Maximum recursion depth

    # Fields that are likely to contain sensitive information
    SENSITIVE_FIELDS: ClassVar[set[str]] = {
        "api_key",
        "secret",
        "password",
        "token",
        "auth",
        "private_key",
        "client_secret",
        "access_token",
        "refresh_token",
    }

    def __init__(self) -> None:
        # Set of already processed objects to prevent infinite recursion
        self._processed_objects: set[int] = set()
        # Check if secrets should be logged in full
        self._log_secrets = os.getenv("LOG_SECRETS", "").upper() == "TRUE"

    def _redact_sensitive_value(self, value: str) -> str:
        """Redact sensitive values to show only first 10 chars."""
        if not value or not isinstance(value, str):
            return value
        if self._log_secrets:
            return value
        if len(value) <= 10:
            return value + "....."
        return value[:10] + "....."

    def serialize(self, obj: Any) -> Any:
        """Main entry point for serialization."""
        # Reset processed objects for new serialization
        self._processed_objects.clear()
        return self._serialize_object(obj, depth=0)

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key likely contains sensitive information."""
        key = strip_casefold(str(key))
        return any(sensitive in key for sensitive in self.SENSITIVE_FIELDS)

    def _serialize_special_object(self, obj: Any) -> Any:
        external_value = _serialize_external_special_object(obj)
        if external_value is not None:
            return external_value
        scalar = _snapshot_scalar(obj)
        if scalar is not None:
            return scalar
        if callable(obj):
            return f"<callable: {obj.__name__}>"
        return None

    def _serialize_model_like(self, obj: Any, depth: int) -> Any:
        extracted = self._extract_model_like_source(obj)
        if extracted is not _MODEL_LIKE_MISSING:
            return self._serialize_object(extracted, depth + 1)

        return None

    def _extract_model_like_source(self, obj: Any) -> Any:
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            module = getattr(obj, "__module__", "")
            if module.startswith("openai.types.responses"):
                return model_dump(warnings="none")
            return model_dump()

        dict_method = getattr(obj, "dict", None)
        if callable(dict_method):
            return dict_method()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        for method_name in ("to_json", "to_dict"):
            method = getattr(obj, method_name, None)
            if callable(method):
                return method()

        return _MODEL_LIKE_MISSING

    def _serialize_mapping(self, obj: Mapping[Any, Any], depth: int) -> dict[str, Any]:
        return {
            str(key): self._redact_sensitive_value(value)
            if self._is_sensitive_key(key)
            else self._serialize_object(value, depth + 1)
            for key, value in obj.items()
        }

    def _serialize_public_attributes(self, obj: Any, depth: int) -> Any:
        raw_dict = getattr(obj, "__dict__", None)
        if isinstance(raw_dict, Mapping):
            return self._serialize_object(raw_dict, depth + 1)

        members = inspect.getmembers(obj)
        if not members:
            return None

        return {
            name: self._redact_sensitive_value(value)
            if self._is_sensitive_key(name)
            else self._serialize_object(value, depth + 1)
            for name, value in members
            if not name.startswith("_") and not inspect.ismethod(value)
        }

    def _depth_exceeded_value(self, obj: Any) -> str:
        warnings.warn(
            f"Maximum recursion depth ({self.MAX_DEPTH}) exceeded while serializing object of type {type(obj).__name__} parent: {type(self._parent_obj).__name__}",
            stacklevel=2,
        )
        return str(obj)

    def _serialize_by_strategy(self, obj: Any, depth: int) -> Any:
        special_value = self._serialize_special_object(obj)
        if special_value is not None:
            return special_value

        model_value = self._serialize_model_like(obj, depth)
        if model_value is not None:
            return model_value

        if isinstance(obj, Mapping):
            return self._serialize_mapping(obj, depth)

        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return [self._serialize_object(item, depth + 1) for item in obj]

        public_attributes = self._serialize_public_attributes(obj, depth)
        if public_attributes is not None:
            return public_attributes

        return str(obj)

    def _serialize_object(self, obj: Any, depth: int = 0) -> Any:
        """Recursively serialize an object using various strategies."""
        # Handle None
        if obj is None:
            return None

        if depth == 0:
            self._parent_obj = obj
        # Check depth
        if depth > self.MAX_DEPTH:
            return self._depth_exceeded_value(obj)

        # Prevent infinite recursion
        obj_id = id(obj)
        if obj_id in self._processed_objects:
            return str(obj)
        self._processed_objects.add(obj_id)

        # Try different serialization strategies in order
        try:
            return self._serialize_by_strategy(obj, depth)

        except Exception as e:
            # If all serialization attempts fail, return string representation
            return f"<unserializable: {type(obj).__name__}, error: {e!s}>"

    def __call__(self, obj: Any) -> Any:
        """Make the serializer callable."""
        return self.serialize(obj)
