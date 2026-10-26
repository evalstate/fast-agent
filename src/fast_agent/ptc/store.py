"""Values sandbox code keeps between ``execute_python`` runs.

Every run starts in a fresh Monty sandbox, so ``store``/``load`` are how a run hands values to
later ones. Values live host-side as tagged JSON: that bounds their size exactly, isolates them
from later mutation, and makes the store a plain document the session can persist and restore.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping
from typing import Any

STORE_FUNCTION_NAMES = frozenset({"store", "load", "stored"})
STORE_STUBS = """\
async def store(key: str, value: Any) -> None:
    \"\"\"Keep a value for later runs, replacing any value under the same key. Plain data only:
    str, bytes, int, float, bool, None and lists, tuples, dicts and sets of them.\"\"\"
    ...

async def load(key: str, default: Any = ...) -> Any:
    \"\"\"Return a value kept by `store`, or `default`; raises KeyError when it is missing and
    no default is given.\"\"\"
    ...

async def stored() -> dict[str, int]:
    \"\"\"Return the stored keys with the size of each value in bytes.\"\"\"
    ...
"""
MAX_STORE_BYTES = 64 * 1024 * 1024
_MAX_KEY_LENGTH = 200
_STATE_VERSION = 1
# Tags for values JSON cannot represent directly. A dict whose keys are all strings and do not
# start with "$" is written as an object; any other dict becomes a list of pairs.
_BYTES, _TUPLE, _SET, _FROZENSET, _DICT, _FLOAT = (
    "$bytes",
    "$tuple",
    "$set",
    "$frozenset",
    "$dict",
    "$float",
)
_MISSING = object()


class SandboxStore:
    """Host-side key/value store exposed to sandbox code as ``store``, ``load`` and ``stored``."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._size = 0
        # Bumped on every change, so savers can skip writing an unchanged store.
        self.revision = 0

    def __len__(self) -> int:
        return len(self._values)

    def functions(self) -> dict[str, Any]:
        return {"store": self._store, "load": self._load, "stored": self._stored}

    def clear(self) -> None:
        if self._values:
            self._values.clear()
            self._size = 0
            self.revision += 1

    def to_state(self) -> str:
        """Serialize the store as a JSON document for session persistence."""
        values = ",".join(f"{json.dumps(key)}:{encoded}" for key, encoded in self._values.items())
        return f'{{"version":{_STATE_VERSION},"values":{{{values}}}}}'

    def restore(self, state: str) -> None:
        """Replace the contents with a document written by ``to_state``."""
        document = json.loads(state)
        if not isinstance(document, Mapping) or document.get("version") != _STATE_VERSION:
            raise ValueError("unsupported sandbox store document")
        values = document.get("values")
        if not isinstance(values, Mapping):
            raise ValueError("sandbox store document has no values")
        self._values = {str(key): _dumps(value) for key, value in values.items()}
        self._size = sum(len(key) + len(value) for key, value in self._values.items())
        self.revision += 1

    async def _store(self, key: Any = _MISSING, value: Any = _MISSING) -> None:
        if key is _MISSING or value is _MISSING:
            raise TypeError("store() needs key= and value=")
        _check_key(key)
        encoded = _dumps(_encode(value))
        previous = self._values.get(key)
        size = self._size - (len(key) + len(previous) if previous is not None else 0)
        size += len(key) + len(encoded)
        if size > MAX_STORE_BYTES:
            raise ValueError(
                f"store(): {key!r} would bring the store to {size} bytes, over its "
                f"{MAX_STORE_BYTES // (1024 * 1024)} MB limit; keep large data in files instead"
            )
        self._values[key] = encoded
        self._size = size
        self.revision += 1

    async def _load(self, key: Any = _MISSING, default: Any = _MISSING) -> Any:
        if key is _MISSING:
            raise TypeError("load() needs key=")
        _check_key(key)
        encoded = self._values.get(key)
        if encoded is None:
            if default is not _MISSING:
                return default
            kept = ", ".join(repr(name) for name in self._values) or "nothing"
            raise KeyError(f"{key!r} was not stored; stored keys: {kept}")
        return _decode(json.loads(encoded))

    async def _stored(self) -> dict[str, int]:
        return {key: len(encoded) for key, encoded in self._values.items()}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _check_key(key: Any) -> None:
    if not isinstance(key, str) or not key:
        raise TypeError("the store key must be a non-empty str")
    if len(key) > _MAX_KEY_LENGTH:
        raise ValueError(f"store keys are limited to {_MAX_KEY_LENGTH} characters")


def _encode(value: Any) -> Any:
    match value:
        case None | bool() | int() | str():
            return value
        case float():
            return value if math.isfinite(value) else {_FLOAT: repr(value)}
        case bytes():
            return {_BYTES: base64.b64encode(value).decode()}
        case list():
            return [_encode(item) for item in value]
        case tuple():
            return {_TUPLE: [_encode(item) for item in value]}
        case frozenset():
            return {_FROZENSET: [_encode(item) for item in value]}
        case set():
            return {_SET: [_encode(item) for item in value]}
        case dict():
            if all(isinstance(key, str) and not key.startswith("$") for key in value):
                return {key: _encode(item) for key, item in value.items()}
            return {_DICT: [[_encode(key), _encode(item)] for key, item in value.items()]}
    raise TypeError(
        f"store(): a {type(value).__name__} value cannot be stored; store plain data (str, bytes, "
        "int, float, bool, None and lists, tuples, dicts and sets of them)"
    )


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        ((tag, payload),) = value.items()
        if (decode := _TAG_DECODERS.get(tag)) is not None:
            return decode(payload)
    return {key: _decode(item) for key, item in value.items()}


_TAG_DECODERS: dict[str, Any] = {
    _BYTES: base64.b64decode,
    _TUPLE: lambda items: tuple(_decode(item) for item in items),
    _SET: lambda items: {_decode(item) for item in items},
    _FROZENSET: lambda items: frozenset(_decode(item) for item in items),
    _DICT: lambda pairs: {_decode(key): _decode(item) for key, item in pairs},
    _FLOAT: float,
}
