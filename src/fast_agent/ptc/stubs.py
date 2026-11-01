"""Render tool definitions as Python function stubs for the programmatic tool sandbox.

Every tool becomes a keyword-only ``async def`` whose parameters come from the tool's JSON
input schema. The same stubs are shown to the model and used by Monty's type checker, so
code that calls a tool incorrectly is rejected before any tool runs.
"""

from __future__ import annotations

import builtins
import json
import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_agent.ptc.store import STORE_FUNCTION_NAMES, STORE_STUBS

if TYPE_CHECKING:
    from mcp_types import Tool

# Modules importable inside Monty; a tool must not shadow them.
SANDBOX_MODULES = frozenset(
    {
        "asyncio",
        "base64",
        "binascii",
        "bisect",
        "collections",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "posixpath",
        "random",
        "re",
        "sys",
        "time",
        "tomllib",
        "typing",
        "unicodedata",
    }
)
# Helpers defined by the runtime (the prelude's settle, the store functions) must remain
# callable alongside tool functions.
_RESERVED_GLOBALS = frozenset(dir(builtins)) | SANDBOX_MODULES | {"settle"} | STORE_FUNCTION_NAMES
_JSON_SCALAR_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
}
_MAX_ANNOTATION_DEPTH = 4
_MAX_RETURNS_CHARS = 1200


@dataclass(frozen=True, slots=True)
class PtcParameter:
    python_name: str
    schema_name: str
    annotation: str
    required: bool
    description: str | None


@dataclass(frozen=True, slots=True)
class PtcFunction:
    """A tool exposed inside the sandbox as an async Python function."""

    python_name: str
    tool_name: str
    description: str
    parameters: tuple[PtcParameter, ...]
    accepts_extra: bool
    returns: str
    returns_shape: str | None = None
    """Compact rendering of the output schema, e.g. ``{id: str, tags?: list[str]}``."""

    def bind_arguments(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Map supplied keywords to schema names, preserving explicit null values."""
        by_python_name = {parameter.python_name: parameter for parameter in self.parameters}
        arguments: dict[str, Any] = {}
        for name, value in kwargs.items():
            parameter = by_python_name.get(name)
            if parameter is None:
                arguments[name] = value
            else:
                arguments[parameter.schema_name] = value
        return arguments

    def signature(self) -> str:
        """Return the one-line call signature, e.g. ``name(*, query: str) -> Any``."""
        parameters = ", ".join(_render_parameter(parameter) for parameter in self._ordered())
        if self.accepts_extra:
            extra_name = "kwargs"
            parameter_names = {parameter.python_name for parameter in self.parameters}
            while extra_name in parameter_names:
                extra_name += "_"
            extra = f"**{extra_name}: Any"
            parameters = f"{parameters}, {extra}" if parameters else extra
        if parameters and not parameters.startswith("**"):
            parameters = f"*, {parameters}"
        return f"{self.python_name}({parameters}) -> {self.returns}"

    def example_call(self) -> str:
        """Return an awaited call with required arguments, e.g. ``await bash(command=...)``."""
        arguments = ", ".join(
            f"{parameter.python_name}=..." for parameter in self._ordered() if parameter.required
        )
        return f"await {self.python_name}({arguments})"

    def stub(self) -> str:
        lines = [f"async def {self.signature()}:"]
        docstring = self._docstring()
        if docstring:
            first, *rest = _docstring_literal(docstring).split("\n")
            body = "\n".join([first, *(f"    {line}" if line else "" for line in rest)])
            lines.append(f'    """{body}"""')
        lines.append("    ...")
        return "\n".join(lines)

    def _ordered(self) -> list[PtcParameter]:
        # Required (no default) parameters first; keyword-only so order is cosmetic.
        return sorted(self.parameters, key=lambda parameter: not parameter.required)

    def _docstring(self) -> str:
        parts: list[str] = []
        if self.tool_name != self.python_name:
            parts.append(f"Tool `{self.tool_name}`.")
        if self.description:
            parts.append(self.description.strip())
        documented = [parameter for parameter in self._ordered() if parameter.description]
        if documented:
            arg_lines = [
                f"    {parameter.python_name}: {_one_line(parameter.description or '')}"
                for parameter in documented
            ]
            parts.append("Args:\n" + "\n".join(arg_lines))
        if self.returns_shape:
            parts.append(f"Returns: {self.returns_shape}")
        return "\n\n".join(parts)


def tool_ptc_options(tool: Tool) -> Mapping[str, Any]:
    """Return the ``meta["fast_agent"]["ptc"]`` options a tool declares for PTC.

    Supported keys: ``expose`` (``"nested"`` default; ``"direct"`` to keep the tool next to
    ``execute_python`` instead; ``"both"`` for a sandbox function that is also directly
    callable); ``returns`` (JSON schema of the value sandbox code receives, for
    tools whose program-facing value differs from their model-facing text); and
    ``input_schema``/``description`` (the keyword arguments and documentation sandbox code
    sees, for tools whose model-facing wire format, such as freeform text, is not a keyword
    call; the tool must accept those arguments from programmatic calls).
    """
    fast_agent_meta = (tool.meta or {}).get("fast_agent")
    if not isinstance(fast_agent_meta, Mapping):
        return {}
    options = fast_agent_meta.get("ptc")
    return options if isinstance(options, Mapping) else {}


def build_functions(tools: Sequence[Tool]) -> dict[str, PtcFunction]:
    """Build sandbox functions keyed by Python name, resolving name collisions."""
    functions: dict[str, PtcFunction] = {}
    for tool in tools:
        python_name = _unique(_global_identifier(tool.name), functions)
        functions[python_name] = _build_function(tool, python_name)
    return functions


# Builtins the Monty runtime provides but its bundled type-checker stubs omit (pydantic-monty
# 1.3 and 1.4), declared so code using them passes type checking.
_MISSING_BUILTIN_STUBS = """
def open(
    file: Any,
    mode: str = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
) -> Any: ...


class FileNotFoundError(OSError): ...
class FileExistsError(OSError): ...
class IsADirectoryError(OSError): ...
class NotADirectoryError(OSError): ...
class PermissionError(OSError): ...
"""


def render_stubs(functions: Mapping[str, PtcFunction]) -> str:
    body = "\n\n\n".join(function.stub() for function in functions.values())
    return (
        f"from typing import Any, Literal\n\n{_MISSING_BUILTIN_STUBS}\n\n{STORE_STUBS}\n\n{body}\n"
    )


def _build_function(tool: Tool, python_name: str) -> PtcFunction:
    options = tool_ptc_options(tool)
    ptc_schema = options.get("input_schema")
    schema = ptc_schema if isinstance(ptc_schema, Mapping) else tool.input_schema
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    required = schema.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    definitions = _definitions(schema)

    parameters: list[PtcParameter] = []
    used_names: set[str] = set()
    for schema_name, property_schema in properties.items():
        parameter_name = _parameter_identifier(str(schema_name))
        while parameter_name in used_names:
            parameter_name = f"{parameter_name}_"
        used_names.add(parameter_name)
        is_required = schema_name in required_names
        # Optional parameters keep their exact schema type; only nullable ones accept None.
        annotation = _annotation(property_schema, definitions)
        description = (
            property_schema.get("description") if isinstance(property_schema, Mapping) else None
        )
        parameters.append(
            PtcParameter(
                python_name=parameter_name,
                schema_name=str(schema_name),
                annotation=annotation,
                required=is_required,
                description=description if isinstance(description, str) else None,
            )
        )

    # Only an explicitly open schema gets **kwargs. JSON Schema allows extra properties when
    # the key is absent, but treating that as closed keeps typo detection in the type check.
    additional = schema.get("additionalProperties")
    accepts_extra = additional is True or isinstance(additional, Mapping)
    returns_schema = options.get("returns")
    output_schema = returns_schema if isinstance(returns_schema, Mapping) else tool.output_schema
    description = options.get("description")
    return PtcFunction(
        python_name=python_name,
        tool_name=tool.name,
        description=description if isinstance(description, str) else tool.description or "",
        parameters=tuple(parameters),
        accepts_extra=accepts_extra,
        returns="dict[str, Any]" if output_schema else "Any",
        returns_shape=_returns_shape(output_schema, _definitions(output_schema or {})),
    )


def _definitions(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    definitions: dict[str, Any] = {}
    for key in ("definitions", "$defs"):
        value = schema.get(key)
        if isinstance(value, Mapping):
            definitions.update(value)
    return definitions


def _returns_shape(schema: Mapping[str, Any] | None, definitions: Mapping[str, Any]) -> str | None:
    """Render an output schema as a compact TypedDict-like shape.

    Top-level field descriptions become trailing comments, one field per line; nested
    descriptions are dropped.
    """
    if not schema:
        return None
    properties = schema.get("properties")
    described = isinstance(properties, Mapping) and any(
        isinstance(field, Mapping) and isinstance(field.get("description"), str)
        for field in properties.values()
    )
    if described:
        rendered = _object_shape(schema, definitions, 0, comments=True)
    else:
        rendered = _annotation(schema, definitions, shape=True)
    if len(rendered) > _MAX_RETURNS_CHARS:
        return f"{rendered[:_MAX_RETURNS_CHARS]}…"
    return rendered


def _object_shape(
    schema: Mapping[str, Any], definitions: Mapping[str, Any], depth: int, *, comments: bool = False
) -> str:
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    required = schema.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    fields: list[tuple[str, str]] = []
    for name, field in properties.items():
        key = name if str(name).isidentifier() else json.dumps(str(name))
        optional = "" if name in required_names else "?"
        rendered = f"{key}{optional}: {_annotation(field, definitions, depth + 1, shape=True)}"
        description = field.get("description") if isinstance(field, Mapping) else None
        fields.append((rendered, _one_line(description) if isinstance(description, str) else ""))
    if not comments:
        return "{" + ", ".join(field for field, _ in fields) + "}"
    lines = [f"    {field},{f'  # {note}' if note else ''}" for field, note in fields]
    return "{\n" + "\n".join(lines) + "\n}"


def _docstring_literal(text: str) -> str:
    """Escape ``text`` for the body of a triple-double-quoted string literal."""
    escaped = "".join(
        char if char == "\n" or char.isprintable() else char.encode("unicode_escape").decode()
        for char in text.replace("\\", "\\\\")
    )
    escaped = escaped.replace('"""', '\\"\\"\\"')
    # A trailing quote would merge with the closing delimiter.
    return escaped[:-1] + '\\"' if escaped.endswith('"') else escaped


def _render_parameter(parameter: PtcParameter) -> str:
    rendered = f"{parameter.python_name}: {parameter.annotation}"
    # `= ...` marks "has a default" in a stub: omitting it lets the tool apply its own default.
    return rendered if parameter.required else f"{rendered} = ..."


def _annotation(
    schema: Any, definitions: Mapping[str, Any], depth: int = 0, *, shape: bool = False
) -> str:
    return _union(_union_members(schema, definitions, depth, shape=shape))


def _union_members(
    schema: Any, definitions: Mapping[str, Any], depth: int, *, shape: bool = False
) -> list[str]:
    """Return flattened union members so nested unions never repeat a member.

    ``shape`` renders documentation rather than a type annotation: objects show their fields
    and enums are written as ``"a" | "b"``.
    """
    if not isinstance(schema, Mapping) or depth > _MAX_ANNOTATION_DEPTH:
        return ["Any"]
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _resolve_local_ref(ref, definitions)
        if target is None:
            return ["Any"]
        siblings = {key: value for key, value in schema.items() if key != "$ref"}
        return _union_members({**target, **siblings}, definitions, depth + 1, shape=shape)
    values = [schema["const"]] if "const" in schema else schema.get("enum")
    if isinstance(values, list) and values:
        return _literal_members(values) if shape else [_literal(values)]
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            return [
                member
                for option in options
                for member in _union_members(option, definitions, depth + 1, shape=shape)
            ]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return [
            member
            for item in schema_type
            for member in _union_members(
                {**schema, "type": item}, definitions, depth + 1, shape=shape
            )
        ]
    if isinstance(schema_type, str) and schema_type in _JSON_SCALAR_TYPES:
        return [_JSON_SCALAR_TYPES[schema_type]]
    if schema_type == "array":
        return [f"list[{_annotation(schema.get('items'), definitions, depth + 1, shape=shape)}]"]
    if schema_type in ("object", None) and shape and isinstance(schema.get("properties"), Mapping):
        return [_object_shape(schema, definitions, depth)]
    if schema_type == "object":
        additional = schema.get("additionalProperties")
        if shape and isinstance(additional, Mapping):
            return [f"dict[str, {_annotation(additional, definitions, depth + 1, shape=True)}]"]
        return ["dict[str, Any]"]
    return ["Any"]


def _resolve_local_ref(ref: str, definitions: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            target = definitions.get(ref.removeprefix(prefix))
            return target if isinstance(target, Mapping) else None
    return None


def _literal(values: list[Any]) -> str:
    if not all(value is None or isinstance(value, str | int | bool) for value in values):
        return "Any"
    return f"Literal[{', '.join(json.dumps(value) if isinstance(value, str) else repr(value) for value in values)}]"


def _literal_members(values: list[Any]) -> list[str]:
    if not all(value is None or isinstance(value, str | int | bool) for value in values):
        return ["Any"]
    return [json.dumps(value) if isinstance(value, str) else repr(value) for value in values]


def _union(annotations: list[str]) -> str:
    unique = list(dict.fromkeys(annotations))
    if "Any" in unique:
        return "Any"
    return " | ".join(unique)


def _identifier(name: str) -> str:
    identifier = re.sub(r"\W", "_", name, flags=re.ASCII) or "_"
    if identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def _parameter_identifier(name: str) -> str:
    identifier = _identifier(name)
    return f"{identifier}_" if keyword.iskeyword(identifier) else identifier


def _global_identifier(name: str) -> str:
    identifier = _parameter_identifier(name)
    return f"{identifier}_tool" if identifier in _RESERVED_GLOBALS else identifier


def _unique(name: str, existing: Mapping[str, object]) -> str:
    candidate = name
    suffix = 2
    while candidate in existing:
        candidate = f"{name}_{suffix}"
        suffix += 1
    return candidate


def _one_line(text: str) -> str:
    return " ".join(text.split())
