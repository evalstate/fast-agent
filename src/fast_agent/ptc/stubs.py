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

if TYPE_CHECKING:
    from mcp_types import Tool

# Modules importable inside Monty; a tool must not shadow them.
SANDBOX_MODULES = frozenset(
    {
        "asyncio",
        "base64",
        "binascii",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "functools",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "random",
        "re",
        "sys",
        "time",
        "typing",
        "unicodedata",
    }
)
_RESERVED_GLOBALS = frozenset(dir(builtins)) | SANDBOX_MODULES
_JSON_SCALAR_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
}
_MAX_ANNOTATION_DEPTH = 4
_MAX_OUTPUT_SCHEMA_CHARS = 1200
_SCHEMA_NOISE_KEYS = frozenset({"title", "$schema", "additionalProperties"})


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
    output_schema: str | None = None

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

    def stub(self) -> str:
        lines = [f"async def {self.signature()}:"]
        docstring = self._docstring()
        if docstring:
            lines.append(f"    {docstring!r}")
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
        if self.output_schema:
            parts.append(f"Returns a dict matching this JSON schema: {self.output_schema}")
        return "\n\n".join(parts)


def tool_ptc_options(tool: Tool) -> Mapping[str, Any]:
    """Return the ``meta["fast_agent"]["ptc"]`` options a tool declares for PTC.

    Supported keys: ``expose`` (``"nested"`` default, or ``"direct"`` to keep the tool next to
    ``execute_python``) and ``returns`` (JSON schema of the value sandbox code receives, for
    tools whose program-facing value differs from their model-facing text).
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


def render_stubs(functions: Mapping[str, PtcFunction]) -> str:
    body = "\n\n\n".join(function.stub() for function in functions.values())
    return f"from typing import Any, Literal\n\n\n{body}\n"


def _build_function(tool: Tool, python_name: str) -> PtcFunction:
    schema = tool.input_schema if isinstance(tool.input_schema, Mapping) else {}
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
    returns_schema = tool_ptc_options(tool).get("returns")
    output_schema = returns_schema if isinstance(returns_schema, Mapping) else tool.output_schema
    return PtcFunction(
        python_name=python_name,
        tool_name=tool.name,
        description=tool.description or "",
        parameters=tuple(parameters),
        accepts_extra=accepts_extra,
        returns="dict[str, Any]" if output_schema else "Any",
        output_schema=_compact_schema(output_schema),
    )


def _definitions(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    definitions: dict[str, Any] = {}
    for key in ("definitions", "$defs"):
        value = schema.get(key)
        if isinstance(value, Mapping):
            definitions.update(value)
    return definitions


def _compact_schema(schema: Mapping[str, Any] | None) -> str | None:
    """Render an output schema compactly, dropping annotation-only keys."""
    if not schema:
        return None
    rendered = json.dumps(_strip_schema(schema), separators=(",", ":"), ensure_ascii=False)
    if len(rendered) > _MAX_OUTPUT_SCHEMA_CHARS:
        return f"{rendered[:_MAX_OUTPUT_SCHEMA_CHARS]}…"
    return rendered


def _strip_schema(value: Any, *, property_map: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_schema(item, property_map=key == "properties" and not property_map)
            for key, item in value.items()
            if property_map or (key not in _SCHEMA_NOISE_KEYS and not str(key).startswith("x-"))
        }
    if isinstance(value, list):
        return [_strip_schema(item) for item in value]
    return value


def _render_parameter(parameter: PtcParameter) -> str:
    rendered = f"{parameter.python_name}: {parameter.annotation}"
    # `= ...` marks "has a default" in a stub: omitting it lets the tool apply its own default.
    return rendered if parameter.required else f"{rendered} = ..."


def _annotation(schema: Any, definitions: Mapping[str, Any], depth: int = 0) -> str:
    return _union(_union_members(schema, definitions, depth))


def _union_members(schema: Any, definitions: Mapping[str, Any], depth: int) -> list[str]:
    """Return flattened union members so nested unions never repeat a member."""
    if not isinstance(schema, Mapping) or depth > _MAX_ANNOTATION_DEPTH:
        return ["Any"]
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _resolve_local_ref(ref, definitions)
        if target is None:
            return ["Any"]
        siblings = {key: value for key, value in schema.items() if key != "$ref"}
        return _union_members({**target, **siblings}, definitions, depth + 1)
    if "const" in schema:
        return [_literal([schema["const"]])]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return [_literal(enum)]
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            return [
                member
                for option in options
                for member in _union_members(option, definitions, depth + 1)
            ]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return [
            member
            for item in schema_type
            for member in _union_members({**schema, "type": item}, definitions, depth + 1)
        ]
    if isinstance(schema_type, str) and schema_type in _JSON_SCALAR_TYPES:
        return [_JSON_SCALAR_TYPES[schema_type]]
    if schema_type == "array":
        return [f"list[{_annotation(schema.get('items'), definitions, depth + 1)}]"]
    if schema_type == "object":
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
