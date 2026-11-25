"""Render tool definitions as Python function stubs for the programmatic tool sandbox.

Every tool becomes a keyword-only ``async def`` whose parameters come from the tool's JSON
input schema. The same stubs are shown to the model and used by Monty's type checker, so
code that calls a tool incorrectly is rejected before any tool runs. The one difference is
``UNSTRUCTURED_RETURNS``: the checker sees ``Any``, because Monty rejects indexing a
``str | Any`` result as a dict even though parsed JSON text arrives as one.

Closed object parameters differ too: the model sees their fields inline (``{gte?: str}``), the
checker sees a private ``TypedDict``, so a dict literal with an unknown key is rejected.
"""

from __future__ import annotations

import builtins
import json
import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from fast_agent.ptc.catalog import ToolRoute
from fast_agent.ptc.store import STORE_FUNCTION_NAMES, STORE_STUBS
from fast_agent.ptc.vocabulary import function_order, is_builtin_tool

if TYPE_CHECKING:
    from mcp_types import Tool

    from fast_agent.ptc.catalog import PtcNameLedger

# Modules importable inside Monty; a tool must not shadow them.
SANDBOX_MODULES = frozenset(
    {
        "annotationlib",
        "asyncio",
        "base64",
        "binascii",
        "bisect",
        "collections",
        "colorsys",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "fnmatch",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "keyword",
        "math",
        "os",
        "pathlib",
        "posixpath",
        "random",
        "re",
        "stat",
        "statistics",
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
UNSTRUCTURED_RETURNS = "str | Any"
"""Return annotation of a tool without an output schema: text, or JSON text parsed into a
dict/list, or media handles."""
_MAX_ANNOTATION_DEPTH = 4
# A safety net for pathological output schemas, not a prompt budget: real catalogues render
# well below it (the Hugging Face PTC tools peak at about 2.5k characters), and the field notes
# it would trim are what tell a model about caps, units and when fields are present.
_MAX_RETURNS_CHARS = 64 * 1024
type _Notes = Literal["all", "numeric", "top"]
"""Which field descriptions a return shape keeps as comments: all; top-level ones and nested
ones with a number (usually a limit, such as "first 100"); or top-level ones only."""
# Ways to fit a return shape in _MAX_RETURNS_CHARS, tried in order: (comments kept, depth from
# which objects collapse to ``{...}``).
_RETURNS_SHAPE_LEVELS: tuple[tuple[_Notes, int | None], ...] = (
    ("all", None),
    ("numeric", None),
    ("top", None),
    *(("top", depth) for depth in range(_MAX_ANNOTATION_DEPTH, 0, -1)),
)
_FASTMCP_WRAP_RESULT = "x-fastmcp-wrap-result"


@dataclass(frozen=True, slots=True)
class PtcParameter:
    python_name: str
    schema_name: str
    annotation: str
    """The type checker's annotation."""
    required: bool
    description: str | None
    display: str | None = None
    """The model-facing annotation, where it differs from ``annotation``."""


@dataclass(frozen=True, slots=True)
class PtcTypedDict:
    """A closed object parameter (or part of one), typed for the checker as a ``TypedDict``."""

    name: str
    path: str
    """Where the object sits in the arguments, e.g. ``createdAt`` or ``filters[].range``."""
    source: str
    shape: str
    """Model-facing rendering of its fields, e.g. ``{gte?: str, lte?: str}``."""


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
    builtin: bool = False
    unwrap_result: bool = False
    """FastMCP wrapped a non-object return as structured ``{"result": value}``."""
    typed_dicts: tuple[PtcTypedDict, ...] = ()
    """Definitions the type checker's stub needs, inner objects first."""

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

    def signature(self, *, type_check: bool = False) -> str:
        """Return the one-line call signature, e.g. ``name(*, query: str) -> Any``.

        ``type_check`` renders the variant Monty's type checker sees (see the module docstring).
        """
        parameters = ", ".join(
            _render_parameter(parameter, type_check=type_check) for parameter in self._ordered()
        )
        if self.accepts_extra:
            extra_name = "kwargs"
            parameter_names = {parameter.python_name for parameter in self.parameters}
            while extra_name in parameter_names:
                extra_name += "_"
            extra = f"**{extra_name}: Any"
            parameters = f"{parameters}, {extra}" if parameters else extra
        if parameters and not parameters.startswith("**"):
            parameters = f"*, {parameters}"
        returns = "Any" if type_check and self.returns == UNSTRUCTURED_RETURNS else self.returns
        return f"{self.python_name}({parameters}) -> {returns}"

    def example_call(self) -> str:
        """Return an awaited call with required arguments, e.g. ``await bash(command=...)``."""
        arguments = ", ".join(
            f"{parameter.python_name}=..." for parameter in self._ordered() if parameter.required
        )
        return f"await {self.python_name}({arguments})"

    def stub(self, *, type_check: bool = False) -> str:
        lines = [f"async def {self.signature(type_check=type_check)}:"]
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
        if self.tool_name != self.python_name and not self.builtin:
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
    ``exec_sandbox`` instead; ``"both"`` for a sandbox function that is also directly
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


def build_functions(
    tools: Sequence[Tool],
    *,
    reserved_names: frozenset[str] = frozenset(),
    profile_names: bool = False,
    ledger: PtcNameLedger | None = None,
    routes: Mapping[str, ToolRoute] | None = None,
) -> dict[str, PtcFunction]:
    """Build sandbox functions keyed by Python name, resolving name collisions.

    ``reserved_names`` are other sandbox globals that tool functions must not shadow.
    ``ledger`` keeps names assigned by earlier builds of the conversation.
    """
    functions: dict[str, PtcFunction] = {}
    typed_dict_names: set[str] = set()
    # ptc-vocab-v0: preserve the parent rendering exactly in the comparison arm.
    ordered = tools if profile_names else sorted(tools, key=function_order)
    for tool in ordered:
        candidate = _global_identifier(tool.name, reserved_names)
        if ledger is None:
            python_name = _unique(candidate, functions)
        else:
            route = routes.get(tool.name) if routes else None
            if route is None:
                route = ToolRoute.from_tool(tool)
            python_name = ledger.allocate(route, candidate, functions, reserved_names)
        function = _build_function(tool, python_name, typed_dict_names)
        if profile_names:
            returns = "Any" if function.returns == UNSTRUCTURED_RETURNS else function.returns
            function = replace(function, builtin=False, returns=returns)
        functions[python_name] = function
    renamed = {
        f.tool_name: f.python_name
        for f in functions.values()
        if f.builtin and f.tool_name != f.python_name
    }
    for name, function in functions.items():
        if function.builtin:
            description = function.description
            for old, new in renamed.items():
                description = re.sub(rf"\b{re.escape(old)}\b", new, description)
            functions[name] = replace(function, description=description)
    return functions


def render_stubs(functions: Mapping[str, PtcFunction]) -> str:
    """Render the type checker's stubs for ``functions`` and the runtime's helpers."""
    body = "\n\n\n".join(
        "\n".join(
            [*(typed.source for typed in function.typed_dicts), function.stub(type_check=True)]
        )
        for function in functions.values()
    )
    return f"from typing import Any, Literal, Required, TypedDict\n\n{STORE_STUBS}\n\n{body}\n"


def _build_function(tool: Tool, python_name: str, typed_dict_names: set[str]) -> PtcFunction:
    options = tool_ptc_options(tool)
    ptc_schema = options.get("input_schema")
    schema = ptc_schema if isinstance(ptc_schema, Mapping) else tool.input_schema
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    required = schema.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    definitions = _definitions(schema)
    typed_dicts = _TypedDicts(f"_{python_name}__", typed_dict_names)
    checked = _Renderer(definitions, "check", typed_dicts=typed_dicts)
    shown = _Renderer(definitions, "display")

    parameters: list[PtcParameter] = []
    used_names: set[str] = set()
    for schema_name, property_schema in properties.items():
        parameter_name = _parameter_identifier(str(schema_name))
        while parameter_name in used_names:
            parameter_name = f"{parameter_name}_"
        used_names.add(parameter_name)
        is_required = schema_name in required_names
        # Optional parameters keep their exact schema type; only nullable ones accept None.
        annotation = checked.annotation(property_schema, path=parameter_name)
        display = shown.annotation(property_schema)
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
                display=display if display != annotation else None,
            )
        )

    # Only an explicitly open schema gets **kwargs. JSON Schema allows extra properties when
    # the key is absent, but treating that as closed keeps typo detection in the type check.
    additional = schema.get("additionalProperties")
    accepts_extra = additional is True or isinstance(additional, Mapping)
    returns_schema = options.get("returns")
    # A declared program value is returned as is; MCP structured content may be wrapped.
    program_schema = returns_schema if isinstance(returns_schema, Mapping) else None
    structured = program_schema is None and tool.output_schema is not None
    output_schema = tool.output_schema if structured else program_schema
    output_definitions = _definitions(output_schema or {})
    wrapped = _wrapped_result_schema(output_schema) if structured else None
    if wrapped is not None:
        output_schema = wrapped
    if output_schema is None:
        returns = UNSTRUCTURED_RETURNS
    else:
        members = _Renderer(output_definitions, "check").members(output_schema, 0)
        if structured:
            # Structured content with media arrives as {"structured_content", "content"}, and a
            # peer's wrapped result may lack "result": both are dicts whatever the schema says.
            members.append("dict[str, Any]")
        returns = _union(members)
    returns_shape = _returns_shape(output_schema, output_definitions)
    description = options.get("description")
    return PtcFunction(
        python_name=python_name,
        tool_name=tool.name,
        builtin=is_builtin_tool(tool),
        description=description if isinstance(description, str) else tool.description or "",
        parameters=tuple(parameters),
        accepts_extra=accepts_extra,
        returns=returns,
        returns_shape=returns_shape if returns_shape != returns else None,
        unwrap_result=wrapped is not None,
        typed_dicts=tuple(typed_dicts.collected),
    )


def _wrapped_result_schema(schema: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The ``result`` schema of a FastMCP-wrapped non-object return, else None."""
    if not schema or schema.get(_FASTMCP_WRAP_RESULT) is not True:
        return None
    properties = schema.get("properties")
    result = properties.get("result") if isinstance(properties, Mapping) else None
    return result if isinstance(result, Mapping) else None


def _definitions(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    definitions: dict[str, Any] = {}
    for key in ("definitions", "$defs"):
        value = schema.get(key)
        if isinstance(value, Mapping):
            definitions.update(value)
    return definitions


def _returns_shape(schema: Mapping[str, Any] | None, definitions: Mapping[str, Any]) -> str | None:
    """Render an output schema as a compact TypedDict-like shape within ``_MAX_RETURNS_CHARS``.

    Field descriptions become trailing comments. Over budget, nested comments go first (those
    with a number last), then the deepest objects collapse to ``{...}``, then trailing fields
    are elided.
    """
    if not schema:
        return None
    rendered = ""
    for notes, collapse_depth in _RETURNS_SHAPE_LEVELS:
        renderer = _Renderer(definitions, "shape", notes=notes, collapse_depth=collapse_depth)
        rendered = renderer.annotation(schema)
        if len(rendered) <= _MAX_RETURNS_CHARS:
            return rendered
    return _elide(rendered)


def _elide(text: str) -> str:
    """Cut ``text`` at the last field boundary within ``_MAX_RETURNS_CHARS``."""
    kept = text[: _MAX_RETURNS_CHARS - 2]
    boundary = max(kept.rfind(", "), kept.rfind(",\n"))
    return f"{kept[: boundary + 1] if boundary > 0 else kept} …"


@dataclass(slots=True)
class _TypedDicts:
    """Names and collects the TypedDicts of one function's closed object parameters."""

    prefix: str
    taken: set[str]
    """Names already used by any function of the same stubs."""
    collected: list[PtcTypedDict] = field(default_factory=list)

    def add(self, path: str, fields: Sequence[tuple[str, str, bool]], shape: str) -> str:
        """Define a TypedDict with ``(key, annotation, required)`` fields; return its name."""
        base = f"{self.prefix}{re.sub(r'\W+', '_', path, flags=re.ASCII).strip('_')}"
        name = base
        suffix = 2
        while name in self.taken:
            name = f"{base}_{suffix}"
            suffix += 1
        self.taken.add(name)
        total = all(required for _, _, required in fields)
        items = ", ".join(
            f"{json.dumps(key)}: {annotation if total or not required else f'Required[{annotation}]'}"
            for key, annotation, required in fields
        )
        source = f'{name} = TypedDict("{name}", {{{items}}}{"" if total else ", total=False"})'
        self.collected.append(PtcTypedDict(name=name, path=path, source=source, shape=shape))
        return name


type _RenderMode = Literal["check", "display", "shape"]


@dataclass(frozen=True, slots=True)
class _Renderer:
    """Render JSON schemas as Python annotations or as documentation.

    ``check``: annotations for the type checker; closed objects become TypedDicts when
    ``typed_dicts`` collects them, otherwise ``dict[str, Any]``. ``display``: the model-facing
    variant, with closed objects written inline as ``{key?: type}``. ``shape``: documentation
    of return values; every object shows its fields and enums are written ``"a" | "b"``.
    """

    definitions: Mapping[str, Any]
    mode: _RenderMode
    typed_dicts: _TypedDicts | None = None
    notes: _Notes = "all"
    """Field descriptions kept as comments (``shape``)."""
    collapse_depth: int | None = None
    """Objects at or below this depth render as ``{...}`` (``shape``)."""

    def annotation(self, schema: Any, depth: int = 0, path: str = "") -> str:
        return _union(self.members(schema, depth, path))

    def members(self, schema: Any, depth: int, path: str = "") -> list[str]:
        """Return flattened union members so nested unions never repeat a member."""
        if not isinstance(schema, Mapping) or depth > _MAX_ANNOTATION_DEPTH:
            return ["Any"]
        ref = schema.get("$ref")
        if isinstance(ref, str):
            target = _resolve_local_ref(ref, self.definitions)
            if target is None:
                return ["Any"]
            siblings = {key: value for key, value in schema.items() if key != "$ref"}
            return self.members({**target, **siblings}, depth + 1, path)
        values = [schema["const"]] if "const" in schema else schema.get("enum")
        if isinstance(values, list) and values:
            return _literal_members(values) if self.mode == "shape" else [_literal(values)]
        # Depth counts object nesting and references, which bound recursion and size; unions,
        # type lists and arrays only wrap this schema's own (finite) branches.
        for key in ("anyOf", "oneOf"):
            options = schema.get(key)
            if isinstance(options, list) and options:
                return [
                    member for option in options for member in self.members(option, depth, path)
                ]

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            return [
                member
                for item in schema_type
                for member in self.members({**schema, "type": item}, depth, path)
            ]
        if isinstance(schema_type, str) and schema_type in _JSON_SCALAR_TYPES:
            return [_JSON_SCALAR_TYPES[schema_type]]
        if schema_type == "array":
            return [f"list[{self.annotation(schema.get('items'), depth, f'{path}[]')}]"]
        properties = schema.get("properties")
        if schema_type in ("object", None) and isinstance(properties, Mapping):
            if self.mode == "shape":
                return [self._object(schema, properties, depth)]
            # Like tool arguments, an object without additionalProperties is treated as closed
            # so that misspelled keys are caught.
            additional = schema.get("additionalProperties")
            if properties and additional in (None, False):
                if self.mode == "display":
                    return [self._object(schema, properties, depth)]
                if self.typed_dicts is not None:
                    return [self._typed_dict(self.typed_dicts, schema, properties, depth, path)]
        if schema_type == "object":
            additional = schema.get("additionalProperties")
            if self.mode == "shape" and isinstance(additional, Mapping):
                return [f"dict[str, {self.annotation(additional, depth + 1)}]"]
            return ["dict[str, Any]"]
        return ["Any"]

    def _fields(
        self, schema: Mapping[str, Any], properties: Mapping[str, Any], depth: int, path: str
    ) -> list[tuple[str, str, bool, str]]:
        """Return ``(key, annotation, required, description)`` for each property."""
        required = schema.get("required")
        required_names = set(required) if isinstance(required, list) else set()
        fields: list[tuple[str, str, bool, str]] = []
        for name, property_schema in properties.items():
            key = str(name)
            annotation = self.annotation(property_schema, depth + 1, f"{path}.{key}")
            description = (
                property_schema.get("description") if isinstance(property_schema, Mapping) else None
            )
            shown = self.mode == "shape" and isinstance(description, str)
            note = _one_line(description) if shown else ""
            fields.append((key, annotation, key in required_names, note))
        return fields

    def _typed_dict(
        self,
        typed_dicts: _TypedDicts,
        schema: Mapping[str, Any],
        properties: Mapping[str, Any],
        depth: int,
        path: str,
    ) -> str:
        fields = self._fields(schema, properties, depth, path)
        shape = _Renderer(self.definitions, "display").annotation(schema, depth)
        return typed_dicts.add(
            path, [(key, annotation, required) for key, annotation, required, _ in fields], shape
        )

    def _object(self, schema: Mapping[str, Any], properties: Mapping[str, Any], depth: int) -> str:
        """Render ``{key: type, optional?: type}``, on one line unless it has comments.

        With comments, each top-level field, commented field or multi-line field gets its own
        line; runs of other nested fields share one.
        """
        if self.collapse_depth is not None and 0 < self.collapse_depth <= depth:
            return "{...}"
        rendered: list[tuple[str, str]] = []
        for key, annotation, required, note in self._fields(schema, properties, depth, ""):
            name = key if key.isidentifier() else json.dumps(key)
            kept = (
                self.notes == "all"
                or depth == 0
                or (self.notes == "numeric" and any(char.isdigit() for char in note))
            )
            rendered.append(
                (f"{name}{'' if required else '?'}: {annotation}", note if kept else "")
            )
        if not any(note or "\n" in text for text, note in rendered):
            return "{" + ", ".join(text for text, _ in rendered) + "}"
        lines: list[str] = []
        group: list[str] = []
        for text, note in rendered:
            if depth and not note and "\n" not in text:
                group.append(text)
                continue
            if group:
                lines.append(", ".join(group) + ",")
                group = []
            first, newline, rest = text.partition("\n")
            comment = f"  # {note}" if note else ""
            lines.append(f"{first}{comment}{newline}{rest}," if newline else f"{text},{comment}")
        if group:
            lines.append(", ".join(group) + ",")
        body = "\n".join(f"    {line}" for block in lines for line in block.split("\n"))
        return "{\n" + body + "\n}"


def _docstring_literal(text: str) -> str:
    """Escape ``text`` for the body of a triple-double-quoted string literal."""
    escaped = "".join(
        char if char == "\n" or char.isprintable() else char.encode("unicode_escape").decode()
        for char in text.replace("\\", "\\\\")
    )
    escaped = escaped.replace('"""', '\\"\\"\\"')
    # A trailing quote would merge with the closing delimiter.
    return escaped[:-1] + '\\"' if escaped.endswith('"') else escaped


def _render_parameter(parameter: PtcParameter, *, type_check: bool) -> str:
    annotation = parameter.annotation if type_check else parameter.display or parameter.annotation
    rendered = f"{parameter.python_name}: {annotation}"
    # `= ...` marks "has a default" in a stub: omitting it lets the tool apply its own default.
    return rendered if parameter.required else f"{rendered} = ..."


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


def _global_identifier(name: str, reserved_names: frozenset[str] = frozenset()) -> str:
    identifier = _parameter_identifier(name)
    # Monty star-imports the type-check stubs, which skips private (``_``-prefixed) names,
    # so a tool function must have a public name to resolve during type checking.
    if identifier.startswith("_"):
        identifier = f"tool{identifier}"
    reserved = identifier in _RESERVED_GLOBALS or identifier in reserved_names
    return f"{identifier}_tool" if reserved else identifier


def _unique(name: str, existing: Mapping[str, object]) -> str:
    candidate = name
    suffix = 2
    while candidate in existing:
        candidate = f"{name}_{suffix}"
        suffix += 1
    return candidate


def _one_line(text: str) -> str:
    return " ".join(text.split())
