"""Harness adapter for Codex's standalone search (no shell dependency)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastmcp.tools import FunctionTool, ToolResult
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from fast_agent.core.exceptions import ModelConfigError, ProviderKeyError
from fast_agent.tools.web_search import SearchCommands, WebSearchError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from fast_agent.agents.mcp_agent import McpAgent
    from fast_agent.llm.request_params import RequestParams
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.session.session_manager import Session


WEB_RUN_TOOL_NAME = "web_run"
SEARCH_SESSION_META = "fast-agent/web-search-session"


def _validation_feedback(exc: ValidationError) -> str:
    # Locations can contain arbitrary extra-field names. Only emit schema-owned names.
    schema = SearchCommands.model_json_schema()
    fields = set(schema["properties"])
    for definition in schema.get("$defs", {}).values():
        fields.update(definition.get("properties", {}))
    details = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False)[:8]:
        location = ".".join(
            str(part) if isinstance(part, int) else part if part in fields else "<field>"
            for part in error["loc"][:6]
        )
        details.append(f"{location or '<root>'}: {error['type']}")
    return ("Invalid web search arguments: " + "; ".join(details))[:1200]


def search_session_id(history: Sequence[PromptMessageExtended], session_id: str | None) -> str:
    for message in history:
        for result in (message.tool_results or {}).values():
            value = (result.meta or {}).get(SEARCH_SESSION_META)
            if isinstance(value, str) and value:
                return value
    if session_id:
        return f"{session_id}:web"
    for message in history:
        for call_id, call in (message.tool_calls or {}).items():
            if call.params.name == WEB_RUN_TOOL_NAME:
                return str(uuid5(NAMESPACE_URL, f"fast-agent:web:{call_id}"))
    return str(uuid4())


class CodexWebSearchAdapter:
    def __init__(self, agent: McpAgent) -> None:
        self.agent = agent
        self.tool: FunctionTool | None = None
        self.model: ContextVar[str | None] = ContextVar("codex_search_model", default=None)
        self.detached = False
        self.identity: str | None = None
        self.scope: str | None = None

    @contextmanager
    def turn(self, params: RequestParams | None) -> Iterator[None]:
        llm = self.agent.llm
        model = llm.get_request_params(params).model if llm and params else self.model.get()
        token = self.model.set(model)
        try:
            yield
        finally:
            self.model.reset(token)

    def _session(self) -> Session | None:
        if self.detached:
            return None
        context = self.agent.context
        if context is None or context.session_manager is None:
            return None
        manager = context.session_manager
        if context.acp is None:
            return manager.current_session
        # ACP agents share a manager: never write another thread's current session.
        name = manager.resolve_session_name(context.acp.session_id)
        if name is None:
            return None
        current = manager.current_session
        return current if current and current.info.name == name else manager.get_session(name)

    def _scope_id(self) -> str | None:
        return None if self.detached else self.agent._current_shell_session_id()

    @property
    def metadata_key(self) -> str:
        return f"{SEARCH_SESSION_META}/{self.agent.name}"

    def history_loaded(self, messages: list[PromptMessageExtended] | None) -> None:
        from fast_agent.history.compaction import is_compaction_message

        if not messages:
            self.identity = str(uuid4())
            self.scope = self._scope_id()
            self._persist_identity()
        else:
            # Compaction may remove all search results; retain the cached identity.
            for message in messages:
                for result in (message.tool_results or {}).values():
                    value = (result.meta or {}).get(SEARCH_SESSION_META)
                    if isinstance(value, str) and value:
                        self.identity = value
                        self.scope = self._scope_id()
                        self._persist_identity()
                        return
            if not any(is_compaction_message(message) for message in messages):
                self.identity = None

    def session_id(self) -> str:
        scope = self._scope_id()
        switched = scope != self.scope
        if switched:
            self.identity = None
            self.scope = scope
        session = self._session()
        metadata = session.info.metadata if session else None
        saved = metadata.get(self.metadata_key) if metadata is not None else None
        if isinstance(saved, str) and saved:
            self.identity = saved
        if self.identity is None:
            self.identity = search_session_id(
                [] if switched else self.agent.message_history,
                f"{scope}:{self.agent.name}" if scope else None,
            )
        self._persist_identity()
        return self.identity

    def _persist_identity(self) -> None:
        session = self._session()
        if session is not None and session.info.metadata.get(self.metadata_key) != self.identity:
            session.info.metadata[self.metadata_key] = self.identity
            session._save_metadata()

    def sync(self) -> None:
        from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
        from fast_agent.tools.web_search import WEB_SEARCH_DESCRIPTION, commands_schema

        enabled = isinstance(
            self.agent.llm, CodexResponsesLLM
        ) and self.agent.llm.standalone_web_search_enabled(self.model.get())
        if enabled and self.tool is None:
            # Never replace a user-supplied function with the built-in.
            if WEB_RUN_TOOL_NAME in self.agent._execution_tools:
                return
            self.tool = FunctionTool(
                name=WEB_RUN_TOOL_NAME,
                description=WEB_SEARCH_DESCRIPTION,
                parameters=commands_schema(),
                meta={"fast_agent": {"inherit_to_clone": False}},
                fn=self.run,
            )
            self.agent.add_tool(self.tool)
        elif not enabled and self.tool is not None:
            if self.agent._execution_tools.get(WEB_RUN_TOOL_NAME) is self.tool:
                self.agent.remove_tool(WEB_RUN_TOOL_NAME)
            self.tool = None

    async def permission_error(
        self, name: str, arguments: dict[str, Any] | None, tool_use_id: str | None
    ) -> CallToolResult | None:
        if (
            name != WEB_RUN_TOOL_NAME
            or self.tool is None
            or self.agent._execution_tools.get(name) is not self.tool
        ):
            return None
        try:
            permission = await self.agent.aggregator.permission_handler.check_permission(
                tool_name=WEB_RUN_TOOL_NAME,
                server_name="web",
                arguments=arguments,
                tool_use_id=tool_use_id,
            )
        except Exception:
            # A failed permission service must never permit outbound browsing.
            message = "Web search permission check failed."
        else:
            if permission.allowed:
                return None
            message = (
                "Web search permission request cancelled."
                if permission.is_cancelled
                else "Web search permission denied."
            )
        return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)

    async def run(self, **arguments: Any) -> ToolResult:
        from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM

        llm = self.agent.llm
        if not isinstance(llm, CodexResponsesLLM) or not llm.standalone_web_search_enabled(
            self.model.get()
        ):
            return ToolResult(
                content=[TextContent(type="text", text="Web search is disabled.")], is_error=True
            )
        identity = self.session_id()
        try:
            response = await llm.run_standalone_web_search(
                identity, arguments, model=self.model.get()
            )
        except ValidationError as exc:
            message = _validation_feedback(exc)
        except WebSearchError as exc:
            message = f"Web search failed ({exc.kind}"
            if exc.status_code is not None:
                message += f", HTTP {exc.status_code}"
            message += "). Retry or check Codex authentication/configuration."
        except (ProviderKeyError, ModelConfigError, ValueError, OSError):
            # Configuration/authentication messages may include secrets or URLs.
            message = "Web search failed (authentication/configuration). Check Codex settings."
        else:
            return ToolResult(
                content=[TextContent(type="text", text=response.output)],
                meta={SEARCH_SESSION_META: identity, "results": response.results},
            )
        return ToolResult(
            content=[TextContent(type="text", text=message)],
            is_error=True,
            meta={SEARCH_SESSION_META: identity},
        )
