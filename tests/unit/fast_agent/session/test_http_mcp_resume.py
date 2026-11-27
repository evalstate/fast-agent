from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import MCPServerSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.session import (
    SessionAgentSnapshot,
    SessionAttachmentRef,
    SessionContinuationSnapshot,
    SessionHttpMcpServerSnapshot,
    SessionHydrator,
    SessionManager,
    SessionSaveIdentity,
    SessionSnapshot,
    capture_session_snapshot,
    load_session_snapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol


class _ProviderStub:
    config_name = "anthropic"


class _FakeMcpAgent:
    def __init__(
        self,
        *,
        name: str = "worker",
        attached_servers: list[str] | None = None,
        server_configs: dict[str, MCPServerSettings] | None = None,
        registered_servers: dict[str, MCPServerSettings] | None = None,
        with_registry_context: bool = True,
    ) -> None:
        self.name = name
        self.instruction = "Base instruction"
        self.config = AgentConfig(name, instruction=self.instruction, model="passthrough")
        self.config.default_request_params = RequestParams(
            use_history=True,
            system_prompt=self.instruction,
        )
        self.llm = SimpleNamespace(
            default_request_params=self.config.default_request_params.model_copy(deep=True),
            model_name="passthrough",
            provider=_ProviderStub(),
            resolved_model=SimpleNamespace(overlay=None),
            web_search_supported=False,
            web_search_enabled=False,
            web_fetch_supported=False,
            web_fetch_enabled=False,
            x_search_supported=False,
            x_search_enabled=False,
        )
        self.usage_accumulator = None
        self.message_history: list[object] = []
        self._attached_servers = list(attached_servers or [])
        self._server_configs = dict(server_configs or {})
        self.attach_calls: list[tuple[str, MCPServerSettings | None]] = []

        if with_registry_context:
            registry = ServerRegistry(config=Settings())
            for server_name, cfg in (registered_servers or {}).items():
                registry.registry[server_name] = cfg
            self.context: Context | None = Context(server_registry=registry)
        else:
            self.context = None
        self._registered_names = set(registered_servers or {})

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts
        self.message_history.clear()

    def list_attached_mcp_servers(self) -> list[str]:
        return list(self._attached_servers)

    def list_attached_mcp_server_configs(self) -> dict[str, MCPServerSettings]:
        return dict(self._server_configs)

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: object | None = None,
    ) -> None:
        del options
        self.attach_calls.append((server_name, server_config))
        if server_config is None and server_name not in self._registered_names:
            raise ValueError(f"Server '{server_name}' not found in registry")
        self._attached_servers.append(server_name)


def _write_snapshot(session_dir: Path, snapshot: SessionSnapshot) -> None:
    (session_dir / "session.json").write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def test_capture_persists_only_credential_free_http_and_sse_servers(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    agent = _FakeMcpAgent(
        attached_servers=[
            "jobs",
            "events-sse",
            "with-headers",
            "with-token",
            "with-userinfo",
            "with-user-only",
            "local-stdio",
            "provider-managed",
        ],
        server_configs={
            "jobs": MCPServerSettings(
                name="jobs",
                transport="http",
                url="http://127.0.0.1:8765/mcp",
                protocol_mode="modern",
            ),
            "events-sse": MCPServerSettings(
                name="events-sse",
                transport="sse",
                url="https://mcp.example.com/sse",
                protocol_mode="legacy",
            ),
            "with-headers": MCPServerSettings(
                name="with-headers",
                transport="http",
                url="https://api.example.com/mcp",
                headers={"Authorization": "Bearer super-secret"},
            ),
            "with-token": MCPServerSettings(
                name="with-token",
                transport="http",
                url="https://token.example.com/mcp",
                access_token="secret-token",
            ),
            "with-userinfo": MCPServerSettings(
                name="with-userinfo",
                transport="http",
                url="https://alice:s3cr3t@private.example.com/mcp",
            ),
            "with-user-only": MCPServerSettings(
                name="with-user-only",
                transport="sse",
                url="https://alice@private.example.com/sse",
            ),
            "local-stdio": MCPServerSettings(
                name="local-stdio",
                transport="stdio",
                command="uvx",
                args=["mcp-server-git"],
            ),
            "provider-managed": MCPServerSettings(
                name="provider-managed",
                management="provider",
                transport="http",
                url="https://provider.example.com/mcp",
            ),
        },
    )

    snapshot = capture_session_snapshot(
        session=session,
        active_agent=cast("AgentProtocol", agent),
        agent_registry={"worker": cast("AgentProtocol", agent)},
        identity=SessionSaveIdentity(
            manager=manager,
            session=session,
            created=False,
            acp_session_id=None,
            session_cwd=tmp_path,
            session_store_scope="workspace",
            session_store_cwd=tmp_path,
        ),
    )
    agent_snapshot = snapshot.continuation.agents["worker"]

    assert agent_snapshot.http_mcp_servers == [
        SessionHttpMcpServerSnapshot(
            name="events-sse",
            url="https://mcp.example.com/sse",
            transport="sse",
            protocol_mode="legacy",
        ),
        SessionHttpMcpServerSnapshot(
            name="jobs",
            url="http://127.0.0.1:8765/mcp",
            transport="http",
            protocol_mode="modern",
        ),
    ]


def test_mcp_agent_exposes_attached_server_configs_from_aggregator() -> None:
    registry = ServerRegistry(config=Settings())
    registry.registry["jobs"] = MCPServerSettings(
        name="jobs",
        transport="http",
        url="http://127.0.0.1:8765/mcp",
        protocol_mode="modern",
    )
    agent = McpAgent(
        AgentConfig("worker", skills=[]),
        context=Context(config=Settings(), server_registry=registry),
    )
    agent.aggregator._attached_server_names.append("jobs")
    agent.aggregator._attached_server_names.append("adhoc-sse")
    agent.aggregator._attachment_configs["adhoc-sse"] = MCPServerSettings(
        name="adhoc-sse",
        transport="sse",
        url="https://mcp.example.com/sse",
    )

    configs = agent.list_attached_mcp_server_configs()
    assert set(configs) == {"jobs", "adhoc-sse"}
    assert configs["jobs"].url == "http://127.0.0.1:8765/mcp"
    assert configs["jobs"].protocol_mode == "modern"
    assert configs["adhoc-sse"].transport == "sse"


@pytest.mark.asyncio
async def test_hydrate_restores_missing_http_and_sse_servers_and_prefers_registry(
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="worker",
            agents={
                "worker": SessionAgentSnapshot(
                    attachment_refs=[
                        SessionAttachmentRef(ref="mcp_server:jobs"),
                        SessionAttachmentRef(ref="mcp_server:events-sse"),
                        SessionAttachmentRef(ref="mcp_server:registered-http"),
                    ],
                    http_mcp_servers=[
                        SessionHttpMcpServerSnapshot(
                            name="jobs",
                            url="http://127.0.0.1:8765/mcp",
                            transport="http",
                            protocol_mode="modern",
                        ),
                        SessionHttpMcpServerSnapshot(
                            name="events-sse",
                            url="https://mcp.example.com/sse",
                            transport="sse",
                            protocol_mode="legacy",
                        ),
                        SessionHttpMcpServerSnapshot(
                            name="registered-http",
                            url="https://stale.example.com/mcp",
                            transport="http",
                            protocol_mode="auto",
                        ),
                    ],
                )
            },
        ),
    )
    _write_snapshot(session.directory, snapshot)

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_agent = _FakeMcpAgent(
        registered_servers={
            "registered-http": MCPServerSettings(
                name="registered-http",
                transport="http",
                url="https://fresh-local-registry.example.com/mcp",
            )
        }
    )

    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={"worker": cast("AgentProtocol", runtime_agent)},
        fallback_agent_name="worker",
    )

    assert result.warnings == []
    assert runtime_agent.list_attached_mcp_servers() == [
        "jobs",
        "events-sse",
        "registered-http",
    ]
    assert len(runtime_agent.attach_calls) == 3

    jobs_name, jobs_cfg = runtime_agent.attach_calls[0]
    assert jobs_name == "jobs"
    assert jobs_cfg is not None
    assert jobs_cfg.transport == "http"
    assert jobs_cfg.url == "http://127.0.0.1:8765/mcp"
    assert jobs_cfg.protocol_mode == "modern"

    sse_name, sse_cfg = runtime_agent.attach_calls[1]
    assert sse_name == "events-sse"
    assert sse_cfg is not None
    assert sse_cfg.transport == "sse"
    assert sse_cfg.url == "https://mcp.example.com/sse"
    assert sse_cfg.protocol_mode == "legacy"

    # Registry wins: attached by name with server_config=None.
    reg_name, reg_cfg = runtime_agent.attach_calls[2]
    assert reg_name == "registered-http"
    assert reg_cfg is None


@pytest.mark.asyncio
async def test_hydrate_revalidates_tampered_persisted_http_mcp_server_urls(
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="worker",
            agents={
                "worker": SessionAgentSnapshot(
                    attachment_refs=[
                        SessionAttachmentRef(ref="mcp_server:bad-scheme"),
                        SessionAttachmentRef(ref="mcp_server:bad-userinfo"),
                    ],
                    http_mcp_servers=[
                        SessionHttpMcpServerSnapshot(
                            name="bad-scheme",
                            url="file:///etc/passwd",
                            transport="http",
                        ),
                        SessionHttpMcpServerSnapshot(
                            name="bad-userinfo",
                            url="https://user:pass@example.com/mcp",
                            transport="http",
                        ),
                    ],
                )
            },
        ),
    )
    _write_snapshot(session.directory, snapshot)

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_agent = _FakeMcpAgent()
    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={"worker": cast("AgentProtocol", runtime_agent)},
        fallback_agent_name="worker",
    )

    assert runtime_agent.list_attached_mcp_servers() == []
    assert all(cfg is None for _, cfg in runtime_agent.attach_calls)
    assert [warning.ref for warning in result.warnings] == ["bad-scheme", "bad-userinfo"]
    assert all(warning.code == "attachment-restore-failed" for warning in result.warnings)


def test_load_older_session_snapshot_without_http_mcp_servers_is_backwards_compatible() -> None:
    payload = {
        "schema_version": 8,
        "session_id": "20261126-legacy",
        "created_at": "2026-11-26T10:00:00+00:00",
        "last_activity": "2026-11-26T10:05:00+00:00",
        "continuation": {
            "active_agent": "worker",
            "agents": {
                "worker": {
                    "attachment_refs": [{"ref": "mcp_server:filesystem"}],
                }
            },
        },
    }

    loaded = load_session_snapshot(payload)
    assert loaded.continuation.agents["worker"].http_mcp_servers == []
    assert loaded.continuation.agents["worker"].attachment_refs == [
        SessionAttachmentRef(ref="mcp_server:filesystem")
    ]


@pytest.mark.asyncio
async def test_ambient_hf_space_auth_header_is_persisted_without_secrets_and_reapplied_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_ambient_token")
    space_url = "https://evalstate-mcp-events.hf.space/mcp"

    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    source_agent = _FakeMcpAgent(
        attached_servers=["hf-ambient", "hf-explicit"],
        server_configs={
            "hf-ambient": MCPServerSettings(
                name="hf-ambient",
                transport="http",
                url=space_url,
                headers={"X-HF-Authorization": "Bearer hf_test_ambient_token"},
            ),
            "hf-explicit": MCPServerSettings(
                name="hf-explicit",
                transport="http",
                url=space_url,
                headers={"Authorization": "Bearer explicit-secret"},
            ),
        },
    )

    snapshot = capture_session_snapshot(
        session=session,
        active_agent=cast("AgentProtocol", source_agent),
        agent_registry={"worker": cast("AgentProtocol", source_agent)},
        identity=SessionSaveIdentity(
            manager=manager,
            session=session,
            created=False,
            acp_session_id=None,
            session_cwd=tmp_path,
            session_store_scope="workspace",
            session_store_cwd=tmp_path,
        ),
    )
    persisted = snapshot.continuation.agents["worker"].http_mcp_servers
    assert [entry.name for entry in persisted] == ["hf-ambient"]
    assert persisted[0].url == space_url
    assert "hf_test_ambient_token" not in snapshot.model_dump_json()

    _write_snapshot(session.directory, snapshot)
    runtime_agent = _FakeMcpAgent(registered_servers={})
    result = await SessionHydrator().hydrate_session(
        session=session,
        agents={"worker": cast("AgentProtocol", runtime_agent)},
        fallback_agent_name="worker",
    )

    assert runtime_agent.list_attached_mcp_servers() == ["hf-ambient"]
    _, restored_cfg = runtime_agent.attach_calls[0]
    assert restored_cfg is not None
    assert restored_cfg.url == space_url
    assert restored_cfg.headers == {"X-HF-Authorization": "Bearer hf_test_ambient_token"}
    assert [w.ref for w in result.warnings if w.code == "attachment-restore-failed"] == [
        "hf-explicit"
    ]
