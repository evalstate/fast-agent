"""Contracts for the lightweight session package and its public exports."""

import subprocess
import sys

import pytest


def test_locking_import_does_not_load_session_implementations() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import fast_agent.session.locking

for module in ("hydrator", "trace_exporter", "session_manager", "snapshot"):
    assert f"fast_agent.session.{module}" not in sys.modules, module
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_exports_support_star_import_and_cache_resolution() -> None:
    import fast_agent.session as session

    namespace: dict[str, object] = {}
    exec("from fast_agent.session import *", namespace)
    assert set(session.__all__) == set(namespace) - {"__builtins__"}
    for name in session.__all__:
        assert namespace[name] is session.__dict__[name]


def test_representative_exports_preserve_identity_and_behavior() -> None:
    from fast_agent.session import (
        SessionHydrator,
        SessionOwner,
        SessionTraceExporter,
        subagent_alias_slug,
    )
    from fast_agent.session.hydrator import SessionHydrator as ConcreteHydrator
    from fast_agent.session.locking import SessionOwner as ConcreteOwner
    from fast_agent.session.trace_exporter import SessionTraceExporter as ConcreteExporter

    assert SessionHydrator is ConcreteHydrator
    assert SessionTraceExporter is ConcreteExporter
    assert SessionOwner is ConcreteOwner
    owner = SessionOwner("host", 123, "start", "acquired", "token")
    assert owner.pid == 123
    assert SessionOwner.from_dict(None) is None
    assert subagent_alias_slug(label="Hello World", task="unused") == "hello_world"


def test_unknown_export_raises_attribute_error() -> None:
    import fast_agent.session as session

    with pytest.raises(AttributeError, match="has no attribute 'unknown_session_export'"):
        session.__getattr__("unknown_session_export")
