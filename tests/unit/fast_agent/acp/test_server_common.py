from fast_agent.acp.server.common import coerce_registry_version


def test_coerce_registry_version_requires_nonnegative_non_bool_int() -> None:
    assert coerce_registry_version(0) == 0
    assert coerce_registry_version(3) == 3
    assert coerce_registry_version(True) == 0
    assert coerce_registry_version(False) == 0
    assert coerce_registry_version(-1) == 0
    assert coerce_registry_version("3") == 0
