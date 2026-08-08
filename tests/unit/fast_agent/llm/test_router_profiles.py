import pytest

from fast_agent.llm.router_profiles import (
    RouterProfileRegistry,
    RouterProfileRule,
    RouterRoute,
)


def test_router_profile_registry_prefers_first_matching_rule() -> None:
    registry = RouterProfileRegistry(
        (
            RouterProfileRule(
                model="org/model",
                backends=frozenset({"special"}),
                profile="special",
            ),
            RouterProfileRule(model="org/model", profile="default"),
        )
    )

    assert registry.resolve(RouterRoute("org/model", "special")) == "special"
    assert registry.resolve(RouterRoute("org/model", "other")) == "default"
    assert registry.resolve(RouterRoute("org/model", None)) == "default"


def test_router_profile_registry_does_not_cross_model_boundaries() -> None:
    registry = RouterProfileRegistry((RouterProfileRule(model="org/model", profile="profile"),))

    assert registry.resolve(RouterRoute("other/model", None)) is None


def test_router_profile_registry_rejects_shadowed_rules() -> None:
    with pytest.raises(ValueError, match="shadowed by an earlier wildcard"):
        RouterProfileRegistry(
            (
                RouterProfileRule(model="org/model", profile="default"),
                RouterProfileRule(
                    model="org/model",
                    backends=frozenset({"special"}),
                    profile="special",
                ),
            )
        )


def test_router_profile_registry_rejects_duplicate_backend_rules() -> None:
    with pytest.raises(ValueError, match="duplicates backends: shared"):
        RouterProfileRegistry(
            (
                RouterProfileRule(
                    model="org/model",
                    backends=frozenset({"first", "shared"}),
                    profile="first",
                ),
                RouterProfileRule(
                    model="org/model",
                    backends=frozenset({"second", "shared"}),
                    profile="second",
                ),
            )
        )
