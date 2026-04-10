from fast_agent.cli.commands.demo import (
    DemoScenario,
    _build_scenario_markdown,
    _resolve_demo_scenarios,
)


def test_resolve_demo_scenarios_defaults_to_mixed() -> None:
    assert _resolve_demo_scenarios(scenarios=None, cycle=False) == [DemoScenario.mixed]


def test_cycle_includes_fence_focus_first() -> None:
    scenarios = _resolve_demo_scenarios(scenarios=None, cycle=True)

    assert scenarios[0] == DemoScenario.fence_focus


def test_build_fence_focus_scenario_contains_mixed_fence_cases() -> None:
    markdown = _build_scenario_markdown(
        DemoScenario.fence_focus,
        lines=120,
        scale=2,
        seed=0,
    )

    assert "## Scenario: Fence Focus" in markdown
    assert "#### Case 1 — prose before and after a fence" in markdown
    assert "```python" in markdown
    assert "```json" in markdown
    assert "```bash" in markdown
    assert "```apply_patch" in markdown
    assert "#### Case 5 — reference definitions around a fenced block" in markdown
    assert "[render-docs]: https://example.com/rendering \"Renderer notes\"" in markdown
    assert "Trailing prose marker" in markdown
