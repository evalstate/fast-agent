import asyncio

from fast_agent.core.instruction_refresh import rebuild_agent_instruction


class StubAgent:
    def __init__(self) -> None:
        self.skill_registry = None
        self.manifests = None
        self.context = None
        self.instruction_context = None
        self.rebuild_calls = 0

    def set_skill_manifests(self, manifests) -> None:
        self.manifests = list(manifests)

    def set_instruction_context(self, context) -> None:
        self.instruction_context = dict(context)

    async def rebuild_instruction_templates(self) -> None:
        await asyncio.sleep(0)
        self.rebuild_calls += 1


def test_rebuild_agent_instruction_updates_fields() -> None:
    agent = StubAgent()
    result = asyncio.run(
        rebuild_agent_instruction(
            agent,
            skill_manifests=[object()],
            instruction_context={"agentSkills": "skills"},
            skill_registry="registry",
        )
    )
    assert agent.manifests is not None
    assert agent.instruction_context == {"agentSkills": "skills"}
    assert agent.skill_registry == "registry"
    assert agent.rebuild_calls == 1
    assert result.updated_skill_manifests is True
    assert result.updated_instruction_context is True
    assert result.updated_skill_registry is True
    assert result.rebuilt_instruction is True


def test_rebuild_agent_instruction_handles_missing_methods() -> None:
    class MinimalAgent:
        pass

    agent = MinimalAgent()
    result = asyncio.run(rebuild_agent_instruction(agent))
    assert result.updated_skill_manifests is False
    assert result.updated_instruction_context is False
    assert result.updated_skill_registry is False
    assert result.rebuilt_instruction is False
