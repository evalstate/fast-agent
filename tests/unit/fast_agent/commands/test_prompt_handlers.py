import pytest
from mcp.types import TextContent
from rich.text import Text

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers.shared import LoadedPromptMessagesResult
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.mcp.mcp_aggregator import SEP
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class StubPromptArgument:
    def __init__(self, name: str, required: bool, description: str | None = None) -> None:
        self.name = name
        self.required = required
        self.description = description


def test_extract_prompt_arguments_normalizes_dict_and_object_arguments() -> None:
    arguments = [
        {"name": "topic", "required": True, "description": "Choose a topic"},
        StubPromptArgument("style", False, "Optional style"),
        {"name": "", "required": True},
    ]

    extracted = prompt_handlers._extract_prompt_arguments(arguments)

    assert extracted.names == ["topic", "style"]
    assert extracted.required == ["topic"]
    assert extracted.optional == ["style"]
    assert extracted.descriptions == {
        "topic": "Choose a topic",
        "style": "Optional style",
    }


def test_extract_prompt_arguments_defaults_missing_required_to_required() -> None:
    extracted = prompt_handlers._extract_prompt_arguments([{"name": "topic"}])

    assert extracted.required == ["topic"]
    assert extracted.optional == []


def test_extract_prompt_arguments_strips_names_and_descriptions() -> None:
    extracted = prompt_handlers._extract_prompt_arguments(
        [
            {"name": " topic ", "required": True, "description": " Choose a topic "},
            {"name": "   ", "required": True, "description": "ignored"},
            StubPromptArgument(" style ", False, " Optional style "),
        ]
    )

    assert extracted.names == ["topic", "style"]
    assert extracted.required == ["topic"]
    assert extracted.optional == ["style"]
    assert extracted.descriptions == {
        "topic": "Choose a topic",
        "style": "Optional style",
    }


def test_extract_prompt_arguments_ignores_unrecognized_argument_shapes() -> None:
    extracted = prompt_handlers._extract_prompt_arguments([object()])

    assert extracted.names == []
    assert extracted.required == []
    assert extracted.optional == []


def test_format_missing_required_prompt_arguments_uses_shared_count_display() -> None:
    assert (
        prompt_handlers._format_missing_required_prompt_arguments(["topic"])
        == "Missing required prompt argument: topic"
    )
    assert (
        prompt_handlers._format_missing_required_prompt_arguments(["topic", "style"])
        == "Missing required prompt arguments: topic, style"
    )


def test_build_prompt_list_text_trims_title_and_description() -> None:
    rendered = prompt_handlers._build_prompt_list_text(
        [
            prompt_handlers._prompt_summary(
                server_name="docs",
                prompt_name="summary",
                title=" Summarize docs ",
                description=" Write a concise summary. ",
                arguments=[],
            ),
            prompt_handlers._prompt_summary(
                server_name="docs",
                prompt_name="blank_title",
                title="   ",
                description="   ",
                arguments=[],
            ),
        ],
        include_usage=False,
    ).plain

    assert "docs•summary Summarize docs" in rendered
    assert "Write a concise summary." in rendered
    assert "docs•blank_title   " not in rendered


def test_add_loaded_prompt_message_formats_singular_count() -> None:
    outcome = CommandOutcome()

    prompt_handlers._add_loaded_prompt_message(
        outcome,
        filename="one.json",
        agent_name="test-agent",
        loaded_count=1,
        buffered_text=None,
    )

    assert [message.plain_text() for message in outcome.messages] == [
        "Loaded 1 message from one.json"
    ]


def test_add_loaded_prompt_message_formats_buffered_singular_count() -> None:
    outcome = CommandOutcome()

    prompt_handlers._add_loaded_prompt_message(
        outcome,
        filename="one.json",
        agent_name="test-agent",
        loaded_count=1,
        buffered_text="hello",
    )

    assert [message.plain_text() for message in outcome.messages] == [
        "Loaded 1 message from one.json. Last user message placed in input buffer."
    ]


def test_selected_prompt_from_selection_validates_selection() -> None:
    prompts = [
        prompt_handlers._prompt_summary(
            server_name="server",
            prompt_name="first",
            title=None,
            description=None,
            arguments=[],
        ),
        prompt_handlers._prompt_summary(
            server_name="server",
            prompt_name="second",
            title=None,
            description=None,
            arguments=[],
        ),
    ]

    assert (
        prompt_handlers._selected_prompt_from_selection(
            prompts,
            " 2 ",
            outcome=CommandOutcome(),
            agent_name="test-agent",
        )
        == prompts[1]
    )

    invalid_outcome = CommandOutcome()
    assert (
        prompt_handlers._selected_prompt_from_selection(
            prompts,
            "later",
            outcome=invalid_outcome,
            agent_name="test-agent",
        )
        is None
    )
    assert invalid_outcome.messages[-1].text == "Invalid input, please enter a number."

    cancelled_outcome = CommandOutcome()
    assert (
        prompt_handlers._selected_prompt_from_selection(
            prompts,
            "",
            outcome=cancelled_outcome,
            agent_name="test-agent",
        )
        is None
    )
    assert cancelled_outcome.messages[-1].text == "Prompt selection cancelled."

    blank_outcome = CommandOutcome()
    assert (
        prompt_handlers._selected_prompt_from_selection(
            prompts,
            "   ",
            outcome=blank_outcome,
            agent_name="test-agent",
        )
        is None
    )
    assert blank_outcome.messages[-1].text == "Prompt selection cancelled."


class StubPrompt:
    def __init__(
        self,
        name: str,
        *,
        title: str | None = None,
        description: str | None = None,
        arguments: list[StubPromptArgument] | None = None,
    ) -> None:
        self.name = name
        self.title = title
        self.description = description
        self.arguments = arguments


class StubPromptList:
    def __init__(self, prompts: list[object]) -> None:
        self.prompts = prompts


class StubPromptResult:
    def __init__(self) -> None:
        self.messages = ["message"]


class StubAgent:
    def __init__(self, prompt_result: StubPromptResult) -> None:
        self._prompt_result = prompt_result
        self.prompt_calls: list[tuple[str, dict[str, str]]] = []
        self.generated_messages: list[object] | None = None

    async def get_prompt(self, namespaced_name: str, arg_values: dict[str, str]) -> StubPromptResult:
        self.prompt_calls.append((namespaced_name, arg_values))
        return self._prompt_result

    async def generate(self, messages, _):
        self.generated_messages = messages


class StubHistoryAgent:
    name = "test-agent"
    usage_accumulator = None

    def __init__(self) -> None:
        self.message_history = []
        self.loaded_messages = None
        self.cleared_prompts = None

    def load_message_history(self, messages):
        self.loaded_messages = messages
        self.message_history = messages or []

    def pop_last_message(self):
        return None

    def clear(self, *, clear_prompts: bool = False) -> None:
        self.cleared_prompts = clear_prompts


class StubAgentProvider:
    def __init__(self, prompts: object, agent: object) -> None:
        self._prompts = prompts
        self._agent_instance = agent

    def _agent(self, name: str) -> object:
        return self._agent_instance

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["test-agent"]

    def registered_agent_names(self):
        return ["test-agent"]

    def registered_agents(self):
        return {"test-agent": self._agent_instance}

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "test-agent"

    async def list_prompts(self, namespace, agent_name=None):
        return self._prompts


class FailingPromptAgentProvider(StubAgentProvider):
    async def list_prompts(self, namespace, agent_name=None):
        raise RuntimeError("prompt listing failed")


class UsageDisplayAgentProvider(StubAgentProvider):
    def __init__(self, prompts: object, agent: object) -> None:
        super().__init__(prompts, agent)
        self.shown_usage_for: list[str] = []

    def _show_turn_usage(self, agent_name: str) -> None:
        self.shown_usage_for.append(agent_name)


class StubCommandIO:
    def __init__(self, arg_values: dict[str, str]) -> None:
        self._arg_values = arg_values
        self.prompted_args: list[tuple[str, str | None, bool]] = []
        self.emitted: list[CommandMessage] = []

    async def emit(self, message: CommandMessage) -> None:
        self.emitted.append(message)

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        return default

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options,
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        return default

    async def prompt_model_selection(
        self,
        *,
        initial_provider: str | None = None,
        default_model: str | None = None,
    ) -> str | None:
        del initial_provider, default_model
        return None

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        self.prompted_args.append((arg_name, description, required))
        return self._arg_values.get(arg_name)

    async def display_history_turn(self, *args, **kwargs):
        return None

    async def display_history_overview(self, *args, **kwargs):
        return None

    async def display_usage_report(self, *args, **kwargs):
        return None

    async def display_system_prompt(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_collect_prompt_argument_values_treats_empty_required_value_as_missing() -> None:
    io = StubCommandIO({"topic": ""})
    ctx = CommandContext(
        agent_provider=StubAgentProvider({}, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=io,
    )

    collected = await prompt_handlers._collect_prompt_argument_values(
        ctx,
        prompt_name="demo",
        required_args=["topic"],
        optional_args=[],
        arg_descriptions={},
        agent_name="test-agent",
        right_info="prompt",
        fail_on_missing_required=True,
    )

    assert collected.values == {}
    assert collected.missing_required == ["topic"]


@pytest.mark.asyncio
async def test_collect_prompt_argument_values_omits_empty_optional_value() -> None:
    io = StubCommandIO({"style": ""})
    ctx = CommandContext(
        agent_provider=StubAgentProvider({}, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=io,
    )

    collected = await prompt_handlers._collect_prompt_argument_values(
        ctx,
        prompt_name="demo",
        required_args=[],
        optional_args=["style"],
        arg_descriptions={},
        agent_name="test-agent",
        right_info="prompt",
        fail_on_missing_required=True,
    )

    assert collected.values == {}
    assert collected.missing_required == []


@pytest.mark.asyncio
async def test_collect_prompt_argument_values_strips_supplied_values() -> None:
    io = StubCommandIO({"topic": "  cats  ", "style": "   "})
    ctx = CommandContext(
        agent_provider=StubAgentProvider({}, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=io,
    )

    collected = await prompt_handlers._collect_prompt_argument_values(
        ctx,
        prompt_name="demo",
        required_args=["topic"],
        optional_args=["style"],
        arg_descriptions={},
        agent_name="test-agent",
        right_info="prompt",
        fail_on_missing_required=True,
    )

    assert collected.values == {"topic": "cats"}
    assert collected.missing_required == []


@pytest.mark.asyncio
async def test_collect_prompt_argument_values_treats_whitespace_required_value_as_missing() -> None:
    io = StubCommandIO({"topic": "   "})
    ctx = CommandContext(
        agent_provider=StubAgentProvider({}, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=io,
    )

    collected = await prompt_handlers._collect_prompt_argument_values(
        ctx,
        prompt_name="demo",
        required_args=["topic"],
        optional_args=[],
        arg_descriptions={},
        agent_name="test-agent",
        right_info="prompt",
        fail_on_missing_required=True,
    )

    assert collected.values == {}
    assert collected.missing_required == ["topic"]


@pytest.mark.asyncio
async def test_handle_select_prompt_prompts_for_required_args(monkeypatch):
    prompt_args = [
        StubPromptArgument("topic", True, "Choose a topic"),
        StubPromptArgument("style", False, "Optional style"),
    ]
    prompt_obj = StubPrompt("demo", description="demo prompt", arguments=prompt_args)
    prompt_result = StubPromptResult()
    agent = StubAgent(prompt_result)
    provider = StubAgentProvider({"server": [prompt_obj]}, agent)
    io = StubCommandIO({"topic": "cats", "style": "haiku"})
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=io,
    )

    monkeypatch.setattr(
        prompt_handlers.PromptMessageExtended,
        "from_get_prompt_result",
        lambda result: ["converted"],
    )
    monkeypatch.setattr(prompt_handlers.progress_display, "resume", lambda: None)
    monkeypatch.setattr(prompt_handlers.progress_display, "pause", lambda: None)

    await prompt_handlers.handle_select_prompt(ctx, agent_name="test-agent", prompt_index=1)

    emitted_text = io.emitted[0].text
    assert isinstance(emitted_text, Text)
    assert (
        emitted_text.plain
        == "Prompt demo requires 1 argument and has 1 optional argument:"
    )
    assert io.prompted_args == [
        ("topic", "Choose a topic", True),
        ("style", "Optional style", False),
    ]
    assert agent.prompt_calls == [
        (f"server{SEP}demo", {"topic": "cats", "style": "haiku"}),
    ]


@pytest.mark.asyncio
async def test_handle_select_prompt_strips_requested_name(monkeypatch):
    prompt_obj = StubPrompt("demo", description="demo prompt")
    prompt_result = StubPromptResult()
    agent = StubAgent(prompt_result)
    provider = StubAgentProvider({"server": [prompt_obj]}, agent)
    io = StubCommandIO({})
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=io,
    )

    monkeypatch.setattr(
        prompt_handlers.PromptMessageExtended,
        "from_get_prompt_result",
        lambda result: ["converted"],
    )
    monkeypatch.setattr(prompt_handlers.progress_display, "resume", lambda: None)
    monkeypatch.setattr(prompt_handlers.progress_display, "pause", lambda: None)

    await prompt_handlers.handle_select_prompt(
        ctx,
        agent_name="test-agent",
        requested_name=" demo ",
    )

    assert agent.prompt_calls == [(f"server{SEP}demo", {})]


@pytest.mark.asyncio
async def test_handle_select_prompt_treats_blank_requested_name_as_missing():
    prompt_obj = StubPrompt("demo", description="demo prompt")
    agent = StubAgent(StubPromptResult())
    provider = StubAgentProvider({"server": [prompt_obj]}, agent)
    io = StubCommandIO({})
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=io,
    )

    outcome = await prompt_handlers.handle_select_prompt(
        ctx,
        agent_name="test-agent",
        requested_name="   ",
    )

    assert agent.prompt_calls == []
    assert outcome.messages[-1].text == "Prompt selection cancelled."


@pytest.mark.asyncio
async def test_handle_select_prompt_shows_usage_when_provider_supports_it(monkeypatch):
    prompt_obj = StubPrompt("demo", description="demo prompt")
    prompt_result = StubPromptResult()
    agent = StubAgent(prompt_result)
    provider = UsageDisplayAgentProvider({"server": [prompt_obj]}, agent)
    io = StubCommandIO({})
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=io,
    )

    monkeypatch.setattr(
        prompt_handlers.PromptMessageExtended,
        "from_get_prompt_result",
        lambda result: ["converted"],
    )
    monkeypatch.setattr(prompt_handlers.progress_display, "resume", lambda: None)
    monkeypatch.setattr(prompt_handlers.progress_display, "pause", lambda: None)

    await prompt_handlers.handle_select_prompt(ctx, agent_name="test-agent", prompt_index=1)

    assert provider.shown_usage_for == ["test-agent"]


@pytest.mark.asyncio
async def test_handle_list_prompts_renders_prompt_arguments() -> None:
    prompts = {
        "server": [
            StubPrompt(
                "one",
                arguments=[StubPromptArgument("topic", True)],
            ),
            {"name": "two", "arguments": [{}, {}]},
        ]
    }
    ctx = CommandContext(
        agent_provider=StubAgentProvider(prompts, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=StubCommandIO({}),
    )

    outcome = await prompt_handlers.handle_list_prompts(ctx, agent_name="test-agent")

    message_text = outcome.messages[0].text
    assert isinstance(message_text, Text)
    rendered = message_text.plain
    assert "server•one" in rendered
    assert "args: topic*" in rendered
    assert "server•two" in rendered
    assert "args: 2 parameters" in rendered


@pytest.mark.asyncio
async def test_handle_list_prompts_accepts_prompt_list_wrapper() -> None:
    prompts = {"server": StubPromptList([StubPrompt("wrapped")])}
    ctx = CommandContext(
        agent_provider=StubAgentProvider(prompts, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=StubCommandIO({}),
    )

    outcome = await prompt_handlers.handle_list_prompts(ctx, agent_name="test-agent")

    message_text = outcome.messages[0].text
    assert isinstance(message_text, Text)
    assert "server•wrapped" in message_text.plain


@pytest.mark.asyncio
async def test_handle_list_prompts_skips_malformed_entries() -> None:
    prompts = {
        "server": [
            object(),
            {"title": "missing name"},
            StubPrompt("valid"),
        ],
    }
    ctx = CommandContext(
        agent_provider=StubAgentProvider(prompts, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=StubCommandIO({}),
    )

    outcome = await prompt_handlers.handle_list_prompts(ctx, agent_name="test-agent")

    message_text = outcome.messages[0].text
    assert isinstance(message_text, Text)
    assert "server•valid" in message_text.plain
    assert "missing name" not in message_text.plain


@pytest.mark.asyncio
async def test_handle_list_prompts_normalizes_prompt_names() -> None:
    prompts = {
        "server": [
            StubPrompt(" object-name "),
            {"name": " dict-name "},
            StubPrompt("   "),
            {"name": "   "},
        ],
    }
    ctx = CommandContext(
        agent_provider=StubAgentProvider(prompts, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=StubCommandIO({}),
    )

    outcome = await prompt_handlers.handle_list_prompts(ctx, agent_name="test-agent")

    message_text = outcome.messages[0].text
    assert isinstance(message_text, Text)
    rendered = message_text.plain
    assert "server•object-name" in rendered
    assert "server•dict-name" in rendered
    assert "server• object-name " not in rendered


@pytest.mark.asyncio
async def test_handle_list_prompts_normalizes_server_names() -> None:
    prompts = {
        " server ": [StubPrompt("valid")],
        "   ": [StubPrompt("blank-server")],
        123: [StubPrompt("numeric-server")],
    }
    ctx = CommandContext(
        agent_provider=StubAgentProvider(prompts, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=StubCommandIO({}),
    )

    outcome = await prompt_handlers.handle_list_prompts(ctx, agent_name="test-agent")

    message_text = outcome.messages[0].text
    assert isinstance(message_text, Text)
    assert "server•valid" in message_text.plain
    assert "blank-server" not in message_text.plain
    assert "numeric-server" not in message_text.plain


@pytest.mark.asyncio
async def test_handle_list_prompts_treats_provider_failure_as_empty() -> None:
    ctx = CommandContext(
        agent_provider=FailingPromptAgentProvider({}, StubAgent(StubPromptResult())),
        current_agent_name="test-agent",
        io=StubCommandIO({}),
    )

    outcome = await prompt_handlers.handle_list_prompts(ctx, agent_name="test-agent")

    assert outcome.messages[0].channel == "warning"
    assert outcome.messages[0].text == "No prompts available for this agent."


def test_local_prompt_template_variables_returns_empty_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_template_error(_filename: str) -> set[str]:
        raise RuntimeError("template parse failed")

    monkeypatch.setattr(
        prompt_handlers,
        "prompt_file_template_variables",
        _raise_template_error,
    )

    assert prompt_handlers._local_prompt_template_variables("prompt.md") == []


def test_split_buffer_prefill_trims_final_user_text() -> None:
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="  Hello  ")],
        )
    ]

    loaded = prompt_handlers._split_buffer_prefill(messages)

    assert loaded.history_messages == []
    assert loaded.buffered_text == "Hello"


def test_split_buffer_prefill_keeps_whitespace_only_user_message_in_history() -> None:
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="   ")],
        )
    ]

    loaded = prompt_handlers._split_buffer_prefill(messages)

    assert loaded.history_messages == messages
    assert loaded.buffered_text is None


@pytest.mark.asyncio
async def test_handle_load_prompt_prompts_for_local_file_template_args(tmp_path):
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Hello {{name}}.", encoding="utf-8")

    agent = StubHistoryAgent()
    provider = StubAgentProvider({}, agent)
    io = StubCommandIO({"name": "Ada"})
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=io,
    )

    outcome = await prompt_handlers.handle_load_prompt(
        ctx,
        agent_name="test-agent",
        filename=str(prompt_path),
    )

    assert io.prompted_args == [("name", "", True)]
    assert outcome.buffer_prefill == "Hello Ada."
    assert agent.loaded_messages == []
    assert agent.cleared_prompts is True


@pytest.mark.asyncio
async def test_handle_load_prompt_returns_loader_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Hello.", encoding="utf-8")

    agent = StubHistoryAgent()
    provider = StubAgentProvider({}, agent)
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=StubCommandIO({}),
    )
    monkeypatch.setattr(
        prompt_handlers,
        "load_prompt_messages_result",
        lambda *_args, **_kwargs: LoadedPromptMessagesResult(
            error="Error loading prompt: invalid template"
        ),
    )

    outcome = await prompt_handlers.handle_load_prompt(
        ctx,
        agent_name="test-agent",
        filename=str(prompt_path),
    )

    assert [message.plain_text() for message in outcome.messages] == [
        "Error loading prompt: invalid template"
    ]
    assert agent.loaded_messages is None


@pytest.mark.asyncio
async def test_handle_load_prompt_reports_single_missing_template_arg(tmp_path):
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Hello {{name}}.", encoding="utf-8")

    agent = StubHistoryAgent()
    provider = StubAgentProvider({}, agent)
    io = StubCommandIO({})
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=io,
    )

    outcome = await prompt_handlers.handle_load_prompt(
        ctx,
        agent_name="test-agent",
        filename=str(prompt_path),
    )

    assert io.prompted_args == [("name", "", True)]
    assert outcome.messages[-1].text == "Missing required prompt argument: name"
    assert agent.loaded_messages is None


@pytest.mark.asyncio
async def test_handle_load_prompt_reports_multiple_missing_template_args(tmp_path):
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Hello {{first}} and {{second}}.", encoding="utf-8")

    agent = StubHistoryAgent()
    provider = StubAgentProvider({}, agent)
    io = StubCommandIO({})
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=io,
    )

    outcome = await prompt_handlers.handle_load_prompt(
        ctx,
        agent_name="test-agent",
        filename=str(prompt_path),
    )

    assert io.prompted_args == [("first", "", True), ("second", "", True)]
    assert outcome.messages[-1].text == "Missing required prompt arguments: first, second"
    assert agent.loaded_messages is None
