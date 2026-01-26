Below is a comprehensive plan for review based on the current code layout and the requirements you listed. I'm
focusing on where the reasoning settings live today, what needs to change, and how to implement the UI gauge +
slash command + model-string parsing cleanly and type‑safely.

-------------------------------------------------------------------------------------------------------------------

                                         IMPORTANT: Terminology

This plan distinguishes between two separate concepts:

 • **Reasoning Format** (`reasoning_format`)
   How reasoning output is structured and parsed by the provider. This is ALREADY implemented in
   `ModelParameters.reasoning` in model_database.py. Values include:
     - `"tags"` — Reasoning in `<thinking>` tags (DeepSeek, Kimi)
     - `"openai"` — OpenAI o-series reasoning format
     - `"anthropic_thinking"` — Anthropic's extended thinking blocks
     - `"reasoning_content"` — Separate `reasoning_content` field (GLM, MiniMax)
     - `"gpt_oss"` — GPT-OSS reasoning format
     - `None` — Model does not emit reasoning output

   **DO NOT CHANGE** the existing `ModelParameters.reasoning` field. It describes output parsing, not input
   configuration.

 • **Reasoning Effort** (`reasoning_effort`)
   How much reasoning the model should perform. This is what needs UNIFICATION. Current state:
     - OpenAI/Responses: `reasoning_effort` field (string: low|medium|high)
     - Bedrock: `reasoning_effort` enum + `REASONING_EFFORT_BUDGETS` mapping
     - Anthropic: `thinking_enabled` (bool) + `thinking_budget_tokens` (int)

   The plan below unifies these into a single `ReasoningEffort` abstraction.

Throughout this document:
 • "format" refers to output parsing (existing, don't touch)
 • "effort" refers to input configuration (the focus of this plan)

-------------------------------------------------------------------------------------------------------------------

                                             Findings (current state)

Model string parsing

 • src/fast_agent/llm/model_factory.py parses provider.model.low via EFFORT_MAP.
 • ReasoningEffort enum is fixed to minimal|low|medium|high.
 • No query parameter support (?reasoning=...).
 • ModelDatabase.normalize_model_name() delegates to ModelFactory.parse_model_string() and currently can't strip
   query params.

Reasoning effort settings (input configuration)

 • OpenAI/Responses/Bedrock expose reasoning_effort settings in src/fast_agent/config.py.
 • Anthropic uses separate thinking_enabled + thinking_budget_tokens (not unified).
 • LLMs store provider-specific fields:
    • OpenAI / Responses: _reasoning_effort (string)
    • Bedrock: _reasoning_effort enum + REASONING_EFFORT_BUDGETS
    • Anthropic: _is_thinking_enabled() + _get_thinking_budget() (config-backed)

Reasoning format (output parsing) — ALREADY IMPLEMENTED

 • ModelParameters.reasoning in model_database.py stores output format.
 • Providers use this to parse reasoning from responses (tags, reasoning_content, etc.).
 • **This is complete and should not be conflated with effort configuration.**

UI

 • There is no reasoning effort indicator in the TUI toolbar yet.
 • The toolbar is constructed in src/fast_agent/ui/enhanced_prompt.py.
 • The gauge prototype lives in examples/reasoning_gauge_test.py (SINGLE style).

Commands

 • No /model slash command exists.
 • Slash commands are implemented in:
    • ACP: src/fast_agent/acp/slash_commands.py
    • TUI: src/fast_agent/ui/enhanced_prompt.py + command_payloads.py
    • Interactive CLI command dispatch: src/fast_agent/ui/interactive_prompt.py

-------------------------------------------------------------------------------------------------------------------

                                               Plan (comprehensive)

1. Introduce unified reasoning effort types

Create a new module: src/fast_agent/llm/reasoning_effort.py

Define typed, provider-agnostic effort configuration:

 • ReasoningEffortSetting (dataclass or pydantic model) with:
    • kind: Literal["effort", "toggle", "budget"]
    • value: str | bool | int
      - For kind="effort": value is "minimal"|"low"|"medium"|"high"|"xhigh"
      - For kind="toggle": value is True|False
      - For kind="budget": value is int (token count)

 • ReasoningEffortSpec (per-model capability info) with:
    • kind: Literal["effort", "toggle", "budget"] — what this model supports
    • allowed_efforts: list[str] | None — for kind="effort", e.g. ["low", "medium", "high"]
    • min_budget_tokens: int | None — for kind="budget"
    • max_budget_tokens: int | None — for kind="budget"
    • default: ReasoningEffortSetting | None

NOTE: This is entirely separate from ModelParameters.reasoning (output format).

Type-safety: define explicit Literal unions for known effort strings and avoid Any.

-------------------------------------------------------------------------------------------------------------------

2. Extend ModelDatabase with reasoning effort capabilities

Add a new field to ModelParameters in src/fast_agent/llm/model_database.py:

    reasoning_effort_spec: ReasoningEffortSpec | None = None

This is SEPARATE from the existing `reasoning` field (output format). Example:

    OPENAI_O3_SERIES = ModelParameters(
        # ... existing fields ...
        reasoning="openai",                          # OUTPUT FORMAT (existing)
        reasoning_effort_spec=ReasoningEffortSpec(   # INPUT EFFORT (new)
            kind="effort",
            allowed_efforts=["low", "medium", "high"],
            default=ReasoningEffortSetting(kind="effort", value="medium"),
        ),
    )

    ANTHROPIC_37_SERIES_THINKING = ModelParameters(
        # ... existing fields ...
        reasoning="anthropic_thinking",              # OUTPUT FORMAT (existing)
        reasoning_effort_spec=ReasoningEffortSpec(   # INPUT EFFORT (new)
            kind="budget",
            min_budget_tokens=1024,
            max_budget_tokens=128000,
default=ReasoningEffortSetting(kind="budget", value=1024),
        ),
    )

This gives a central source of truth for validation, gauge calculation, and UI display.

-------------------------------------------------------------------------------------------------------------------

3. Update ModelFactory parsing to accept query params

Modify ModelFactory.parse_model_string() to accept ?reasoning=....

Rules:

 • Strip query params at the outer boundary:
    • In parse_model_string()
    • In ModelDatabase.normalize_model_name() so the database lookup ignores the query suffix.
 • Parse query parameters first (using urllib.parse.parse_qs).
 • Supported values:
    • effort strings: minimal|low|medium|high|xhigh
    • toggles: true|false|on|off|0|1
    • budgets: integers
 • Preserve existing suffix handling (model.low) for backwards compatibility.

Result: ModelConfig should carry reasoning_effort: ReasoningEffortSetting | None (renamed from reasoning_effort:
ReasoningEffort enum).

-------------------------------------------------------------------------------------------------------------------

4. Unify config-level reasoning effort settings

Update provider settings in src/fast_agent/config.py:

 • Add a new flexible field: reasoning: ReasoningEffortSetting | str | int | bool | None
 • Keep legacy fields (reasoning_effort, thinking_enabled, thinking_budget_tokens) as fallback with deprecation
   warnings (to avoid breaking configs).

For Anthropic:

 • Map thinking_enabled + thinking_budget_tokens into ReasoningEffortSetting(kind="budget", value=budget).
 • If thinking_enabled=False, use ReasoningEffortSetting(kind="toggle", value=False).

For OpenAI/Responses/Bedrock:

 • Map reasoning_effort string to ReasoningEffortSetting(kind="effort", value=effort_string).

-------------------------------------------------------------------------------------------------------------------

5. Expose reasoning effort on LLM classes

Extend FastAgentLLMProtocol with:

 • def set_reasoning_effort(self, setting: ReasoningEffortSetting | None) -> None
 • @property def reasoning_effort(self) -> ReasoningEffortSetting | None
 • @property def reasoning_effort_spec(self) -> ReasoningEffortSpec | None

Implement in:

 • FastAgentLLM (base storage + default behavior)
 • Provider classes override to:
    • validate against ReasoningEffortSpec
    • translate into provider parameters (reasoning_effort, thinking, maxReasoningTokens, etc.)

This allows the slash command and UI to query/update reasoning effort consistently.

-------------------------------------------------------------------------------------------------------------------

6. Update provider request payloads

Translate ReasoningEffortSetting into provider-native parameters:

 • OpenAI / Responses
    • kind="effort" → "reasoning_effort": "low|...|xhigh"
    • kind="toggle", value=False → omit reasoning fields
    • kind="budget" → (if supported by model) use new provider capability map; otherwise reject.

 • Bedrock
    • kind="effort" → budget via REASONING_EFFORT_BUDGETS
    • kind="budget" → use raw budget if within spec range
    • kind="toggle", value=False → reasoning disabled

 • Anthropic
    • kind="toggle" → thinking.type = "enabled" | "disabled"
    • kind="budget" → thinking.budget_tokens = value
    • (Ensure min 1024 + compatibility warnings remain)

-------------------------------------------------------------------------------------------------------------------

7. Add reasoning effort gauge to the status toolbar

Implement a reusable renderer in src/fast_agent/ui/reasoning_effort_display.py:

 • Use SINGLE gauge style from examples/reasoning_gauge_test.py.
 • Map reasoning effort to levels:
    • no effort support (reasoning_effort_spec is None) → no gauge displayed
    • kind="toggle", value=False → level 0 (bright_black full block)
    • kind="effort":
       • minimal → level 0
       • low → level 1 (green)
       • medium → level 2 (yellow)
       • high → level 3 (yellow)
       • xhigh → level 4 (red)
    • kind="budget" → derive level from min..max range:
       • min_budget → level 1
       • max_budget → level 4
       • interpolate between

Then integrate in enhanced_prompt.get_toolbar():

 • Resolve current agent's llm.reasoning_effort and llm.reasoning_effort_spec.
 • Insert gauge segment next to model chip (or after TDV flags).
 • Only show gauge if model supports reasoning effort (spec is not None).

-------------------------------------------------------------------------------------------------------------------

8. Slash command: /model reasoning <value>

Add a new command surface:

TUI

 • src/fast_agent/ui/command_payloads.py: add ModelReasoningCommand.
 • parse_special_input() (enhanced_prompt.py): parse /model reasoning XXX.
 • AgentCompleter: add command + completions (based on current model's reasoning_effort_spec).
 • interactive_prompt.py: handle the new payload and call handler.

ACP

 • src/fast_agent/acp/slash_commands.py: add AvailableCommand(name="model", ...).
 • Implement _handle_model + subcommand reasoning.
 • Reuse the shared handler (next section).

-------------------------------------------------------------------------------------------------------------------

9. Shared handler for reasoning effort updates

Add a shared command handler in src/fast_agent/commands/handlers/model.py:

 • Resolve agent + LLM
 • Parse/validate input via ReasoningEffortSetting + ReasoningEffortSpec
 • Call llm.set_reasoning_effort(...)
 • Update UI feedback + prompt user if invalid
 • Return CommandOutcome

-------------------------------------------------------------------------------------------------------------------

                                              Summary of naming

To avoid confusion, this plan uses consistent naming:

 | Concept              | Type Name               | Field Name (ModelParams) | Field Name (LLM)     |
 |----------------------|-------------------------|--------------------------|----------------------|
 | Output parsing       | (existing str)          | reasoning                | (via model_info)     |
 | Input configuration  | ReasoningEffortSetting  | reasoning_effort_spec    | reasoning_effort     |
 | Model capabilities   | ReasoningEffortSpec     | reasoning_effort_spec    | reasoning_effort_spec|

The existing `ModelParameters.reasoning` field remains unchanged and describes output format.
The new `reasoning_effort_spec` field describes input effort configuration capabilities.
