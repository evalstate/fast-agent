<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

| Model | Provider | Tokenizes | Structured Output | Reasoning | Verbosity | Built-in Web Tools |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen-long` | `aliyun` | Text | `json` (schema) | — | — | — |
| `qwen-max` | `aliyun` | Text, Vision | `json` (object) | — | — | — |
| `qwen-plus` | `aliyun` | Text, Vision | `json` (object) | — | — | — |
| `qwen-turbo` | `aliyun` | Text, Vision | `json` (object) | — | — | — |
| `qwen3-max` | `aliyun` | Text | `json` (schema) | — | — | — |
| `claude-3-5-haiku-20241022` | `anthropic` | Text, Vision, Document | `tool_use` | — | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-3-5-haiku-latest` | `anthropic` | Text, Vision, Document | `tool_use` | — | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-3-5-haiku` | `anthropic` | Text, Vision, Document | `tool_use` | — | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-haiku-4-5-20251001` | `anthropic` | Text, Vision, Document | `json` (schema) | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-haiku-4-5-20251001?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-opus-4-0` | `anthropic` | Text, Vision, Document | `tool_use` | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-opus-4-0?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-opus-4-1` | `anthropic` | Text, Vision, Document | `json` (schema) | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-opus-4-1?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-opus-4-20250514` | `anthropic` | Text, Vision, Document | `tool_use` | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-opus-4-20250514?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-opus-4-5` | `anthropic` | Text, Vision, Document | `json` (schema) | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-opus-4-5?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-sonnet-4-0` | `anthropic` | Text, Vision, Document | `tool_use` | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-sonnet-4-0?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-sonnet-4-20250514` | `anthropic` | Text, Vision, Document | `tool_use` | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-sonnet-4-20250514?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-sonnet-4-5-20250929` | `anthropic` | Text, Vision, Document | `json` (schema) | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-sonnet-4-5-20250929?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude-sonnet-4-5` | `anthropic` | Text, Vision, Document | `json` (schema) | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `claude-sonnet-4-5?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `claude` | `anthropic` | Text, Vision, Document | `json` (schema) | effort: `auto`, `low`, `medium`, `high`, `max`, `off`<br>Example: `claude.auto` | — | `web_search` (web_search_20260209)<br>`web_fetch` (web_fetch_20260209)<br>beta: `code-execution-web-tools-2026-02-09` |
| `haiku` | `anthropic` | Text, Vision, Document | `json` (schema) | budget: `low`, `medium`, `high`, `max`, `0`, `1024`, `16000`, `32000`, `off`<br>Example: `haiku?reasoning=1024` | — | `web_search` (web_search_20250305)<br>`web_fetch` (web_fetch_20250910) |
| `opus46` | `anthropic` | Text, Vision, Document | `json` (schema) | effort: `auto`, `low`, `medium`, `high`, `max`, `off`<br>Example: `opus46.auto` | — | `web_search` (web_search_20260209)<br>`web_fetch` (web_fetch_20260209)<br>beta: `code-execution-web-tools-2026-02-09` |
| `opus47` | `anthropic` | Text, Vision, Document | `json` (schema) | effort: `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `off`<br>Example: `opus47.auto` | — | `web_search` (web_search_20260209)<br>`web_fetch` (web_fetch_20260209)<br>beta: `code-execution-web-tools-2026-02-09` |
| `opus` | `anthropic` | Text, Vision, Document | `json` (schema) | effort: `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `off`<br>Example: `opus.auto` | — | `web_search` (web_search_20260209)<br>`web_fetch` (web_fetch_20260209)<br>beta: `code-execution-web-tools-2026-02-09` |
| `codexspark` | `codexresponses` | Text | `json` (schema) | — | — | — |
| `deepseek-chat` | `deepseek` | Text | `json` (object) | — | — | — |
| `deepseek-reasoner` | `deepseek` | Text | `json` (schema) | effort: `high`, `max`, `off`<br>Example: `deepseek-reasoner.high` | — | — |
| `deepseek.deepseek-ai/deepseek-v3.1` | `deepseek` | Text | `json` (schema) | — | — | — |
| `deepseek4flash` | `deepseek` | Text | `json` (object) | effort: `high`, `max`, `off`<br>Example: `deepseek4flash.high` | — | — |
| `deepseek` | `deepseek` | Text | `json` (object) | effort: `high`, `max`, `off`<br>Example: `deepseek.high` | — | — |
| `passthrough` | `fast-agent` | Text | `json` (schema) | — | — | — |
| `playback` | `fast-agent` | Text | `json` (schema) | — | — | — |
| `silent` | `fast-agent` | Text | `json` (schema) | — | — | — |
| `slow` | `fast-agent` | Text | `json` (schema) | — | — | — |
| `gemini25` | `google` | Text, Vision, Document, Audio, Video | `json` (schema) | effort: `auto`, `minimal`, `low`, `medium`, `high`, `off`<br>Example: `gemini25.auto` | — | — |
| `gemini25pro` | `google` | Text, Vision, Document, Audio, Video | `json` (schema) | effort: `auto`, `minimal`, `low`, `medium`, `high`, `off`<br>Example: `gemini25pro.auto` | — | — |
| `gemini2` | `google` | Text, Vision, Document, Audio, Video | `json` (schema) | — | — | — |
| `gemini3.1flashlite` | `google` | Text, Vision, Document, Audio, Video | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gemini3.1flashlite.medium` | — | — |
| `gemini35` | `google` | Text, Vision, Document, Audio, Video | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gemini35.medium` | — | — |
| `gemini3` | `google` | Text, Vision, Document, Audio, Video | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gemini3.medium` | — | — |
| `gemini3flash` | `google` | Text, Vision, Document, Audio, Video | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gemini3flash.medium` | — | — |
| `gemini` | `google` | Text, Vision, Document, Audio, Video | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gemini.medium` | — | — |
| `groq.deepseek-r1-distill-llama-70b` | `groq` | Text | `json` (object) | — | — | — |
| `groq.qwen/qwen3-32b` | `groq` | Text | `json` (object) | — | — | — |
| `moonshotai/kimi-k2-instruct-0905` | `groq` | Text | `json` (schema) | — | — | — |
| `moonshotai/kimi-k2-thinking` | `groq` | Text | `json` (schema) | — | — | — |
| `moonshotai/kimi-k2` | `groq` | Text | `json` (schema) | — | — | — |
| `deepseek-hf` | `hf` | Text | `json` (schema) | — | — | — |
| `deepseek32` | `hf` | Text | `json` (schema) | — | — | — |
| `glm47` | `hf` | Text | `json` (schema) | toggle: `on`, `off`<br>Example: `glm47?reasoning=off` | — | — |
| `glm5` | `hf` | Text | `json` (schema) | toggle: `on`, `off`<br>Example: `glm5?reasoning=off` | — | — |
| `glm` | `hf` | Text | `json` (schema) | toggle: `on`, `off`<br>Example: `glm?reasoning=off` | — | — |
| `gpt-oss-20b` | `hf` | Text | `json` (schema) | — | — | — |
| `gpt-oss` | `hf` | Text | `json` (schema) | — | — | — |
| `hf.minimaxai/minimax-m2` | `hf` | Text | `json` (schema) | — | — | — |
| `hf.qwen/qwen3-next-80b-a3b-instruct` | `hf` | Text | `json` (schema) | — | — | — |
| `hf.zai-org/glm-4.6` | `hf` | Text | `json` (schema) | — | — | — |
| `kimi25` | `hf` | Text, Vision | `json` (schema) | toggle: `on`, `off`<br>Example: `kimi25?reasoning=off` | — | — |
| `kimi` | `hf` | Text, Vision | `json` (schema) | toggle: `on`, `off`<br>Example: `kimi?reasoning=off` | — | — |
| `minimax21` | `hf` | Text | `json` (schema) | — | — | — |
| `minimax25` | `hf` | Text | `json` (schema) | toggle: `on`, `off`<br>Example: `minimax25?reasoning=off` | — | — |
| `minimax` | `hf` | Text | `json` (schema) | — | — | — |
| `qwen35` | `hf` | Text, Vision | `json` (object) | toggle: `on`, `off`<br>Example: `qwen35?reasoning=off` | — | — |
| `gpt-4.1-2025-04-14` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4.1-mini-2025-04-14` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4.1-mini` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4.1-nano-2025-04-14` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4.1-nano` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4.1` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4o-2024-11-20` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4o-mini-2024-07-18` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4o-mini` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-4o` | `openai` | Text, Vision, Document | `json` (schema) | — | — | — |
| `chat-latest` | `responses` | Text, Vision, Document | `json` (schema) | — | — | — |
| `codex` | `responses` | Text, Vision, Document | `json` (schema) | effort: `low`, `medium`, `high`, `xhigh`<br>Example: `codex.medium` | `low`, `medium`, `high`<br>Example: `codex?verbosity=low` | — |
| `gpt-5-mini` | `responses` | Text, Vision, Document | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gpt-5-mini.medium` | `low`, `medium`, `high`<br>Example: `gpt-5-mini?verbosity=low` | — |
| `gpt-5-nano-2025-08-07` | `responses` | Text, Vision, Document | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gpt-5-nano-2025-08-07.medium` | `low`, `medium`, `high`<br>Example: `gpt-5-nano-2025-08-07?verbosity=low` | — |
| `gpt-5-nano` | `responses` | Text, Vision, Document | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gpt-5-nano.medium` | `low`, `medium`, `high`<br>Example: `gpt-5-nano?verbosity=low` | — |
| `gpt-5.3-chat-latest` | `responses` | Text, Vision, Document | `json` (schema) | — | — | — |
| `gpt-5` | `responses` | Text, Vision, Document | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`<br>Example: `gpt-5.medium` | `low`, `medium`, `high`<br>Example: `gpt-5?verbosity=low` | — |
| `gpt51` | `responses` | Text, Vision, Document | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `xhigh`, `off`<br>Example: `gpt51.none` | `low`, `medium`, `high`<br>Example: `gpt51?verbosity=low` | — |
| `gpt52` | `responses` | Text, Vision, Document | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `xhigh`, `off`<br>Example: `gpt52.none` | `low`, `medium`, `high`<br>Example: `gpt52?verbosity=low` | — |
| `gpt54-mini` | `responses` | Text, Vision | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `xhigh`, `off`<br>Example: `gpt54-mini.none` | `low`, `medium`, `high`<br>Example: `gpt54-mini?verbosity=low` | — |
| `gpt54-nano` | `responses` | Text, Vision | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `xhigh`, `off`<br>Example: `gpt54-nano.none` | `low`, `medium`, `high`<br>Example: `gpt54-nano?verbosity=low` | — |
| `gpt54` | `responses` | Text, Vision, Document | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `xhigh`, `off`<br>Example: `gpt54.none` | `low`, `medium`, `high`<br>Example: `gpt54?verbosity=low` | — |
| `gpt55` | `responses` | Text, Vision, Document | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `xhigh`, `off`<br>Example: `gpt55.none` | `low`, `medium`, `high`<br>Example: `gpt55?verbosity=low` | — |
| `o1-2024-12-17` | `responses` | Text, Vision | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`, `xhigh`<br>Example: `o1-2024-12-17.medium` | — | — |
| `o1-mini` | `responses` | Text, Vision | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`, `xhigh`<br>Example: `o1-mini.medium` | — | — |
| `o1-preview` | `responses` | Text, Vision | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`, `xhigh`<br>Example: `o1-preview.medium` | — | — |
| `o1` | `responses` | Text, Vision | `json` (schema) | effort: `minimal`, `low`, `medium`, `high`, `xhigh`<br>Example: `o1.medium` | — | — |
| `o3-2025-04-16` | `responses` | Text, Vision, Document | `json` (schema) | effort: `low`, `medium`, `high`<br>Example: `o3-2025-04-16.medium` | — | — |
| `o3-mini-2025-01-31` | `responses` | Text | `json` (schema) | effort: `low`, `medium`, `high`<br>Example: `o3-mini-2025-01-31.medium` | — | — |
| `o3-mini` | `responses` | Text | `json` (schema) | effort: `low`, `medium`, `high`<br>Example: `o3-mini.medium` | — | — |
| `o3` | `responses` | Text, Vision, Document | `json` (schema) | effort: `low`, `medium`, `high`<br>Example: `o3.medium` | — | — |
| `o4-mini-2025-04-16` | `responses` | Text, Vision, Document | `json` (schema) | effort: `low`, `medium`, `high`<br>Example: `o4-mini-2025-04-16.medium` | — | — |
| `o4-mini` | `responses` | Text, Vision, Document | `json` (schema) | effort: `low`, `medium`, `high`<br>Example: `o4-mini.medium` | — | — |
| `responses.o3-pro` | `responses` | Text | `json` (schema) | — | — | — |
| `grok-3-fast` | `xai` | Text | `json` (schema) | — | — | — |
| `grok-3-latest` | `xai` | Text | `json` (schema) | — | — | — |
| `grok-3-mini-fast` | `xai` | Text | `json` (schema) | — | — | — |
| `grok-3-mini` | `xai` | Text | `json` (schema) | — | — | — |
| `grok-3` | `xai` | Text | `json` (schema) | — | — | — |
| `grok-4-0709` | `xai` | Text | `json` (schema) | — | — | — |
| `grok-4-1-fast-non-reasoning` | `xai` | Text, Vision | `json` (schema) | — | — | — |
| `grok-4-1-fast-reasoning` | `xai` | Text, Vision | `json` (schema) | — | — | — |
| `grok-4-fast-reasoning` | `xai` | Text, Vision | `json` (schema) | — | — | — |
| `grok-4-fast` | `xai` | Text, Vision | `json` (schema) | — | — | — |
| `grok-4-latest` | `xai` | Text, Vision | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `off`<br>Example: `grok-4-latest.low` | — | — |
| `grok-4.3-latest` | `xai` | Text, Vision | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `off`<br>Example: `grok-4.3-latest.low` | — | — |
| `grok-4` | `xai` | Text, Vision | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `off`<br>Example: `grok-4.low` | — | — |
| `grok` | `xai` | Text, Vision | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `off`<br>Example: `grok.low` | — | — |
| `grok` | `xai` | Text, Vision | `json` (schema) | effort: `none`, `low`, `medium`, `high`, `off`<br>Example: `grok.low` | — | — |
