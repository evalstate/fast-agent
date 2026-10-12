<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py / fast_agent.config.CompactionSettings
-->

| Setting | Default | Description |
| --- | --- | --- |
| `compaction.auto` | `true` | Automatically compact history when context usage crosses the threshold |
| `compaction.threshold` | `null` | Fraction of the model context window that triggers auto-compaction, for every model. Unset uses the model's default (e.g. Grok 4.7 compacts at 190k tokens), else 0.85. A model string's compact_at overrides it |
| `compaction.keep_turns` | `2` | Number of recent complete turns kept verbatim after compaction |
| `compaction.strategy` | `'auto'` | How compaction is performed. 'auto' uses provider-native compaction where the model supports it and fast-agent's summary otherwise; 'client' always uses fast-agent's summary |
| `compaction.prompt` | `null` | Custom summarization prompt for compaction. Inline text, or a path to a text/markdown file. None uses the built-in prompt (see /compact prompt). |
