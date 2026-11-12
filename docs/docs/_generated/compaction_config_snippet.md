<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py / fast_agent.config.CompactionSettings
-->

```yaml
compaction:
  auto:         true  # automatically compact when the threshold is crossed
  # threshold:    null  # context-window fraction that triggers compaction; unset: model default
  keep_turns:   2     # recent turns kept verbatim after compaction
  strategy:     'auto' # auto: provider-native where supported; client: always summarize
  # prompt:       null  # built-in prompt; set inline text or a relative file path
```
