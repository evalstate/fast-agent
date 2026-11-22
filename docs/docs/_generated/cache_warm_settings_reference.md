<!-- GENERATED FILE — DO NOT EDIT. Source: CacheWarmSettings -->

| Setting | Default | Description |
| --- | --- | --- |
| `cache_warm.auto` | `true` | Keep the prompt cache warm automatically while a turn waits on tools or subagents, and while an idle session waits for herdr-core children to report |
| `cache_warm.max_minutes` | `30` | How long automatic warming continues after the last model request; also the default /warm duration |
| `cache_warm.interval_seconds` | `null` | Seconds between keep-warm requests. Unset derives it from the provider's cache retention; set it to warm a model whose retention fast-agent does not know |
