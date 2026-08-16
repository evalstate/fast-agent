from typing import Any

OPENAI_APPS_SDK_MIME_TYPE = "text/html+skybridge"
OPENAI_OUTPUT_TEMPLATE_KEY = "openai/outputTemplate"


def resource_uri(meta: dict[str, Any]) -> str | None:
    value = meta.get(OPENAI_OUTPUT_TEMPLATE_KEY)
    return value if isinstance(value, str) and value else None
