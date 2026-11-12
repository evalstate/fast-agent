"""JSON telemetry preserves scalar types even when Python reuses object identities."""

import json

from fast_agent.core.logging.json_serializer import JSONSerializer


def test_repeated_json_scalars_keep_their_types() -> None:
    values = [False, False, True, True, 2, 2, 0.0, 0.0, None, None, "same", "same"]
    result = json.loads(json.dumps(JSONSerializer()({"first": values, "second": values.copy()})))
    for key in ("first", "second"):
        assert result[key] == values
        assert [type(value) for value in result[key]] == [type(value) for value in values]
