from __future__ import annotations

from datetime import datetime, tzinfo

import pytest


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> _FixedDatetime:
        return cls(2026, 9, 1, 12, 34, 56, 789012, tzinfo=tz)


@pytest.fixture
def fixed_datetime() -> type[datetime]:
    """A ``datetime`` replacement whose ``now()`` returns 2026-09-01 12:34:56.789012."""
    return _FixedDatetime
