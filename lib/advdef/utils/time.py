"""Time helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_timestamp() -> str:
    """Return an ISO8601 timestamp in UTC with second precision."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
