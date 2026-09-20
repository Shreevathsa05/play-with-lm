"""Sanitized job-scoped logging. Never log tokens, usernames, or dataset samples."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

_SECRET_KEYS = frozenset(
    {
        "token",
        "hf_token",
        "access_token",
        "password",
        "secret",
        "authorization",
        "api_key",
    }
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class LogCarrier:
    """Emit structured events keyed only by job_uuid."""

    def __init__(
        self,
        job_uuid: str,
        logger: Optional[logging.Logger] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        if not job_uuid or not str(job_uuid).strip():
            raise ValueError("job_uuid is required")
        self.job_uuid = str(job_uuid).strip()
        self._logger = logger or logging.getLogger(f"job.{self.job_uuid}")
        self.on_event = on_event

    def _sanitize(self, payload: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = key.lower()
            if lowered in _SECRET_KEYS or any(s in lowered for s in ("token", "password", "secret")):
                clean[key] = "[REDACTED]"
            elif isinstance(value, str):
                clean[key] = _EMAIL_RE.sub("[REDACTED_EMAIL]", value)
            elif isinstance(value, dict):
                clean[key] = self._sanitize(value)
            else:
                clean[key] = value
        return clean

    def event(self, event: str, level: int = logging.INFO, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "job_uuid": self.job_uuid,
            "event": event,
            **self._sanitize(fields),
        }
        self._logger.log(level, json.dumps(record, default=str))
        if self.on_event is not None:
            try:
                self.on_event(record)
            except Exception:
                self._logger.exception("on_event callback failed for job %s", self.job_uuid)
        return record

    def info(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.event(event, level=logging.INFO, **fields)

    def warning(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.event(event, level=logging.WARNING, **fields)

    def error(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.event(event, level=logging.ERROR, **fields)
