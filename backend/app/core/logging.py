"""Structured logging foundation.

Why structured logs rather than ``print`` or bare ``logging``
-------------------------------------------------------------
This system will eventually emit a decision (route, safety verdict, ETA) that
somebody may need to audit after the fact. Free-text logs cannot be queried by
``scenario_id`` or ``junction_id``. Every record here is a flat object with
stable keys, so JSON output can be shipped to any log store and filtered.

Two formats are supported:

* ``console`` -- aligned, human-readable; the default for local development.
* ``json``    -- one JSON object per line; use in CI, staging and production.

Correlation ids from :mod:`app.core.context` are attached automatically, so call
sites do not have to thread them through function signatures.

Standard-library ``logging`` is used deliberately rather than a third-party
framework: it has zero dependency risk, and it is what uvicorn and every library
we will add already emit into -- capturing those loggers is the whole point.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any, Final

from app.core.context import current_context

#: Attributes present on every ``logging.LogRecord``; anything else a caller
#: passed via ``extra=`` is treated as structured payload and emitted.
_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

_CONFIGURED: bool = False


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """Extract caller-supplied structured fields from a log record."""
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(current_context())
        payload.update(_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        # default=str keeps the logger from ever raising on an exotic value:
        # a logging call must not be able to break a request.
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Aligned, human-readable output for local development."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = dt.datetime.fromtimestamp(record.created, tz=dt.UTC)
        ts = stamp.strftime("%H:%M:%S.%f")[:-3]
        ctx = current_context()
        rid = f" [{ctx['request_id']}]" if "request_id" in ctx else ""
        sid = f" [scenario={ctx['scenario_id']}]" if "scenario_id" in ctx else ""
        extras = _extras(record)
        suffix = ""
        if extras:
            suffix = " " + " ".join(f"{k}={v!r}" for k, v in sorted(extras.items()))
        head = f"{ts} {record.levelname:<8} {record.name:<30}"
        line = f"{head}{rid}{sid} {record.getMessage()}{suffix}"
        if record.exc_info:
            line = line + "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(*, level: str = "INFO", fmt: str = "console", force: bool = False) -> None:
    """Install the root logging configuration.

    Idempotent: repeated calls are ignored unless ``force`` is set, so importing
    the app twice (as test collection does) will not duplicate every log line.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    formatter: logging.Formatter = JsonFormatter() if fmt == "json" else ConsoleFormatter()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn's own loggers through our handler so access logs and
    # application logs share one format and one correlation id.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Always call with ``__name__`` so the emitted ``logger`` field identifies the
    subsystem (``app.ingestion.recorded``, ``app.domain.safety``, ...).
    """
    return logging.getLogger(name)
