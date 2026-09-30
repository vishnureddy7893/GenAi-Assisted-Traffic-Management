"""Request-scoped correlation context.

Every log line emitted while handling a request carries the same ``request_id``,
and — once scenarios exist (Phase 6+) — the same ``scenario_id``.  Without this,
tracing a single emergency decision through ingestion, routing, safety and GenAI
means grepping timestamps across subsystems, which does not survive concurrency.

Implemented with :mod:`contextvars` so it is correct under asyncio: each task
gets its own copy rather than sharing process-global state.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_scenario_id: ContextVar[str | None] = ContextVar("scenario_id", default=None)


def new_request_id() -> str:
    """Generate a fresh correlation id."""
    return uuid.uuid4().hex[:16]


def get_request_id() -> str | None:
    """Return the correlation id bound to the current context, if any."""
    return _request_id.get()


def set_request_id(value: str | None) -> Token[str | None]:
    """Bind a correlation id to the current context."""
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous correlation id."""
    _request_id.reset(token)


def get_scenario_id() -> str | None:
    """Return the scenario id bound to the current context, if any."""
    return _scenario_id.get()


def set_scenario_id(value: str | None) -> Token[str | None]:
    """Bind a scenario id to the current context."""
    return _scenario_id.set(value)


@contextmanager
def bind_context(
    *, request_id: str | None = None, scenario_id: str | None = None
) -> Iterator[None]:
    """Temporarily bind correlation ids, restoring previous values on exit."""
    tokens: list[tuple[ContextVar[Any], Token[Any]]] = []
    if request_id is not None:
        tokens.append((_request_id, _request_id.set(request_id)))
    if scenario_id is not None:
        tokens.append((_scenario_id, _scenario_id.set(scenario_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_context() -> dict[str, str]:
    """Snapshot the bound correlation ids for inclusion in a log record."""
    ctx: dict[str, str] = {}
    if (rid := _request_id.get()) is not None:
        ctx["request_id"] = rid
    if (sid := _scenario_id.get()) is not None:
        ctx["scenario_id"] = sid
    return ctx
