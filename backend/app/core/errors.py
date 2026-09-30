"""Application error taxonomy and its HTTP mapping.

Rationale
---------
Distinct failure modes must be distinguishable by a client *and* by a log query.
A bare 500 tells an operator nothing about whether the cause was a bad upload, a
disconnected camera, or a genuine bug.

``NotAvailableYetError`` deserves special mention: it represents "this capability
belongs to a later development phase", and maps to **501 Not Implemented** rather
than a 404 or an empty success payload. An unimplemented feature must never
return an empty-but-successful response, because the frontend would then render a
legitimate-looking blank dashboard that is indistinguishable from "there is
genuinely no traffic". That distinction is safety-relevant.
"""

from __future__ import annotations

from typing import Any


class TrafficSystemError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    error_code: str = "internal_error"

    is_expected: bool = False
    """Whether this condition is a designed outcome rather than a fault.

    Drives log severity. Status code alone is not sufficient to decide: a 501
    from phased development is a 5xx that is nonetheless working exactly as
    intended, and logging it at ERROR trains an operator to ignore ERROR -- or,
    worse, makes a healthy system look broken in the console while the HTTP
    responses are entirely correct.
    """

    def __init__(self, message: str, /, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_payload(self) -> dict[str, Any]:
        """Render a stable, machine-readable error body."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }


class ConfigurationError(TrafficSystemError):
    """Invalid or missing configuration. Non-recoverable at runtime."""

    status_code = 500
    error_code = "configuration_error"


class ValidationFailedError(TrafficSystemError):
    """Caller supplied structurally valid but semantically invalid input."""

    status_code = 422
    error_code = "validation_failed"


class NotAvailableYetError(TrafficSystemError):
    """Capability belongs to a development phase that is not implemented.

    Carries the owning phase so the UI can label the gap precisely instead of
    silently rendering an empty panel.
    """

    status_code = 501
    error_code = "not_available_yet"
    is_expected = True

    def __init__(self, message: str, /, *, phase: int, phase_name: str, **details: Any) -> None:
        super().__init__(message, phase=phase, phase_name=phase_name, **details)


class ResourceNotFoundError(TrafficSystemError):
    """A referenced resource does not exist.

    Distinct from :class:`DataSourceError`: "this id was never ingested" is a
    client mistake (404), whereas "the file is there but will not decode" is an
    upstream failure (502). Collapsing them would tell an operator to check the
    decoder when the real problem is a stale identifier.
    """

    status_code = 404
    error_code = "not_found"


class NetworkNotLoadedError(ResourceNotFoundError):
    """No road network has been loaded into this process.

    A 404 like any other missing resource, but a *designed* state rather than a
    fault: a freshly started service has no map until someone chooses one, and
    that is the normal condition rather than an error. It is therefore expected,
    so a healthy service serving its first request does not fill the console
    with ERROR -- the same reasoning that applies to a phase boundary.

    A distinct ``error_code`` from a generic 404 so a client can tell "you asked
    for an id that does not exist" from "nothing has been loaded at all", which
    call for completely different responses.
    """

    error_code = "network_not_loaded"
    is_expected = True


class DataSourceError(TrafficSystemError):
    """A traffic data source failed to open, read or decode."""

    status_code = 502
    error_code = "data_source_error"


class SourceCapabilityError(TrafficSystemError):
    """An operation was requested that the source does not support.

    Example: seeking a live RTSP stream. Rejected explicitly rather than silently
    ignored, because a silently-ignored seek produces a decision based on the
    wrong point in time.
    """

    status_code = 409
    error_code = "unsupported_source_capability"


class SafetyViolationError(TrafficSystemError):
    """A hard safety constraint was violated by a proposed action.

    This is never caught and downgraded. If it is raised, the action is refused.
    """

    status_code = 409
    error_code = "safety_violation"


class NoAdmissibleRouteError(TrafficSystemError):
    """Every candidate route was pruned by a hard safety constraint.

    A first-class outcome, not a bug: the correct response is to report that no
    safe route exists and surface why, never to fall back to an unsafe route.
    """

    status_code = 409
    error_code = "no_admissible_route"
