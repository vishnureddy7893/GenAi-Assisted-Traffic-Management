"""API response schemas.

Separate from the domain schemas on purpose. Domain models describe traffic;
these describe *this service*. Serving domain models directly from endpoints
couples the wire format to internal modelling, so that a refactor of how queues
are represented becomes a breaking API change for the frontend.

Where an endpoint genuinely returns a domain object (the safety catalog, source
capability profiles), it is returned as-is -- wrapping a stable, already
serialisable value in a near-identical DTO buys nothing.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.domain.emergency import EmergencyState


class ApiModel(BaseModel):
    """Base for API responses. Strict, but not frozen -- these are transient DTOs."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ComponentStatus(StrEnum):
    """Health of one subsystem."""

    OK = "ok"
    NOT_IMPLEMENTED = "not_implemented"
    """The subsystem belongs to a later phase. Not a fault -- but reported
    distinctly from ``ok`` so a dashboard never shows a green tick for something
    that does not exist."""

    NOT_CONFIGURED = "not_configured"
    DEGRADED = "degraded"
    FAILED = "failed"


class ComponentHealth(ApiModel):
    """Health of one named subsystem."""

    name: Annotated[str, Field(min_length=1, max_length=60)]
    status: ComponentStatus
    detail: Annotated[str, Field(max_length=300)] | None = None
    phase: Annotated[int, Field(ge=1, le=14)] | None = None
    """Phase that implements this subsystem, when it is not yet implemented."""


class HealthResponse(ApiModel):
    """Service health.

    ``status`` is liveness: is this process serving requests correctly?
    ``pipeline_ready`` is a separate question: can an end-to-end decision
    actually be produced? Through Phase 4 the answers are ``true`` and ``false``
    respectively, and reporting a single merged flag would have to lie about one
    of them.
    """

    status: ComponentStatus
    pipeline_ready: bool
    pipeline_blockers: tuple[str, ...] = ()
    """Why an end-to-end decision cannot be produced yet. Explicit so the
    dashboard states the gap rather than rendering an empty panel."""

    app_version: str
    schema_version: str
    implemented_phase: int
    implemented_phase_name: str
    environment: str
    server_time: dt.datetime
    uptime_s: float
    components: tuple[ComponentHealth, ...] = ()


class VersionsResponse(ApiModel):
    """Version pins recorded into every decision record.

    Exposed because a screenshot of a recommendation is only evidence if the
    versions that produced it are recoverable.
    """

    app_version: str
    schema_version: str
    safety_rules_version: str
    route_weights_version: str
    implemented_phase: int
    implemented_phase_name: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    """Empty until Phase 10. Empty is the honest value; a placeholder string
    would imply a model artifact exists."""


class PhaseInfo(ApiModel):
    """One development phase and whether it is implemented."""

    phase: Annotated[int, Field(ge=1, le=14)]
    name: Annotated[str, Field(min_length=1, max_length=100)]
    implemented: bool
    summary: Annotated[str, Field(max_length=300)]


class PhasesResponse(ApiModel):
    """The full phase plan and current position in it."""

    current_phase: int
    phases: tuple[PhaseInfo, ...]


class EmergencyScenariosResponse(ApiModel):
    """Declared emergency responses currently tracked by the system.

    Returns **200 with an empty collection**, not 501, and the distinction is
    principled rather than cosmetic.

    A ``NetworkState`` describes traffic, which exists whether or not anything
    observes it -- so an empty one asserts "no congestion, no incidents", a claim
    about the world that we are in no position to make. That endpoint must 501.

    An emergency scenario exists *only because someone declared it*. Zero
    declared scenarios is therefore a complete and true description of the
    system's state, not an absence of knowledge. "No emergency is currently
    declared" is exactly right, and saying it plainly is more honest than an
    error that implies the question cannot be answered.

    Declaration machinery landed in Phase 6, so ``declaration_supported``
    reports whether a network is loaded -- an emergency is anchored to junctions
    and facilities, and none can be declared against a map that is not there.
    """

    scenarios: tuple[EmergencyState, ...] = ()
    """Declared responses, newest first. Empty because nothing was declared,
    never because something was filtered."""

    count: Annotated[int, Field(ge=0)] = 0
    active_count: Annotated[int, Field(ge=0)] = 0

    declaration_supported: bool = False
    """Whether an emergency can currently be declared at all."""

    declaration_phase: Annotated[int, Field(ge=1, le=14)] | None = None
    """Phase that implements declaration. Retained now that declaration exists,
    because a client may still be told *why* it is unavailable -- no loaded
    network -- and the phase label keeps that message consistent with the rest
    of the API."""

    note: Annotated[str, Field(max_length=400)]
    """Operator-facing explanation of why the collection is empty. Prevents the
    empty list from being read as "the emergency was cleared"."""


class ErrorBody(ApiModel):
    """Machine-readable error detail."""

    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(ApiModel):
    """Uniform error envelope for every failure this service returns."""

    error: ErrorBody
