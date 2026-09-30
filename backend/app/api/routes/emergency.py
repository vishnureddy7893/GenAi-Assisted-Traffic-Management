"""Emergency declaration, lifecycle, and route recommendation endpoints.

Phase 6 scope: **declaring a response and tracking it**. Which unit is
responding, at what priority, from where, to what, and how far along it is.

Routing lives elsewhere. Phase 7 implements route optimization at
``POST /api/routes/optimize``, which accepts an ``emergency_id`` and reads the
response's own record for its origin, destination and vehicle type. The
placeholder below therefore no longer claims the capability is missing -- it
says where it went, because a 501 for something that exists is its own kind of
false statement.

Nothing here returns a sample or demonstration route. A fabricated emergency
recommendation is the one output of this system that must never exist, because
it is the one a person might act on.

Declaration, not detection
--------------------------
An emergency exists because somebody declared it. A camera recognising an
ambulance silhouette is corroborating evidence and nothing more: letting a
visual detection create a response would let one false positive trigger corridor
recommendations across a network.

Every declaration is anchored to the loaded network
---------------------------------------------------
A response names junctions, segments and facilities, and each must exist on the
map that is loaded. Rejecting a typo here costs one 422; discovering it inside
route search costs a lookup failure on a live emergency request.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import Field

from app.api.deps import (
    EmergencyRegistryDep,
    IncidentRegistryDep,
    NetworkStoreDep,
)
from app.api.schemas import ApiModel, EmergencyScenariosResponse, ErrorResponse
from app.core.errors import NotAvailableYetError, ValidationFailedError
from app.core.logging import get_logger
from app.core.phases import phase
from app.domain.emergency import (
    EmergencyDestination,
    EmergencyState,
    EmergencyVehicle,
    NetworkPosition,
)
from app.domain.enums import EmergencyPriority, EmergencyStatus
from app.domain.provenance import Provenance, ValueSource
from app.emergency.responses import EMERGENCY_TRANSITIONS

logger = get_logger(__name__)
router = APIRouter(prefix="/emergency", tags=["emergency"])

_NOT_IMPLEMENTED_RESPONSES: dict[int | str, dict[str, object]] = {
    501: {"model": ErrorResponse, "description": "Capability belongs to a later phase"}
}

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "No network loaded, or unknown emergency"},
    422: {"model": ErrorResponse, "description": "Declaration does not fit the loaded network"},
}

#: Sources a dispatch may claim. A response is declared by a human or a scenario,
#: never derived or predicted.
_DECLARABLE_SOURCES = frozenset(
    {ValueSource.OBSERVED, ValueSource.ASSUMED, ValueSource.STATIC, ValueSource.SIMULATED}
)


class DeclareEmergencyRequest(ApiModel):
    """A new emergency response."""

    emergency_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    vehicle: EmergencyVehicle
    priority: EmergencyPriority
    current_position: NetworkPosition
    destination: EmergencyDestination
    origin: NetworkPosition | None = None

    declared_at: dt.datetime | None = None
    """Origin of the response-time clock. Every ETA claim is measured against
    this, not against when this system happened to compute a route."""

    triggering_incident_id: Annotated[str, Field(max_length=64)] | None = None
    notes: Annotated[str, Field(max_length=1000)] | None = None

    declared_by: Annotated[str, Field(min_length=1, max_length=200)] = "operator"
    source: ValueSource = ValueSource.OBSERVED
    source_notes: Annotated[str, Field(max_length=1000)] | None = None
    """Required when ``source`` is ``assumed``. A scenario-declared emergency
    must stay distinguishable from a real dispatch record in every downstream
    artifact."""


class TransitionEmergencyRequest(ApiModel):
    """A lifecycle change."""

    status: EmergencyStatus
    at: dt.datetime | None = None


class UpdatePositionRequest(ApiModel):
    """Where the responding unit now is."""

    position: NetworkPosition
    at: dt.datetime | None = None


class EmergencyResponseBody(ApiModel):
    """One declared response, with what it may do next."""

    emergency: EmergencyState
    permitted_next_statuses: tuple[EmergencyStatus, ...] = ()
    warrants_corridor_preemption: bool = False
    """Whether the priority tier justifies recommending corridor pre-emption.
    Only P1: pre-emption imposes real delay and real risk on every conflicting
    movement, so it is reserved for immediately life-threatening responses."""

    elapsed_response_time_s: float | None = None


def _permitted_next(status: EmergencyStatus) -> tuple[EmergencyStatus, ...]:
    """Statuses this response may move to next."""
    return tuple(
        sorted(EMERGENCY_TRANSITIONS.get(status, frozenset()), key=lambda item: item.value)
    )


def _body(emergency: EmergencyState, *, now: dt.datetime) -> EmergencyResponseBody:
    """Render one response with its derived affordances."""
    return EmergencyResponseBody(
        emergency=emergency,
        permitted_next_statuses=_permitted_next(emergency.status),
        warrants_corridor_preemption=emergency.warrants_corridor_preemption,
        elapsed_response_time_s=round(emergency.elapsed_response_time_s(now), 3),
    )


@router.get(
    "/scenarios",
    response_model=EmergencyScenariosResponse,
    summary="Declared emergency scenarios",
)
def list_scenarios(
    emergencies: EmergencyRegistryDep,
    store: NetworkStoreDep,
    active_only: Annotated[bool, Query()] = False,
) -> EmergencyScenariosResponse:
    """Return the declared emergency responses.

    Answers 200 with an empty collection rather than 501, and the distinction is
    principled: an emergency exists only because someone declared it, so "none
    declared" is a complete and true description of our records. That differs
    from ``/api/network/state``, which describes traffic that exists whether or
    not we observe it.
    """
    items = emergencies.list(active_only=active_only)
    active = emergencies.active()
    loaded = store.is_loaded
    target = phase(6)

    if not loaded:
        note = (
            "No road network is loaded, so no emergency can be declared: a response "
            "is anchored to junctions and facilities that must exist on a map. Load "
            "one with POST /api/network/load."
        )
    elif items:
        note = f"{len(items)} declared response(s), {len(active)} still active."
    else:
        note = (
            "No emergency response has been declared. This collection is empty "
            "because nothing was declared, not because anything was cleared or "
            "filtered."
        )

    return EmergencyScenariosResponse(
        scenarios=items,
        count=len(items),
        active_count=len(active),
        declaration_supported=loaded,
        declaration_phase=None if loaded else target.number,
        note=note,
    )


@router.post(
    "/declare",
    response_model=EmergencyResponseBody,
    status_code=201,
    responses=_ERRORS,
    summary="Declare an emergency response",
)
def declare_emergency(
    request: DeclareEmergencyRequest,
    emergencies: EmergencyRegistryDep,
    incidents: IncidentRegistryDep,
    store: NetworkStoreDep,
) -> EmergencyResponseBody:
    """Declare a response against the loaded network."""
    network = store.require()
    now = dt.datetime.now(tz=dt.UTC)
    declared_at = request.declared_at or now

    if request.source not in _DECLARABLE_SOURCES:
        msg = (
            f"source {request.source.value!r} cannot be claimed by a declaration; "
            f"choose one of {sorted(item.value for item in _DECLARABLE_SOURCES)}."
        )
        raise ValidationFailedError(msg, source=request.source.value)

    if declared_at > now:
        msg = (
            "declared_at is in the future. The response-time clock starts at "
            "declaration, and a future start would report negative elapsed time on "
            "every subsequent ETA."
        )
        raise ValidationFailedError(msg)

    emergency = EmergencyState(
        emergency_id=request.emergency_id or f"emergency-{uuid.uuid4().hex[:12]}",
        vehicle=request.vehicle,
        priority=request.priority,
        status=EmergencyStatus.DECLARED,
        declared_at=declared_at,
        updated_at=now,
        origin=request.origin,
        current_position=request.current_position,
        destination=request.destination,
        triggering_incident_id=request.triggering_incident_id,
        provenance=Provenance(
            source=request.source,
            producer=request.declared_by,
            producer_version="1.0",
            observed_at=declared_at,
            computed_at=now,
            notes=request.source_notes,
        ),
        notes=request.notes,
    )

    known_incidents = frozenset(item.incident_id for item in incidents.list())
    stored = emergencies.declare(emergency, network, known_incident_ids=known_incidents)

    logger.info(
        "emergency declared",
        extra={
            "emergency_id": stored.emergency_id,
            "vehicle_type": stored.vehicle.vehicle_type.value,
            "priority": stored.priority.value,
        },
    )
    return _body(stored, now=now)


# ---------------------------------------------------------------------------
# Routing and corridor staging
#
# Declared before the ``/{emergency_id}`` routes below. FastAPI matches in
# declaration order, so a literal path registered after a path parameter would
# be shadowed by it -- ``/emergency/recommendation`` would resolve as an
# emergency whose id is "recommendation".
# ---------------------------------------------------------------------------
@router.get(
    "/recommendation",
    responses={422: {"model": ErrorResponse, "description": "Routing moved"}},
    summary="Emergency route recommendation (moved to /api/routes/optimize)",
)
def recommendation() -> None:
    """Point the caller at the optimizer, which now exists.

    This endpoint predates route optimization and names no emergency, so it
    cannot route anything on its own. It answers 422 rather than 501: the
    capability is implemented, and claiming otherwise would be a false statement
    about the system in the direction that makes it look less capable than it is.
    """
    msg = (
        "Route optimization is implemented at POST /api/routes/optimize. Pass "
        "{\"emergency_id\": \"...\"} to route a declared response -- its origin, "
        "destination and vehicle type are read from its own record rather than "
        "restated here, so the two cannot disagree. This endpoint names no "
        "emergency and so has nothing to route."
    )
    raise ValidationFailedError(msg, moved_to="/api/routes/optimize")


@router.get(
    "/corridor",
    responses=_NOT_IMPLEMENTED_RESPONSES,
    summary="Emergency green corridor advisory (phase 9)",
)
def corridor() -> None:
    """Return a staged corridor advisory. Not available until Phase 9.

    Distinct from routing, which now exists. A corridor advisory proposes
    *changing signal state* along a recommended route, which is a
    :class:`~app.domain.decision.SignalAdvisory` and belongs to the decision
    layer -- and it needs conflicting-movement checks against junction phasing
    that no phase has built. Phase 7 chooses a route; it does not ask anyone to
    change a light.
    """
    target = phase(9)
    msg = (
        "Corridor staging is not implemented. A route is now produced by "
        "POST /api/routes/optimize, but staging one into signal advisories is a "
        "proposal to change the network's state rather than a description of it, "
        "and it requires conflicting-movement checks against junction phasing. "
        "That belongs to the decision engine."
    )
    raise NotAvailableYetError(msg, phase=target.number, phase_name=target.name)


@router.get(
    "/{emergency_id}",
    response_model=EmergencyResponseBody,
    responses=_ERRORS,
    summary="One declared response",
)
def get_emergency(
    emergency_id: str, emergencies: EmergencyRegistryDep
) -> EmergencyResponseBody:
    """Return one response, or 404."""
    return _body(emergencies.get(emergency_id), now=dt.datetime.now(tz=dt.UTC))


@router.post(
    "/{emergency_id}/status",
    response_model=EmergencyResponseBody,
    responses=_ERRORS,
    summary="Move a response along its lifecycle",
)
def transition_emergency(
    emergency_id: str,
    request: TransitionEmergencyRequest,
    emergencies: EmergencyRegistryDep,
) -> EmergencyResponseBody:
    """Change a response's status, following the lifecycle transition table.

    ``completed`` and ``cancelled`` are terminal: a further response is a new
    declaration with its own id and its own response-time clock.
    """
    now = dt.datetime.now(tz=dt.UTC)
    updated = emergencies.transition(emergency_id, request.status, at=request.at or now)
    logger.info(
        "emergency status changed",
        extra={"emergency_id": emergency_id, "status": updated.status.value},
    )
    return _body(updated, now=now)


@router.post(
    "/{emergency_id}/position",
    response_model=EmergencyResponseBody,
    responses=_ERRORS,
    summary="Update a responding unit's position",
)
def update_emergency_position(
    emergency_id: str,
    request: UpdatePositionRequest,
    emergencies: EmergencyRegistryDep,
    store: NetworkStoreDep,
) -> EmergencyResponseBody:
    """Record where the unit now is, validated against the loaded network."""
    network = store.require()
    now = dt.datetime.now(tz=dt.UTC)
    updated = emergencies.update_position(
        emergency_id, request.position, network, at=request.at or now
    )
    return _body(updated, now=now)

