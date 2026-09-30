"""Incident declaration and lifecycle API.

Phase 6 scope: **what is disrupting the network, and how far it reaches**.
Declaring incidents, resolving exclusion zones onto graph elements, and moving
an incident along its lifecycle to clearance.

Every incident is validated against the loaded network
------------------------------------------------------
An incident naming a segment the map does not have is rejected. It would prune
nothing and penalise nothing while appearing, in every listing and on every
dashboard, to be an active blockage -- the failure mode where a safety
constraint quietly stops constraining.

An exclusion zone is resolved once, here
----------------------------------------
A zone is declared as a centre and a radius, but route pruning cannot afford a
geometric query in its inner loop. The zone is resolved into the set of segments
and junctions it covers at declaration, and pruning becomes a set membership
test. Anything that could not be tested -- an element with no surveyed position
-- is reported rather than skipped, because a zone that looks resolved and
covers less than it claims is worse than one that admits the gap.

Nothing here plans a route
--------------------------
This phase establishes what a route will have to avoid. Producing one is
Phase 7.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import Field

from app.api.deps import IncidentRegistryDep, NetworkStoreDep, SettingsDep
from app.api.schemas import ApiModel, ErrorResponse
from app.core.config import Settings
from app.core.errors import ValidationFailedError
from app.core.logging import get_logger
from app.domain.enums import IncidentSeverity, IncidentStatus, IncidentType
from app.domain.incident import ExclusionZone, Incident, IncidentLocation
from app.domain.network import GeoPoint, RoadNetwork
from app.domain.provenance import FloatMeasurement, Provenance, Unit, ValueSource
from app.emergency.incidents import INCIDENT_TRANSITIONS
from app.emergency.zones import ZoneResolution, resolve_zone

logger = get_logger(__name__)
router = APIRouter(prefix="/incidents", tags=["incidents"])

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "No network loaded, or unknown incident"},
    422: {"model": ErrorResponse, "description": "Incident does not fit the loaded network"},
}

#: Sources an operator-facing declaration may claim. ``DERIVED`` and
#: ``PREDICTED`` are excluded: they carry obligations (input refs, a model) that
#: an HTTP caller is in no position to satisfy honestly.
_DECLARABLE_SOURCES = frozenset(
    {ValueSource.OBSERVED, ValueSource.ASSUMED, ValueSource.STATIC, ValueSource.SIMULATED}
)


class ExclusionZoneRequest(ApiModel):
    """An exclusion zone, either resolved from a centre or given explicitly."""

    radius_m: Annotated[float, Field(gt=0)]
    reason: Annotated[str, Field(min_length=1, max_length=300)]

    centre: GeoPoint | None = None
    """Zone centre. When given, the covered elements are resolved against the
    loaded network's geometry."""

    excluded_segment_ids: frozenset[str] = frozenset()
    excluded_junction_ids: frozenset[str] = frozenset()
    """Explicit coverage, for a network without surveyed coordinates. Unioned
    with anything resolved from ``centre``."""


class DeclareIncidentRequest(ApiModel):
    """A new incident.

    The server assigns the id and stamps provenance when they are not supplied,
    so a caller cannot accidentally create an incident that claims to have come
    from somewhere it did not.
    """

    incident_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.REPORTED
    location: IncidentLocation

    reported_at: dt.datetime | None = None
    estimated_clearance_at: dt.datetime | None = None
    """A prediction, presented as an estimate wherever it is rendered. Never a
    scheduled fact."""

    blocks_segment_ids: frozenset[str] = frozenset()
    """Segments rendered impassable. Pruned before search, never priced."""

    affected_segment_ids: frozenset[str] = frozenset()
    """Degraded but passable. Priced, never pruned."""

    exclusion_zone: ExclusionZoneRequest | None = None
    description: Annotated[str, Field(max_length=1000)] | None = None

    reported_by: Annotated[str, Field(min_length=1, max_length=200)] = "operator"
    """Who or what reported this. Recorded as the provenance producer, because a
    CV-detected accident is weaker evidence than a confirmed dispatch record and
    downstream confidence must be able to tell them apart."""

    source: ValueSource = ValueSource.OBSERVED
    source_notes: Annotated[str, Field(max_length=1000)] | None = None
    """Required when ``source`` is ``assumed``: an unjustified assumption is
    indistinguishable from a fabrication."""


class TransitionIncidentRequest(ApiModel):
    """A lifecycle change."""

    status: IncidentStatus
    at: dt.datetime | None = None
    note: Annotated[str, Field(max_length=1000)] | None = None


class IncidentResponse(ApiModel):
    """One incident, with how its exclusion zone was resolved."""

    incident: Incident
    zone_resolution: ZoneResolution | None = None
    """Present when a zone was resolved from a centre at declaration. Retained
    so the coverage -- and any element that could not be tested -- stays
    visible after the fact."""

    permitted_next_statuses: tuple[IncidentStatus, ...] = ()


class IncidentListResponse(ApiModel):
    """Declared incidents."""

    incidents: tuple[Incident, ...] = ()
    count: int = 0
    active_count: int = 0
    network_id: str | None = None
    note: str = ""


def _permitted_next(status: IncidentStatus) -> tuple[IncidentStatus, ...]:
    """Statuses this incident may move to next."""
    return tuple(sorted(INCIDENT_TRANSITIONS.get(status, frozenset()), key=lambda s: s.value))


@router.get("", response_model=IncidentListResponse, summary="Declared incidents")
def list_incidents(
    incidents: IncidentRegistryDep,
    store: NetworkStoreDep,
    active_only: Annotated[bool, Query()] = False,
) -> IncidentListResponse:
    """Return declared incidents, newest report first.

    Answers 200 with an empty list when nothing has been declared. An incident
    exists only because someone reported it, so "none declared" is a complete
    and true description of our records rather than an absence of knowledge.
    """
    items = incidents.list(active_only=active_only)
    all_items = incidents.list()
    source = store.source
    return IncidentListResponse(
        incidents=items,
        count=len(items),
        active_count=sum(1 for item in all_items if item.is_active),
        network_id=source.network_id if source else None,
        note=(
            "Incidents are held in memory for this process and are discarded when a "
            "different network is loaded, because their segment and junction ids "
            "refer to the network they were declared against."
        ),
    )


@router.post(
    "",
    response_model=IncidentResponse,
    status_code=201,
    responses=_ERRORS,
    summary="Declare an incident",
)
def declare_incident(
    request: DeclareIncidentRequest,
    incidents: IncidentRegistryDep,
    store: NetworkStoreDep,
    settings: SettingsDep,
) -> IncidentResponse:
    """Declare a new incident against the loaded network."""
    network = store.require()
    now = dt.datetime.now(tz=dt.UTC)
    reported_at = request.reported_at or now

    if request.source not in _DECLARABLE_SOURCES:
        msg = (
            f"source {request.source.value!r} cannot be claimed by a declaration; "
            f"choose one of {sorted(item.value for item in _DECLARABLE_SOURCES)}."
        )
        raise ValidationFailedError(msg, source=request.source.value)

    if request.status in (IncidentStatus.CLEARED, IncidentStatus.FALSE_ALARM):
        msg = (
            f"An incident cannot be declared as {request.status.value}: it would be "
            "terminal on arrival and could never have constrained anything. Declare "
            "it, then transition it."
        )
        raise ValidationFailedError(msg, status=request.status.value)

    zone, resolution = _build_zone(request, network, settings, now=now)

    incident = Incident(
        incident_id=request.incident_id or f"incident-{uuid.uuid4().hex[:12]}",
        incident_type=request.incident_type,
        severity=request.severity,
        status=request.status,
        location=request.location,
        reported_at=reported_at,
        confirmed_at=now if request.status is IncidentStatus.CONFIRMED else None,
        estimated_clearance_at=request.estimated_clearance_at,
        provenance=Provenance(
            source=request.source,
            producer=request.reported_by,
            producer_version="1.0",
            observed_at=reported_at,
            computed_at=now,
            notes=request.source_notes,
        ),
        blocks_segment_ids=request.blocks_segment_ids,
        affected_segment_ids=request.affected_segment_ids,
        exclusion_zone=zone,
        description=request.description,
    )

    stored = incidents.declare(incident, network)
    logger.info(
        "incident declared",
        extra={
            "incident_id": stored.incident_id,
            "incident_type": stored.incident_type.value,
            "severity": stored.severity.value,
            "blocked_segments": len(stored.hard_blocked_segment_ids),
        },
    )
    return IncidentResponse(
        incident=stored,
        zone_resolution=resolution,
        permitted_next_statuses=_permitted_next(stored.status),
    )


def _build_zone(
    request: DeclareIncidentRequest,
    network: RoadNetwork,
    settings: Settings,
    *,
    now: dt.datetime,
) -> tuple[ExclusionZone | None, ZoneResolution | None]:
    """Resolve the declared exclusion zone, if there is one."""
    spec = request.exclusion_zone
    if spec is None:
        return None, None

    if spec.radius_m > settings.exclusion_zone_max_radius_m:
        msg = (
            f"Exclusion radius {spec.radius_m} m exceeds the configured maximum of "
            f"{settings.exclusion_zone_max_radius_m} m. A hard constraint cannot be "
            "outbid once applied, so an implausible radius is rejected here rather "
            "than discovered when no route can be found."
        )
        raise ValidationFailedError(msg, radius_m=spec.radius_m)

    resolution: ZoneResolution | None = None
    segment_ids = set(spec.excluded_segment_ids)
    junction_ids = set(spec.excluded_junction_ids)

    if spec.centre is not None:
        resolution = resolve_zone(network, centre=spec.centre, radius_m=spec.radius_m)
        segment_ids |= set(resolution.excluded_segment_ids)
        junction_ids |= set(resolution.excluded_junction_ids)
    elif not segment_ids and not junction_ids:
        msg = (
            "An exclusion zone needs either a centre to resolve from or an explicit "
            "set of excluded elements. A zone covering nothing excludes nothing, "
            "which is never what a declaration meant."
        )
        raise ValidationFailedError(msg, radius_m=spec.radius_m)

    zone = ExclusionZone(
        radius_m=FloatMeasurement(
            value=spec.radius_m,
            unit=Unit.METER,
            provenance=Provenance(
                source=request.source,
                producer=request.reported_by,
                producer_version="1.0",
                observed_at=now,
                computed_at=now,
                notes=request.source_notes,
            ),
        ),
        excluded_segment_ids=frozenset(segment_ids),
        excluded_junction_ids=frozenset(junction_ids),
        reason=spec.reason,
    )
    return zone, resolution


@router.get(
    "/{incident_id}",
    response_model=IncidentResponse,
    responses=_ERRORS,
    summary="One incident",
)
def get_incident(incident_id: str, incidents: IncidentRegistryDep) -> IncidentResponse:
    """Return one incident, or 404."""
    incident = incidents.get(incident_id)
    return IncidentResponse(
        incident=incident, permitted_next_statuses=_permitted_next(incident.status)
    )


@router.post(
    "/{incident_id}/status",
    response_model=IncidentResponse,
    responses=_ERRORS,
    summary="Move an incident along its lifecycle",
)
def transition_incident(
    incident_id: str,
    request: TransitionIncidentRequest,
    incidents: IncidentRegistryDep,
) -> IncidentResponse:
    """Change an incident's status, following the lifecycle transition table.

    ``cleared`` and ``false_alarm`` are terminal. Reopening a cleared incident
    would make "was this segment blocked at 14:05?" unanswerable from the
    record, because one id would have held both answers; a recurrence is a new
    incident.
    """
    updated = incidents.transition(
        incident_id,
        request.status,
        at=request.at or dt.datetime.now(tz=dt.UTC),
        note=request.note,
    )
    logger.info(
        "incident status changed",
        extra={"incident_id": incident_id, "status": updated.status.value},
    )
    return IncidentResponse(
        incident=updated, permitted_next_statuses=_permitted_next(updated.status)
    )
