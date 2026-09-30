"""Incidents: disruptive events on the network.

An incident is not the same thing as an emergency response. A fire is an
incident; the fire engine dispatched to it is an emergency (see
:mod:`app.domain.emergency`). One incident may produce several responses, and a
response may exist without any incident record. Modelling them as one type
forces every consumer to disambiguate at read time, and makes "multiple
simultaneous incidents" -- an explicit requirement -- awkward to express.

Incidents matter to routing in two distinct ways, and the distinction is the
hard/soft safety tiering:

* An **exclusion zone** around an active fire makes segments *inadmissible*.
  This prunes the graph before search.
* A minor accident on the shoulder *degrades* a segment. This adds a penalty,
  in equivalent seconds, and raises a warning.

Which of the two applies is decided by the versioned safety rule catalog, not
hard-coded here, so that the policy can be revised and audited independently of
the incident data model.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import IncidentSeverity, IncidentStatus, IncidentType
from app.domain.network import GeoPoint, JunctionId, SegmentId
from app.domain.provenance import FloatMeasurement, IntMeasurement, Provenance

IncidentId = Annotated[str, Field(min_length=1, max_length=64)]


class IncidentLocation(DomainModel):
    """Where an incident is, expressed against the network.

    At least one of segment or junction must be given. A bare coordinate is not
    enough: routing operates on the graph, and an incident that cannot be mapped
    to a graph element cannot prune or penalise anything. If only a coordinate
    is available, map-matching happens at ingestion, and the result carries
    ``ASSUMED`` provenance so the uncertainty stays visible.
    """

    segment_id: SegmentId | None = None
    offset_m: FloatMeasurement | None = None
    """Distance along the segment from its upstream junction."""

    junction_id: JunctionId | None = None
    position: GeoPoint | None = None
    description: Annotated[str, Field(max_length=300)] | None = None

    @model_validator(mode="after")
    def _require_graph_anchor(self) -> IncidentLocation:
        if self.segment_id is None and self.junction_id is None:
            msg = (
                "incident location must reference a segment or a junction; "
                "a coordinate alone cannot be reasoned about on the graph"
            )
            raise ValueError(msg)
        if self.offset_m is not None and self.segment_id is None:
            msg = "offset_m is only meaningful together with segment_id"
            raise ValueError(msg)
        return self


class ExclusionZone(DomainModel):
    """An area that emergency and general traffic must not enter.

    Produced by incidents such as an active fire, a chemical spill or a
    structural collapse. Its effect is absolute: affected segments are removed
    from the search graph, never merely penalised. A route that is "mostly
    outside" a fire exclusion zone is not a safer route, it is an unsafe one.
    """

    radius_m: FloatMeasurement
    excluded_segment_ids: frozenset[str] = frozenset()
    """Segments determined to fall inside the zone. Resolved at ingestion so
    that route pruning is a set membership test rather than a geometric query in
    the inner loop of the search."""

    excluded_junction_ids: frozenset[str] = frozenset()
    reason: Annotated[str, Field(min_length=1, max_length=300)]


class Incident(DomainModel):
    """A disruptive event affecting the network."""

    incident_id: IncidentId
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus
    location: IncidentLocation

    reported_at: dt.datetime
    confirmed_at: dt.datetime | None = None
    cleared_at: dt.datetime | None = None
    estimated_clearance_at: dt.datetime | None = None
    """A prediction. Carries the provenance of whatever produced it and must be
    presented as an estimate, never as a scheduled fact."""

    provenance: Provenance
    """Where the incident report came from: dataset column, operator report,
    CV-based detection. A CV-detected accident is weaker evidence than a
    confirmed dispatch record, and downstream confidence must reflect that."""

    blocks_segment_ids: frozenset[str] = frozenset()
    """Segments rendered impassable. A hard constraint: these are pruned."""

    blocked_lane_count: IntMeasurement | None = None
    """Partial obstruction. A capacity reduction, therefore a soft penalty --
    materially different from a full blockage and modelled differently."""

    exclusion_zone: ExclusionZone | None = None
    affected_segment_ids: frozenset[str] = frozenset()
    """Degraded but passable. Contributes penalties, not pruning."""

    description: Annotated[str, Field(max_length=1000)] | None = None

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> Incident:
        if self.reported_at.tzinfo is None:
            msg = "reported_at must be timezone-aware"
            raise ValueError(msg)
        for name in ("confirmed_at", "cleared_at", "estimated_clearance_at"):
            value: dt.datetime | None = getattr(self, name)
            if value is not None and value.tzinfo is None:
                msg = f"{name} must be timezone-aware"
                raise ValueError(msg)
        if self.cleared_at is not None and self.cleared_at < self.reported_at:
            msg = "cleared_at cannot precede reported_at"
            raise ValueError(msg)
        if self.status is IncidentStatus.CLEARED and self.cleared_at is None:
            msg = "an incident with status CLEARED must record cleared_at"
            raise ValueError(msg)
        overlap = self.blocks_segment_ids & self.affected_segment_ids
        if overlap:
            msg = (
                f"segments {sorted(overlap)} are listed as both fully blocked and "
                "merely affected; a segment is one or the other, and the ambiguity "
                "would change whether it is pruned or penalised"
            )
            raise ValueError(msg)
        return self

    @property
    def is_active(self) -> bool:
        """Whether this incident still constrains the network."""
        return self.status not in (IncidentStatus.CLEARED, IncidentStatus.FALSE_ALARM)

    @property
    def hard_blocked_segment_ids(self) -> frozenset[str]:
        """All segments this incident makes inadmissible, blockage plus exclusion."""
        if not self.is_active:
            return frozenset()
        zone = self.exclusion_zone.excluded_segment_ids if self.exclusion_zone else frozenset()
        return self.blocks_segment_ids | zone
