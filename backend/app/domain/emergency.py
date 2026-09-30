"""Emergency response events.

An emergency is a **declared event with a destination and a priority**, not a
detection. This distinction is deliberate and safety-relevant: a camera that
recognises an ambulance silhouette has produced a low-confidence visual
observation, and acting on it alone would let a false positive trigger corridor
recommendations across a network. Visual detection is corroborating evidence
(:attr:`~app.domain.observation.ApproachObservation.emergency_vehicle_detected`);
declaration is what creates an :class:`EmergencyState`.

Lifetime is the other reason this is its own type. An observation is valid for
one instant and a network state for a few seconds; an emergency persists across
thousands of both, accumulating position updates and route revisions. Embedding
it in either would tie its lifetime to the wrong thing.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import (
    EmergencyPriority,
    EmergencyStatus,
    EmergencyVehicleType,
    FacilityType,
)
from app.domain.incident import IncidentId
from app.domain.network import FacilityId, GeoPoint, JunctionId, SegmentId
from app.domain.provenance import FloatMeasurement, Provenance

EmergencyId = Annotated[str, Field(min_length=1, max_length=64)]

#: Which facility types are valid destinations for each responder type.
#: A fire engine does not transport a patient to a hospital; an ambulance does
#: not report to a fire station mid-response. Encoding this prevents a class of
#: nonsensical routing request from ever reaching the optimiser.
VALID_FACILITY_DESTINATIONS: dict[EmergencyVehicleType, frozenset[FacilityType]] = {
    EmergencyVehicleType.AMBULANCE: frozenset(
        {FacilityType.HOSPITAL, FacilityType.TRAUMA_CENTER}
    ),
    EmergencyVehicleType.FIRE_ENGINE: frozenset({FacilityType.FIRE_STATION}),
    EmergencyVehicleType.POLICE: frozenset({FacilityType.POLICE_STATION}),
    EmergencyVehicleType.RESCUE: frozenset(
        {FacilityType.HOSPITAL, FacilityType.TRAUMA_CENTER, FacilityType.FIRE_STATION}
    ),
    EmergencyVehicleType.OTHER: frozenset(),
}


class NetworkPosition(DomainModel):
    """A position expressed against the network graph.

    Either mid-segment (segment plus offset) or at a junction. Routing needs a
    graph anchor; a bare coordinate would have to be map-matched first, and the
    result of that matching is itself uncertain and belongs in provenance.
    """

    segment_id: SegmentId | None = None
    offset_m: FloatMeasurement | None = None
    junction_id: JunctionId | None = None
    coordinate: GeoPoint | None = None
    heading_degrees: Annotated[float, Field(ge=0.0, lt=360.0)] | None = None
    observed_at: dt.datetime | None = None

    @model_validator(mode="after")
    def _require_graph_anchor(self) -> NetworkPosition:
        if self.segment_id is None and self.junction_id is None:
            msg = (
                "position must reference a segment or a junction; a coordinate "
                "alone is not routable"
            )
            raise ValueError(msg)
        if self.offset_m is not None and self.segment_id is None:
            msg = "offset_m is only meaningful together with segment_id"
            raise ValueError(msg)
        if self.observed_at is not None and self.observed_at.tzinfo is None:
            msg = "observed_at must be timezone-aware"
            raise ValueError(msg)
        return self


class EmergencyDestination(DomainModel):
    """Where the responding unit is going.

    Exactly one of facility, incident or explicit position. The three are
    genuinely different: a hospital is a fixed node with admission constraints,
    an incident is a moving target that may be recategorised or cleared, and an
    explicit position is a fallback for a dispatch that names neither.
    """

    facility_id: FacilityId | None = None
    incident_id: IncidentId | None = None
    position: NetworkPosition | None = None
    facility_type: FacilityType | None = None
    """Expected type of the referenced facility, validated against the responder
    type. Redundant with the facility record on purpose: it lets a dispatch
    request be rejected as incoherent before any network lookup happens."""

    label: Annotated[str, Field(max_length=200)] | None = None

    @model_validator(mode="after")
    def _exactly_one_target(self) -> EmergencyDestination:
        provided = [
            name
            for name, value in (
                ("facility_id", self.facility_id),
                ("incident_id", self.incident_id),
                ("position", self.position),
            )
            if value is not None
        ]
        if len(provided) != 1:
            msg = (
                "destination must specify exactly one of facility_id, incident_id "
                f"or position; got {provided or 'none'}"
            )
            raise ValueError(msg)
        if self.facility_type is not None and self.facility_id is None:
            msg = "facility_type is only meaningful together with facility_id"
            raise ValueError(msg)
        return self


class EmergencyVehicle(DomainModel):
    """The responding unit itself."""

    vehicle_id: Annotated[str, Field(min_length=1, max_length=64)]
    vehicle_type: EmergencyVehicleType
    callsign: Annotated[str, Field(max_length=50)] | None = None
    height_m: FloatMeasurement | None = None
    weight_t: FloatMeasurement | None = None
    """Checked against segment physical restrictions. A ladder truck excluded by
    a low bridge is a hard constraint that has nothing to do with traffic."""

    requires_wide_turns: bool | None = None


class EmergencyState(DomainModel):
    """A declared emergency response, tracked from declaration to completion."""

    emergency_id: EmergencyId
    vehicle: EmergencyVehicle
    priority: EmergencyPriority
    status: EmergencyStatus

    declared_at: dt.datetime
    """When the response was declared. The origin of the response-time clock;
    every ETA claim is measured against this, not against when this system
    happened to compute a route."""

    updated_at: dt.datetime
    completed_at: dt.datetime | None = None

    origin: NetworkPosition | None = None
    current_position: NetworkPosition
    destination: EmergencyDestination

    triggering_incident_id: IncidentId | None = None
    provenance: Provenance
    """Where the declaration came from: dispatch record, dataset, operator entry
    or scenario definition. A scenario-declared emergency must be distinguishable
    from a real dispatch record in every downstream artifact."""

    assigned_route_id: Annotated[str, Field(max_length=64)] | None = None
    """Set once a route has been recommended and accepted. Advisory: this system
    recommends, an operator decides."""

    notes: Annotated[str, Field(max_length=1000)] | None = None

    @model_validator(mode="after")
    def _validate(self) -> EmergencyState:
        for name in ("declared_at", "updated_at", "completed_at"):
            value: dt.datetime | None = getattr(self, name)
            if value is not None and value.tzinfo is None:
                msg = f"{name} must be timezone-aware"
                raise ValueError(msg)
        if self.updated_at < self.declared_at:
            msg = "updated_at cannot precede declared_at"
            raise ValueError(msg)
        if self.status is EmergencyStatus.COMPLETED and self.completed_at is None:
            msg = "an emergency with status COMPLETED must record completed_at"
            raise ValueError(msg)

        # Responder type must match destination class.
        want = self.destination.facility_type
        if want is not None:
            allowed = VALID_FACILITY_DESTINATIONS.get(self.vehicle.vehicle_type, frozenset())
            if allowed and want not in allowed:
                msg = (
                    f"{self.vehicle.vehicle_type.value} cannot be dispatched to a "
                    f"{want.value}; valid facility destinations are "
                    f"{sorted(f.value for f in allowed)}"
                )
                raise ValueError(msg)

        # A fire engine or police unit responding to an event should be routed to
        # the event, not to a station. Enforced as coherence between the
        # triggering incident and the destination when both are present.
        if (
            self.triggering_incident_id is not None
            and self.destination.incident_id is not None
            and self.destination.incident_id != self.triggering_incident_id
        ):
            msg = (
                "destination incident does not match the triggering incident; if the "
                "unit is being redirected, that is a new response, not an edit"
            )
            raise ValueError(msg)

        return self

    @property
    def is_active(self) -> bool:
        """Whether this response is still in progress and needs routing."""
        return self.status in (
            EmergencyStatus.DECLARED,
            EmergencyStatus.EN_ROUTE,
        )

    @property
    def warrants_corridor_preemption(self) -> bool:
        """Whether the priority tier justifies recommending corridor pre-emption.

        Only P1. Recommending pre-emption across a corridor imposes real delay
        and real risk on every conflicting movement, so it is reserved for
        immediately life-threatening responses. Lower tiers still receive route
        optimisation -- they simply do not warrant disrupting the network.
        """
        return self.priority is EmergencyPriority.P1_IMMEDIATE and self.is_active

    def elapsed_response_time_s(self, now: dt.datetime) -> float:
        """Seconds since declaration, on the caller's clock.

        Takes ``now`` as an argument rather than reading the wall clock: under
        recorded replay the correct 'now' is media time, and a domain object must
        not decide which clock it is being evaluated against.
        """
        if now.tzinfo is None:
            msg = "now must be timezone-aware"
            raise ValueError(msg)
        return (now - self.declared_at).total_seconds()
