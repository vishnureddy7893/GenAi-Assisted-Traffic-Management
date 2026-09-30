"""Static road network topology.

Static vs dynamic
-----------------
This module contains **only** things that do not change between snapshots: where
the junctions are, how the segments connect them, how long each segment is, how
many lanes it has, which turns are physically permitted.

Live quantities -- queues, speeds, blockages, signal indications -- live in
:mod:`app.domain.state`, keyed by the identifiers defined here.

Keeping them apart is not tidiness. A ``RoadNetwork`` is loaded once and shared
by every request; a ``NetworkState`` is a per-instant snapshot. If congestion
were a mutable field on ``RoadSegment``, two concurrent emergency decisions
would race on the same object, and replaying a decision against the state that
produced it would be impossible.

Why movements are first-class
-----------------------------
A junction modelled as a bare graph node cannot express the two facts this
system exists to reason about: "the left-turn approach is the congested one" and
"you may not turn right here". Both are properties of a *movement*, not of a
node. Modelling :class:`TurnMovement` now means Phase 7 can expand the graph to
a movement graph -- the standard formulation for turn-aware and turn-penalised
routing -- without breaking a single schema.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import (
    DrivingSide,
    EmergencyVehicleType,
    FacilityType,
    JunctionKind,
    MovementType,
    RoadClass,
)
from app.domain.provenance import FloatMeasurement, Provenance

# Identifier aliases. Distinct names make signatures self-documenting and let a
# future migration to typed ids happen in one place.
JunctionId = Annotated[str, Field(min_length=1, max_length=64)]
SegmentId = Annotated[str, Field(min_length=1, max_length=64)]
MovementId = Annotated[str, Field(min_length=1, max_length=64)]
FacilityId = Annotated[str, Field(min_length=1, max_length=64)]
ApproachId = Annotated[str, Field(min_length=1, max_length=64)]
SignalGroupId = Annotated[str, Field(min_length=1, max_length=64)]


class GeoPoint(DomainModel):
    """WGS-84 coordinate."""

    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]


class PhysicalRestrictions(DomainModel):
    """Permanent physical limits on a segment.

    Distinct from a temporary blockage (which is dynamic state): a low bridge is
    a property of the map. It matters because a fire engine may be excluded from
    a segment an ambulance can use, so emergency admissibility is
    vehicle-type-dependent even before any incident occurs.
    """

    max_height_m: FloatMeasurement | None = None
    max_weight_t: FloatMeasurement | None = None
    max_width_m: FloatMeasurement | None = None
    excluded_vehicle_types: frozenset[EmergencyVehicleType] = frozenset()
    """Emergency vehicle types that can never traverse this segment."""

    notes: Annotated[str, Field(max_length=500)] | None = None


class RoadSegment(DomainModel):
    """A directed road segment: the edge of the network graph.

    Directed, not undirected. A two-way road is two segments. Modelling it as one
    undirected edge makes it impossible to express one-way streets, direction-
    specific queues, or a contraflow emergency corridor -- and every one of those
    matters for emergency routing.
    """

    segment_id: SegmentId
    from_junction_id: JunctionId
    to_junction_id: JunctionId
    name: Annotated[str, Field(max_length=200)] | None = None
    road_class: RoadClass = RoadClass.UNKNOWN

    length_m: FloatMeasurement
    """Centreline length. Required: no route cost can be computed without it, and
    a defaulted length would silently distort every ETA on the network."""

    lane_count: Annotated[int, Field(ge=1, le=12)]

    free_flow_speed_kph: FloatMeasurement | None = None
    """Observed or surveyed free-flow speed. Not the posted speed limit -- the
    two differ materially, and using the limit as if it were measured behaviour
    is a fabrication."""

    speed_limit_kph: FloatMeasurement | None = None

    capacity_veh_per_hour: FloatMeasurement | None = None
    """Design or estimated capacity for the whole segment (all lanes). Left
    unset until it is genuinely calibrated; a guessed capacity poisons every
    volume-to-capacity ratio computed from it."""

    physical_restrictions: PhysicalRestrictions = PhysicalRestrictions()
    geometry: tuple[GeoPoint, ...] = ()
    """Optional polyline for display. Not used for distance -- ``length_m`` is
    authoritative, because a sparse polyline understates true length."""

    @model_validator(mode="after")
    def _reject_self_loop(self) -> RoadSegment:
        if self.from_junction_id == self.to_junction_id:
            msg = (
                f"segment {self.segment_id!r} starts and ends at "
                f"{self.from_junction_id!r}; self-loops are not a valid road segment"
            )
            raise ValueError(msg)
        return self

    def permits(self, vehicle_type: EmergencyVehicleType) -> bool:
        """Whether the segment's permanent restrictions admit this vehicle type.

        Physical admissibility only. Temporary blockages and incident exclusion
        zones are evaluated against :mod:`app.domain.safety` using dynamic state.
        """
        return vehicle_type not in self.physical_restrictions.excluded_vehicle_types


class Lane(DomainModel):
    """One lane on an approach, with the movements it serves."""

    index: Annotated[int, Field(ge=0)]
    """Zero-based index from the kerb on the driving side."""

    permitted_movements: frozenset[MovementType]
    is_exclusive_turn_lane: bool = False


class Approach(DomainModel):
    """One incoming direction at a junction.

    This is the unit the specification calls a "lane" in phrases like *"Junction
    11 Lane 1 is fully congested"*. Queues, delays and detector zones are all
    per-approach, never per-junction: a junction with a blocked left turn and a
    free through movement has no single meaningful queue length.
    """

    approach_id: ApproachId
    incoming_segment_id: SegmentId
    lanes: tuple[Lane, ...] = ()
    bearing_degrees: Annotated[float, Field(ge=0.0, lt=360.0)] | None = None
    """Compass bearing of approaching traffic; used to label an approach as
    "northbound" in generated narrative without inventing geography."""

    has_pedestrian_crossing: bool = False
    crossing_is_signalized: bool | None = None
    """``None`` means unsurveyed, which is not the same as ``False``."""


class TurnMovement(DomainModel):
    """A permitted (or explicitly prohibited) turn at a junction.

    The ``(junction, from_segment, to_segment)`` triple is the unit of routing in
    a movement-expanded graph and the unit of conflict in signal safety logic.
    """

    movement_id: MovementId
    junction_id: JunctionId
    from_segment_id: SegmentId
    to_segment_id: SegmentId
    movement_type: MovementType
    is_permitted: bool = True
    """``False`` records a banned turn explicitly. Recording the prohibition is
    better than omitting the movement: omission is indistinguishable from an
    incomplete map import, and "unknown" must not be read as "allowed"."""

    signal_group_id: SignalGroupId | None = None
    """Signal group controlling this movement at a signalized junction."""

    conflicting_movement_ids: frozenset[MovementId] = frozenset()
    """Movements that must not run concurrently. Populated from the junction's
    signal phasing when available; the basis of the conflicting-movement safety
    check for an emergency corridor recommendation."""


class Facility(DomainModel):
    """An emergency destination attached to the network.

    An ambulance's destination is a hospital, not a junction. Modelling the
    facility separately, with its own access junction, is what lets the router
    answer "route to Hospital A" rather than requiring the caller to already know
    which node the hospital hangs off.
    """

    facility_id: FacilityId
    name: Annotated[str, Field(min_length=1, max_length=200)]
    facility_type: FacilityType
    access_junction_id: JunctionId
    """Junction through which the facility is entered."""

    position: GeoPoint | None = None
    accepts_emergency_types: frozenset[EmergencyVehicleType] = frozenset()
    """Which responder types this facility is a valid destination for. A fire
    station is not a valid destination for an ambulance carrying a patient."""

    notes: Annotated[str, Field(max_length=500)] | None = None


class Junction(DomainModel):
    """A node of the network graph."""

    junction_id: JunctionId
    name: Annotated[str, Field(max_length=200)] | None = None
    kind: JunctionKind = JunctionKind.UNSIGNALIZED
    position: GeoPoint | None = None
    approaches: tuple[Approach, ...] = ()
    movements: tuple[TurnMovement, ...] = ()

    @model_validator(mode="after")
    def _movements_belong_here(self) -> Junction:
        bad = [m.movement_id for m in self.movements if m.junction_id != self.junction_id]
        if bad:
            msg = (
                f"junction {self.junction_id!r} lists movements belonging to another "
                f"junction: {bad}"
            )
            raise ValueError(msg)
        return self


class RoadNetwork(DomainModel):
    """Immutable topology of the whole modelled area.

    Referential integrity is validated at construction rather than trusted.
    A dangling segment reference does not fail at import time if unchecked -- it
    fails much later, inside route relaxation, as a ``KeyError`` on an emergency
    request. Failing at load is strictly better.
    """

    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    name: Annotated[str, Field(max_length=200)] | None = None
    version: Annotated[str, Field(min_length=1, max_length=50)]
    """Topology version. Recorded in every decision so a recommendation can be
    replayed against the map that produced it."""

    driving_side: DrivingSide
    """Explicit, because it determines which turn crosses opposing traffic and
    therefore which movements conflict."""

    provenance: Provenance
    """Where the topology came from -- OSM extract, manual survey, dataset
    metadata. A hand-drawn network carries ``ASSUMED`` and says so."""

    junctions: tuple[Junction, ...]
    segments: tuple[RoadSegment, ...]
    facilities: tuple[Facility, ...] = ()

    @model_validator(mode="after")
    def _validate_referential_integrity(self) -> RoadNetwork:
        junction_ids = {j.junction_id for j in self.junctions}
        segment_ids = {s.segment_id for s in self.segments}
        by_segment = {s.segment_id: s for s in self.segments}

        if len(junction_ids) != len(self.junctions):
            msg = "duplicate junction_id in network"
            raise ValueError(msg)
        if len(segment_ids) != len(self.segments):
            msg = "duplicate segment_id in network"
            raise ValueError(msg)

        for seg in self.segments:
            missing = {seg.from_junction_id, seg.to_junction_id} - junction_ids
            if missing:
                msg = f"segment {seg.segment_id!r} references unknown junction(s) {sorted(missing)}"
                raise ValueError(msg)

        for jn in self.junctions:
            for ap in jn.approaches:
                if ap.incoming_segment_id not in segment_ids:
                    msg = (
                        f"approach {ap.approach_id!r} at junction {jn.junction_id!r} "
                        f"references unknown segment {ap.incoming_segment_id!r}"
                    )
                    raise ValueError(msg)
                if by_segment[ap.incoming_segment_id].to_junction_id != jn.junction_id:
                    msg = (
                        f"approach {ap.approach_id!r} at junction {jn.junction_id!r} uses "
                        f"segment {ap.incoming_segment_id!r}, which does not terminate here"
                    )
                    raise ValueError(msg)

            for mv in jn.movements:
                unknown = {mv.from_segment_id, mv.to_segment_id} - segment_ids
                if unknown:
                    msg = (
                        f"movement {mv.movement_id!r} references unknown "
                        f"segment(s) {sorted(unknown)}"
                    )
                    raise ValueError(msg)
                if by_segment[mv.from_segment_id].to_junction_id != jn.junction_id:
                    msg = (
                        f"movement {mv.movement_id!r} enters from segment "
                        f"{mv.from_segment_id!r}, which does not arrive at {jn.junction_id!r}"
                    )
                    raise ValueError(msg)
                if by_segment[mv.to_segment_id].from_junction_id != jn.junction_id:
                    msg = (
                        f"movement {mv.movement_id!r} exits to segment "
                        f"{mv.to_segment_id!r}, which does not depart from {jn.junction_id!r}"
                    )
                    raise ValueError(msg)

        for fac in self.facilities:
            if fac.access_junction_id not in junction_ids:
                msg = (
                    f"facility {fac.facility_id!r} references unknown access junction "
                    f"{fac.access_junction_id!r}"
                )
                raise ValueError(msg)

        return self

    def build_index(self) -> NetworkIndex:
        """Build the derived lookup/adjacency structure used by graph search.

        Deliberately *not* cached on the model. The network schema is an
        immutable value; an index is a derived acceleration structure with a
        different lifetime, and mutating a frozen model to memoise it is the kind
        of shortcut that turns into a concurrency bug. Routing builds the index
        once per network load and holds it.
        """
        return NetworkIndex.from_network(self)


class NetworkIndex:
    """Adjacency and id lookups derived from a :class:`RoadNetwork`.

    Build once per loaded network. Route relaxation runs in the inner loop of
    every emergency request, so the successor set must be an O(1) lookup rather
    than a scan over all segments.
    """

    __slots__ = ("_by_junction", "_by_segment", "_facilities", "_incoming", "_outgoing")

    def __init__(
        self,
        *,
        by_junction: dict[str, Junction],
        by_segment: dict[str, RoadSegment],
        facilities: dict[str, Facility],
        outgoing: dict[str, tuple[RoadSegment, ...]],
        incoming: dict[str, tuple[RoadSegment, ...]],
    ) -> None:
        self._by_junction = by_junction
        self._by_segment = by_segment
        self._facilities = facilities
        self._outgoing = outgoing
        self._incoming = incoming

    @classmethod
    def from_network(cls, network: RoadNetwork) -> NetworkIndex:
        """Construct an index from a validated network."""
        outgoing: dict[str, list[RoadSegment]] = {j.junction_id: [] for j in network.junctions}
        incoming: dict[str, list[RoadSegment]] = {j.junction_id: [] for j in network.junctions}
        for seg in network.segments:
            outgoing[seg.from_junction_id].append(seg)
            incoming[seg.to_junction_id].append(seg)
        return cls(
            by_junction={j.junction_id: j for j in network.junctions},
            by_segment={s.segment_id: s for s in network.segments},
            facilities={f.facility_id: f for f in network.facilities},
            outgoing={k: tuple(v) for k, v in outgoing.items()},
            incoming={k: tuple(v) for k, v in incoming.items()},
        )

    def junction(self, junction_id: str) -> Junction:
        """Return a junction by id, raising ``KeyError`` if absent."""
        return self._by_junction[junction_id]

    def segment(self, segment_id: str) -> RoadSegment:
        """Return a segment by id, raising ``KeyError`` if absent."""
        return self._by_segment[segment_id]

    def facility(self, facility_id: str) -> Facility:
        """Return a facility by id, raising ``KeyError`` if absent."""
        return self._facilities[facility_id]

    def outgoing_segments(self, junction_id: str) -> tuple[RoadSegment, ...]:
        """Segments departing a junction: the successor set for graph search."""
        return self._outgoing.get(junction_id, ())

    def incoming_segments(self, junction_id: str) -> tuple[RoadSegment, ...]:
        """Segments arriving at a junction."""
        return self._incoming.get(junction_id, ())

    @property
    def junction_ids(self) -> frozenset[str]:
        """All junction ids in the indexed network."""
        return frozenset(self._by_junction)

    @property
    def segment_ids(self) -> frozenset[str]:
        """All segment ids in the indexed network."""
        return frozenset(self._by_segment)
