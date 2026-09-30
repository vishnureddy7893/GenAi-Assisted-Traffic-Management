"""Builders for synthetic road networks used by the Phase 6 tests.

Networks are built programmatically rather than loaded from a fixture file, so a
test can state the exact geometry its assertion depends on -- a segment that
crosses an exclusion zone with both endpoints outside it, a junction with no
surveyed position -- instead of relying on a shared file that some other test
also depends on.
"""

from __future__ import annotations

import datetime as dt

from app.domain.enums import (
    DrivingSide,
    EmergencyVehicleType,
    FacilityType,
    JunctionKind,
    MovementType,
    RoadClass,
)
from app.domain.network import (
    Approach,
    Facility,
    GeoPoint,
    Junction,
    Lane,
    RoadNetwork,
    RoadSegment,
    TurnMovement,
)
from app.domain.provenance import FloatMeasurement, Provenance, Unit, ValueSource

T0 = dt.datetime(2024, 3, 14, 8, 30, tzinfo=dt.UTC)

PROV = Provenance(
    source=ValueSource.ASSUMED,
    producer="tests.network_fixtures",
    producer_version="1.0",
    observed_at=T0,
    computed_at=T0,
    notes="Synthetic test network. Every coordinate and length is invented.",
)


def measure(value: float, unit: Unit = Unit.METER) -> FloatMeasurement:
    """A measurement carrying the fixture provenance."""
    return FloatMeasurement(value=value, unit=unit, provenance=PROV)


def make_segment(
    segment_id: str,
    from_junction_id: str,
    to_junction_id: str,
    *,
    length_m: float = 500.0,
    lane_count: int = 2,
    geometry: tuple[GeoPoint, ...] = (),
    excluded_vehicle_types: frozenset[EmergencyVehicleType] = frozenset(),
    free_flow_speed_kph: float | None = None,
) -> RoadSegment:
    """One directed segment.

    ``free_flow_speed_kph`` is ``None`` by default, which is what a segment with
    no surveyed speed looks like: it has a length and nothing to divide it by, so
    no traversal time can be derived. Tests that need a timed route pass a speed
    explicitly, and tests of the no-time-basis path leave it out.
    """
    from app.domain.network import PhysicalRestrictions

    return RoadSegment(
        segment_id=segment_id,
        from_junction_id=from_junction_id,
        to_junction_id=to_junction_id,
        road_class=RoadClass.ARTERIAL,
        length_m=measure(length_m),
        lane_count=lane_count,
        free_flow_speed_kph=(
            measure(free_flow_speed_kph, Unit.KILOMETER_PER_HOUR)
            if free_flow_speed_kph is not None
            else None
        ),
        geometry=geometry,
        physical_restrictions=PhysicalRestrictions(
            excluded_vehicle_types=excluded_vehicle_types
        ),
    )


def make_junction(
    junction_id: str,
    *,
    position: GeoPoint | None = None,
    approaches: tuple[Approach, ...] = (),
    movements: tuple[TurnMovement, ...] = (),
) -> Junction:
    """One junction."""
    return Junction(
        junction_id=junction_id,
        name=f"Junction {junction_id}",
        kind=JunctionKind.SIGNALIZED,
        position=position,
        approaches=approaches,
        movements=movements,
    )


def make_facility(
    facility_id: str,
    access_junction_id: str,
    *,
    facility_type: FacilityType = FacilityType.HOSPITAL,
    accepts: frozenset[EmergencyVehicleType] = frozenset(),
) -> Facility:
    """One emergency destination."""
    return Facility(
        facility_id=facility_id,
        name=f"Facility {facility_id}",
        facility_type=facility_type,
        access_junction_id=access_junction_id,
        accepts_emergency_types=accepts,
    )


def make_network(
    *,
    network_id: str = "test-net",
    version: str = "1.0",
    junctions: tuple[Junction, ...] | None = None,
    segments: tuple[RoadSegment, ...] | None = None,
    facilities: tuple[Facility, ...] = (),
) -> RoadNetwork:
    """A network from the given parts, or a default two-junction pair."""
    if junctions is None:
        junctions = (make_junction("J1"), make_junction("J2"))
    if segments is None:
        segments = (make_segment("SEG-12", "J1", "J2"), make_segment("SEG-21", "J2", "J1"))
    return RoadNetwork(
        network_id=network_id,
        name="Synthetic test network",
        version=version,
        driving_side=DrivingSide.LEFT,
        provenance=PROV,
        junctions=junctions,
        segments=segments,
        facilities=facilities,
    )


def square_network(
    *, positioned: bool = True, free_flow_speed_kph: float | None = None
) -> RoadNetwork:
    """A four-junction square with bidirectional segments.

    Coordinates are a regular grid about (16.50, 80.60) with 0.005 degree sides,
    roughly 550 m. ``positioned=False`` drops every junction position, which is
    what an unsurveyed map import looks like and is the case exclusion-zone
    resolution has to report rather than silently skip.
    """
    coords = {
        "J1": (16.5050, 80.6000),
        "J2": (16.5050, 80.6050),
        "J3": (16.5000, 80.6050),
        "J4": (16.5000, 80.6000),
    }
    pairs = [
        ("J1", "J2"),
        ("J2", "J1"),
        ("J2", "J3"),
        ("J3", "J2"),
        ("J3", "J4"),
        ("J4", "J3"),
        ("J4", "J1"),
        ("J1", "J4"),
    ]
    segments = tuple(
        make_segment(
            f"SEG-{a[1]}{b[1]}",
            a,
            b,
            length_m=550.0,
            free_flow_speed_kph=free_flow_speed_kph,
            geometry=(
                GeoPoint(latitude=coords[a][0], longitude=coords[a][1]),
                GeoPoint(latitude=coords[b][0], longitude=coords[b][1]),
            )
            if positioned
            else (),
        )
        for a, b in pairs
    )

    def approaches_for(junction_id: str) -> tuple[Approach, ...]:
        return tuple(
            Approach(
                approach_id=f"{junction_id}-A-{s.segment_id}",
                incoming_segment_id=s.segment_id,
                lanes=(
                    Lane(index=0, permitted_movements=frozenset({MovementType.THROUGH})),
                ),
            )
            for s in segments
            if s.to_junction_id == junction_id
        )

    def movements_for(junction_id: str) -> tuple[TurnMovement, ...]:
        incoming = [s for s in segments if s.to_junction_id == junction_id]
        outgoing = [s for s in segments if s.from_junction_id == junction_id]
        out = []
        for inc in incoming:
            for outg in outgoing:
                is_u_turn = outg.to_junction_id == inc.from_junction_id
                out.append(
                    TurnMovement(
                        movement_id=f"MV-{inc.segment_id}-{outg.segment_id}",
                        junction_id=junction_id,
                        from_segment_id=inc.segment_id,
                        to_segment_id=outg.segment_id,
                        movement_type=(
                            MovementType.U_TURN if is_u_turn else MovementType.THROUGH
                        ),
                        is_permitted=not is_u_turn,
                    )
                )
        return tuple(out)

    junctions = tuple(
        make_junction(
            j,
            position=(
                GeoPoint(latitude=coords[j][0], longitude=coords[j][1])
                if positioned
                else None
            ),
            approaches=approaches_for(j),
            movements=movements_for(j),
        )
        for j in ("J1", "J2", "J3", "J4")
    )

    facilities = (
        make_facility(
            "FAC-HOSP-1",
            "J3",
            facility_type=FacilityType.HOSPITAL,
            accepts=frozenset({EmergencyVehicleType.AMBULANCE}),
        ),
        make_facility(
            "FAC-FIRE-1",
            "J1",
            facility_type=FacilityType.FIRE_STATION,
            accepts=frozenset({EmergencyVehicleType.FIRE_ENGINE}),
        ),
    )
    return make_network(junctions=junctions, segments=segments, facilities=facilities)
