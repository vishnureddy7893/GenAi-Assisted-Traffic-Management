"""Hard constraints, applied before search rather than after selection.

Filtering first is the whole architecture
-----------------------------------------
Selecting a route and then testing it for safety is the obvious design and it is
wrong twice. It wastes work proportional to the number of candidates, and when
the best route fails the test there is no principled fallback because the
alternatives have already been discarded. Pruning first means the search only
ever explores the admissible subgraph, so whatever it returns is admissible *by
construction* -- there is no code path on which a blocked segment can appear in
a candidate.

The corollary is faced, not avoided: pruning can sever the graph. "No admissible
route" is then a real answer with real causes, never an occasion to relax a
constraint until something comes back.

What is hard here
-----------------
* A segment an active incident blocks, including via its exclusion zone.
* A segment whose permanent physical restrictions exclude the responding
  vehicle -- a ladder truck under a low bridge is a hard constraint that has
  nothing to do with traffic.
* A turn the junction records as prohibited.
* A turn the junction does not record at all, when it records others from the
  same approach. Omission is indistinguishable from an incomplete map import,
  and "unknown" must never be read as "allowed".

Congestion is not here, at any level. A gridlocked road is slow, not
impassable, and pruning it would discard a route an ambulance could still use.
"""

from __future__ import annotations

from app.domain.enums import EmergencyVehicleType
from app.domain.incident import Incident
from app.domain.network import RoadNetwork, TurnMovement
from app.domain.state import NetworkState
from app.routing.schemas import RejectionReason, RouteRejection


class AdmissibleSubgraph:
    """The network reduced to what may actually be used, with the reasons why."""

    __slots__ = (
        "_blocked_by",
        "_movements",
        "excluded_vehicle_segment_ids",
        "network",
        "pruned_segment_ids",
        "rejections",
    )

    def __init__(
        self,
        network: RoadNetwork,
        *,
        pruned_segment_ids: frozenset[str],
        excluded_vehicle_segment_ids: frozenset[str],
        blocked_by: dict[str, tuple[str, ...]],
        movements: dict[tuple[str, str, str], TurnMovement],
        rejections: tuple[RouteRejection, ...],
    ) -> None:
        self.network = network
        self.pruned_segment_ids = pruned_segment_ids
        self.excluded_vehicle_segment_ids = excluded_vehicle_segment_ids
        self._blocked_by = blocked_by
        self._movements = movements
        self.rejections = rejections

    def permits_segment(self, segment_id: str) -> bool:
        """Whether a segment survived hard pruning."""
        return segment_id not in self.pruned_segment_ids

    def blocking_incident_ids(self, segment_id: str) -> tuple[str, ...]:
        """Which incidents pruned a segment, if any."""
        return self._blocked_by.get(segment_id, ())

    def movement(
        self, junction_id: str, from_segment_id: str, to_segment_id: str
    ) -> TurnMovement | None:
        """The declared turn for this transition, if the junction declares one."""
        return self._movements.get((junction_id, from_segment_id, to_segment_id))

    def declares_movements_from(self, junction_id: str, from_segment_id: str) -> bool:
        """Whether the junction records any turn out of this approach.

        Used to distinguish "this junction was never surveyed for turns" from
        "this particular turn is not among the surveyed ones". The first admits
        every transition; the second forbids the undeclared one.
        """
        return any(
            key[0] == junction_id and key[1] == from_segment_id for key in self._movements
        )

    def permits_movement(
        self, junction_id: str, from_segment_id: str, to_segment_id: str
    ) -> bool:
        """Whether a turn may be taken."""
        movement = self.movement(junction_id, from_segment_id, to_segment_id)
        if movement is not None:
            return movement.is_permitted
        # No declaration for this turn. Permitted only if the junction declares
        # nothing at all from this approach, i.e. turns were never surveyed here.
        return not self.declares_movements_from(junction_id, from_segment_id)


def build_subgraph(
    network: RoadNetwork,
    *,
    network_state: NetworkState | None = None,
    incidents: tuple[Incident, ...] = (),
    vehicle_type: EmergencyVehicleType | None = None,
) -> AdmissibleSubgraph:
    """Prune the network down to what a route may use.

    ``vehicle_type`` applies the responder's physical exclusions. It is ``None``
    for general traffic, which is not the same as "no restrictions apply": it
    means no *vehicle-specific* restriction is known to apply, and a segment
    excluding fire engines stays open to everything else.
    """
    blocked_by: dict[str, list[str]] = {}

    for incident in incidents:
        for segment_id in incident.hard_blocked_segment_ids:
            blocked_by.setdefault(segment_id, []).append(incident.incident_id)

    if network_state is not None:
        for segment_state in network_state.segment_states:
            if segment_state.is_blocked:
                existing = blocked_by.setdefault(segment_state.segment_id, [])
                for incident_id in sorted(segment_state.blocked_by_incident_ids):
                    if incident_id not in existing:
                        existing.append(incident_id)

    known_segment_ids = {segment.segment_id for segment in network.segments}
    blocked = {
        segment_id: tuple(sorted(set(causes)))
        for segment_id, causes in blocked_by.items()
        if segment_id in known_segment_ids
    }

    vehicle_excluded: set[str] = set()
    if vehicle_type is not None:
        vehicle_excluded = {
            segment.segment_id
            for segment in network.segments
            if not segment.permits(vehicle_type)
        }

    pruned = frozenset(blocked) | frozenset(vehicle_excluded)

    movements: dict[tuple[str, str, str], TurnMovement] = {}
    prohibited: list[str] = []
    for junction in network.junctions:
        for movement in junction.movements:
            movements[
                (junction.junction_id, movement.from_segment_id, movement.to_segment_id)
            ] = movement
            if not movement.is_permitted:
                prohibited.append(movement.movement_id)

    rejections: list[RouteRejection] = []
    if blocked:
        incident_ids = sorted({item for causes in blocked.values() for item in causes})
        rejections.append(
            RouteRejection(
                reason=RejectionReason.SEGMENT_BLOCKED,
                message=(
                    f"{len(blocked)} segment(s) are impassable because an active "
                    "incident blocks them or an exclusion zone covers them. A hard "
                    "constraint is never traded against travel time."
                ),
                segment_ids=tuple(sorted(blocked)),
                incident_ids=tuple(incident_ids),
            )
        )
    if vehicle_excluded:
        rejections.append(
            RouteRejection(
                reason=RejectionReason.VEHICLE_PHYSICALLY_EXCLUDED,
                message=(
                    f"{len(vehicle_excluded)} segment(s) physically exclude a "
                    f"{vehicle_type.value if vehicle_type else 'responder'}. This is "
                    "a property of the map, not of traffic, and applies before any "
                    "incident is considered."
                ),
                segment_ids=tuple(sorted(vehicle_excluded)),
            )
        )
    if prohibited:
        rejections.append(
            RouteRejection(
                reason=RejectionReason.MOVEMENT_PROHIBITED,
                message=(
                    f"{len(prohibited)} turn(s) are recorded as prohibited and are "
                    "not available to any route. Banned turns are recorded "
                    "explicitly rather than omitted, so an incomplete map cannot "
                    "read as a permissive one."
                ),
                movement_ids=tuple(sorted(prohibited)),
            )
        )

    return AdmissibleSubgraph(
        network,
        pruned_segment_ids=pruned,
        excluded_vehicle_segment_ids=frozenset(vehicle_excluded),
        blocked_by=blocked,
        movements=movements,
        rejections=tuple(rejections),
    )
