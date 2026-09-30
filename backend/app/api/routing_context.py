"""Resolving one routing request against the process's loaded state.

Extracted so that route optimization and traffic simulation resolve a request
the *same* way. A simulation that re-derived the origin, the destination, the
network snapshot or the weight set would be simulating a slightly different
request from the one the optimizer answered, and the difference would be
invisible in both results -- which is the worst kind of disagreement, because
the comparison it produces still looks valid.

Nothing here decides anything. It gathers what a route request needs, validates
what a caller supplied, and hands back an optimizer bound to that instant.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import ValidationFailedError
from app.core.versions import SCHEMA_VERSION
from app.domain.emergency import EmergencyState
from app.domain.enums import EmergencyVehicleType
from app.domain.incident import Incident
from app.domain.network import RoadNetwork
from app.domain.state import NetworkState, StateAssemblyPolicy
from app.emergency.incidents import IncidentRegistry
from app.emergency.responses import EmergencyRegistry
from app.network.assembler import FUSION_STRATEGY, assemble_network_state
from app.network.store import NetworkStore
from app.routing.engine import RouteOptimizer, new_request_id
from app.routing.mapping import MappingStore
from app.routing.schemas import RouteRequest
from app.routing.weights import RoutingMode
from app.state.assembler import assemble as assemble_traffic_state
from app.state.schemas import NetworkTrafficState


@dataclass(frozen=True, slots=True)
class RoutingContext:
    """Everything one route request needs, resolved at one instant."""

    network: RoadNetwork
    optimizer: RouteOptimizer
    request: RouteRequest
    destination_junction_id: str
    network_state: NetworkState
    traffic_state: NetworkTrafficState
    incidents: tuple[Incident, ...]
    at: dt.datetime


def emergency_target(emergency: EmergencyState) -> tuple[str, str | None, str | None]:
    """Origin junction, destination facility and destination junction for a response.

    A dispatch's own record is the authority on where it is and where it is
    going. Letting a caller restate them invites the two to disagree.
    """
    origin = emergency.current_position.junction_id
    if origin is None:
        msg = (
            f"Emergency {emergency.emergency_id!r} is positioned mid-segment with no "
            "junction anchor, so it has no node to route from. Update its position "
            "to a junction first."
        )
        raise ValidationFailedError(msg, emergency_id=emergency.emergency_id)

    destination = emergency.destination
    if destination.facility_id is not None:
        return origin, destination.facility_id, None
    if destination.position is not None and destination.position.junction_id:
        return origin, None, destination.position.junction_id

    msg = (
        f"Emergency {emergency.emergency_id!r} has no routable destination: it names "
        "neither a facility nor a position anchored to a junction."
    )
    raise ValidationFailedError(msg, emergency_id=emergency.emergency_id)


def current_network_state(
    network: RoadNetwork,
    incidents: tuple[Incident, ...],
    settings: Settings,
    *,
    at: dt.datetime,
) -> NetworkState:
    """Assemble the per-instant network snapshot under the configured budget."""
    return assemble_network_state(
        network,
        incidents=incidents,
        policy=StateAssemblyPolicy(
            max_staleness_s=settings.max_observation_staleness_s,
            drop_after_s=settings.drop_observation_after_s,
            fusion_strategy=FUSION_STRATEGY,
        ),
        schema_version=SCHEMA_VERSION,
        as_of=at,
        assembled_at=at,
    )


def prepare_routing(
    *,
    store: NetworkStore,
    settings: Settings,
    incidents: IncidentRegistry,
    emergencies: EmergencyRegistry,
    states: dict[str, object],
    mappings: MappingStore,
    mode: RoutingMode,
    origin_junction_id: str | None,
    destination_junction_id: str | None,
    destination_facility_id: str | None,
    emergency_id: str | None,
    max_candidates: int | None,
    at: dt.datetime,
) -> RoutingContext:
    """Validate a route request and build the optimizer that will answer it."""
    network = store.require()

    origin = origin_junction_id
    facility_id = destination_facility_id
    destination = destination_junction_id
    vehicle_type: EmergencyVehicleType | None = None

    if emergency_id is not None:
        emergency = emergencies.get(emergency_id)
        mode = RoutingMode.EMERGENCY
        vehicle_type = emergency.vehicle.vehicle_type
        origin, facility_id, destination = emergency_target(emergency)

    if origin is None:
        msg = (
            "A route request needs an origin: give origin_junction_id, or an "
            "emergency_id whose declared position names a junction."
        )
        raise ValidationFailedError(msg)

    junction_ids = {junction.junction_id for junction in network.junctions}
    if origin not in junction_ids:
        msg = f"Origin junction {origin!r} is not on network {network.network_id!r}."
        raise ValidationFailedError(msg, origin_junction_id=origin)

    if facility_id is not None:
        facility = next(
            (f for f in network.facilities if f.facility_id == facility_id), None
        )
        if facility is None:
            msg = f"Facility {facility_id!r} is not on network {network.network_id!r}."
            raise ValidationFailedError(msg, facility_id=facility_id)
        resolved_destination = facility.access_junction_id
    elif destination is not None:
        if destination not in junction_ids:
            msg = (
                f"Destination junction {destination!r} is not on network "
                f"{network.network_id!r}."
            )
            raise ValidationFailedError(msg, destination_junction_id=destination)
        resolved_destination = destination
    else:
        msg = (
            "A route request needs a destination: give destination_junction_id or "
            "destination_facility_id."
        )
        raise ValidationFailedError(msg)

    active_incidents = incidents.list()
    network_state = current_network_state(network, active_incidents, settings, at=at)
    traffic_state = assemble_traffic_state(
        tuple(states.values()),  # type: ignore[arg-type]
        settings,
        assembled_at=at,
    )

    request = RouteRequest(
        request_id=new_request_id(),
        mode=mode,
        origin_junction_id=origin,
        destination_junction_id=None if facility_id else resolved_destination,
        destination_facility_id=facility_id,
        emergency_id=emergency_id,
        vehicle_type=vehicle_type,
        max_candidates=max_candidates,
        requested_at=at,
    )

    optimizer = RouteOptimizer(
        network,
        settings=settings,
        mode=mode,
        network_state=network_state,
        incidents=active_incidents,
        traffic_state=traffic_state,
        mapping=mappings.mapping,
        vehicle_type=vehicle_type,
        index=store.require_index(),
    )

    return RoutingContext(
        network=network,
        optimizer=optimizer,
        request=request,
        destination_junction_id=resolved_destination,
        network_state=network_state,
        traffic_state=traffic_state,
        incidents=active_incidents,
        at=at,
    )
