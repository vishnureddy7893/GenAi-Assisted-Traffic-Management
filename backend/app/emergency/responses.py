"""Declared emergency responses, tracked against the loaded network.

Declaration, not detection
--------------------------
An :class:`~app.domain.emergency.EmergencyState` exists because somebody
declared it. A camera recognising an ambulance silhouette is corroborating
evidence and nothing more -- letting a visual detection create a response would
let one false positive trigger corridor recommendations across a network. This
module is therefore the only way a response comes into being, and every entry
point is an explicit declaration carrying its own provenance.

What is checked at declaration
------------------------------
Every graph anchor the response names must exist on the loaded network, and the
destination must be coherent for the responder type. Both are cheap here and
expensive later: a dispatch naming a junction the map does not have fails inside
route search as a lookup error on a live emergency request, which is the worst
possible moment to discover a typo.

The responder/destination pairing is checked twice, deliberately. The domain
model rejects an ambulance dispatched to a fire station from the declared
``facility_type`` alone, before any network lookup. This module then checks the
*actual* facility record, which is what catches a facility whose declared type
and real type disagree.
"""

from __future__ import annotations

import datetime as dt
from typing import Final

from app.core.errors import ResourceNotFoundError, ValidationFailedError
from app.domain.emergency import (
    VALID_FACILITY_DESTINATIONS,
    EmergencyState,
    NetworkPosition,
)
from app.domain.enums import EmergencyStatus
from app.domain.network import RoadNetwork

#: Which status may follow which. Absent keys are terminal.
EMERGENCY_TRANSITIONS: Final[dict[EmergencyStatus, frozenset[EmergencyStatus]]] = {
    EmergencyStatus.DECLARED: frozenset(
        {EmergencyStatus.EN_ROUTE, EmergencyStatus.CANCELLED}
    ),
    EmergencyStatus.EN_ROUTE: frozenset(
        {EmergencyStatus.ARRIVED, EmergencyStatus.CANCELLED}
    ),
    EmergencyStatus.ARRIVED: frozenset(
        {
            EmergencyStatus.RETURNING,
            EmergencyStatus.COMPLETED,
            EmergencyStatus.CANCELLED,
        }
    ),
    EmergencyStatus.RETURNING: frozenset(
        {EmergencyStatus.COMPLETED, EmergencyStatus.CANCELLED}
    ),
    EmergencyStatus.COMPLETED: frozenset(),
    EmergencyStatus.CANCELLED: frozenset(),
}

#: Statuses that end a response.
TERMINAL_STATUSES: Final[frozenset[EmergencyStatus]] = frozenset(
    {EmergencyStatus.COMPLETED, EmergencyStatus.CANCELLED}
)


def revalidated(emergency: EmergencyState, updates: dict[str, object]) -> EmergencyState:
    """Apply updates and re-run every model validator.

    ``model_copy(update=...)`` does not re-validate in pydantic v2, and the
    invariants at stake here are the ones worth keeping: a ``COMPLETED``
    response must record ``completed_at``, ``updated_at`` may not precede
    ``declared_at``, and a responder may not be pointed at a destination class
    it cannot serve.
    """
    return EmergencyState.model_validate({**emergency.model_dump(), **updates})


def _check_position(
    position: NetworkPosition | None,
    network: RoadNetwork,
    *,
    label: str,
    emergency_id: str,
) -> None:
    """Reject a position anchored to a graph element the network does not have."""
    if position is None:
        return
    segment_ids = {segment.segment_id for segment in network.segments}
    junction_ids = {junction.junction_id for junction in network.junctions}

    if position.segment_id is not None and position.segment_id not in segment_ids:
        msg = (
            f"Emergency {emergency_id!r} {label} references unknown segment "
            f"{position.segment_id!r} on network {network.network_id!r}."
        )
        raise ValidationFailedError(
            msg, emergency_id=emergency_id, segment_id=position.segment_id
        )
    if position.junction_id is not None and position.junction_id not in junction_ids:
        msg = (
            f"Emergency {emergency_id!r} {label} references unknown junction "
            f"{position.junction_id!r} on network {network.network_id!r}."
        )
        raise ValidationFailedError(
            msg, emergency_id=emergency_id, junction_id=position.junction_id
        )


def validate_against_network(
    emergency: EmergencyState,
    network: RoadNetwork,
    *,
    known_incident_ids: frozenset[str],
) -> None:
    """Reject a declaration that does not fit the loaded network."""
    _check_position(
        emergency.current_position,
        network,
        label="current position",
        emergency_id=emergency.emergency_id,
    )
    _check_position(
        emergency.origin, network, label="origin", emergency_id=emergency.emergency_id
    )
    _check_position(
        emergency.destination.position,
        network,
        label="destination position",
        emergency_id=emergency.emergency_id,
    )

    destination = emergency.destination
    if destination.facility_id is not None:
        facilities = {facility.facility_id: facility for facility in network.facilities}
        facility = facilities.get(destination.facility_id)
        if facility is None:
            msg = (
                f"Emergency {emergency.emergency_id!r} is destined for unknown facility "
                f"{destination.facility_id!r} on network {network.network_id!r}."
            )
            raise ValidationFailedError(
                msg,
                emergency_id=emergency.emergency_id,
                facility_id=destination.facility_id,
            )

        vehicle_type = emergency.vehicle.vehicle_type
        allowed = VALID_FACILITY_DESTINATIONS.get(vehicle_type, frozenset())
        if allowed and facility.facility_type not in allowed:
            msg = (
                f"Facility {facility.facility_id!r} is a {facility.facility_type.value}, "
                f"which is not a valid destination for a {vehicle_type.value}. Valid "
                f"types: {sorted(item.value for item in allowed)}."
            )
            raise ValidationFailedError(
                msg,
                emergency_id=emergency.emergency_id,
                facility_id=facility.facility_id,
            )

        if (
            facility.accepts_emergency_types
            and vehicle_type not in facility.accepts_emergency_types
        ):
            msg = (
                f"Facility {facility.facility_id!r} does not accept "
                f"{vehicle_type.value} responders."
            )
            raise ValidationFailedError(
                msg,
                emergency_id=emergency.emergency_id,
                facility_id=facility.facility_id,
            )

    for incident_id, label in (
        (destination.incident_id, "destination"),
        (emergency.triggering_incident_id, "triggering incident"),
    ):
        if incident_id is not None and incident_id not in known_incident_ids:
            msg = (
                f"Emergency {emergency.emergency_id!r} names {label} incident "
                f"{incident_id!r}, which has not been declared."
            )
            raise ValidationFailedError(
                msg, emergency_id=emergency.emergency_id, incident_id=incident_id
            )


class EmergencyRegistry:
    """In-memory declared responses for this process.

    Per-process and lost on restart, consistent with the rest of the system
    before a persistence schema has settled. Stated in the API description
    rather than left for an operator to discover during an incident.
    """

    __slots__ = ("_emergencies",)

    def __init__(self) -> None:
        self._emergencies: dict[str, EmergencyState] = {}

    def declare(
        self,
        emergency: EmergencyState,
        network: RoadNetwork,
        *,
        known_incident_ids: frozenset[str],
    ) -> EmergencyState:
        """Record a new response after validating it against the network."""
        if emergency.emergency_id in self._emergencies:
            msg = (
                f"Emergency {emergency.emergency_id!r} already exists. Re-declaring an "
                "id would overwrite the response-time clock the original started."
            )
            raise ValidationFailedError(msg, emergency_id=emergency.emergency_id)
        validate_against_network(
            emergency, network, known_incident_ids=known_incident_ids
        )
        self._emergencies[emergency.emergency_id] = emergency
        return emergency

    def get(self, emergency_id: str) -> EmergencyState:
        """Return one response, or raise a 404."""
        emergency = self._emergencies.get(emergency_id)
        if emergency is None:
            msg = f"No emergency {emergency_id!r} has been declared."
            raise ResourceNotFoundError(msg, emergency_id=emergency_id)
        return emergency

    def list(self, *, active_only: bool = False) -> tuple[EmergencyState, ...]:
        """Return responses, newest declaration first."""
        items = sorted(
            self._emergencies.values(), key=lambda item: item.declared_at, reverse=True
        )
        if active_only:
            items = [item for item in items if item.is_active]
        return tuple(items)

    def active(self) -> tuple[EmergencyState, ...]:
        """Responses still in progress."""
        return self.list(active_only=True)

    def transition(
        self,
        emergency_id: str,
        new_status: EmergencyStatus,
        *,
        at: dt.datetime,
    ) -> EmergencyState:
        """Move a response along the lifecycle, or explain why it cannot move."""
        current = self.get(emergency_id)

        if current.status in TERMINAL_STATUSES:
            msg = (
                f"Emergency {emergency_id!r} is {current.status.value} and cannot change "
                "status. A further response is a new declaration with its own id and "
                "its own response-time clock."
            )
            raise ValidationFailedError(
                msg, emergency_id=emergency_id, status=current.status.value
            )

        allowed = EMERGENCY_TRANSITIONS.get(current.status, frozenset())
        if new_status not in allowed:
            msg = (
                f"Emergency {emergency_id!r} cannot move from {current.status.value} to "
                f"{new_status.value}. Permitted next states: "
                f"{sorted(status.value for status in allowed)}."
            )
            raise ValidationFailedError(
                msg,
                emergency_id=emergency_id,
                status=current.status.value,
                requested_status=new_status.value,
            )

        updates: dict[str, object] = {"status": new_status.value, "updated_at": at}
        if new_status in TERMINAL_STATUSES:
            updates["completed_at"] = at

        updated = revalidated(current, updates)
        self._emergencies[emergency_id] = updated
        return updated

    def update_position(
        self,
        emergency_id: str,
        position: NetworkPosition,
        network: RoadNetwork,
        *,
        at: dt.datetime,
    ) -> EmergencyState:
        """Record where a responding unit now is."""
        current = self.get(emergency_id)
        if current.status in TERMINAL_STATUSES:
            msg = (
                f"Emergency {emergency_id!r} is {current.status.value}; a finished "
                "response does not move."
            )
            raise ValidationFailedError(
                msg, emergency_id=emergency_id, status=current.status.value
            )

        _check_position(
            position, network, label="position update", emergency_id=emergency_id
        )
        updated = revalidated(
            current,
            {"current_position": position.model_dump(), "updated_at": at},
        )
        self._emergencies[emergency_id] = updated
        return updated

    def clear_all(self) -> None:
        """Forget every response. Used when a new network is loaded."""
        self._emergencies.clear()

    def __len__(self) -> int:
        """Number of responses on record, in any status."""
        return len(self._emergencies)
