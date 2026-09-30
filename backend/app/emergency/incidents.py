"""Incident lifecycle, held against the loaded network.

Two responsibilities, and they are separate on purpose:

* **Referential validation.** Every segment and junction an incident names must
  exist on the loaded network. An incident referencing an unknown segment is
  accepted nowhere: it would prune nothing, penalise nothing, and look active on
  a dashboard while constraining no route at all.
* **Lifecycle.** Status moves along an explicit transition table rather than
  being assigned freely. ``CLEARED`` is not a label an operator types over
  ``REPORTED`` -- it is the end of a path, and which paths exist is a decision
  worth writing down where it can be reviewed.

Terminal means terminal
-----------------------
``CLEARED`` and ``FALSE_ALARM`` accept no further transition. Reopening a
cleared incident would make "was this segment blocked at 14:05?" unanswerable
from the record, because the same id would have held both answers. A recurrence
is a new incident, with its own id and its own reported time.

Incidents are immutable values
------------------------------
:class:`~app.domain.incident.Incident` is frozen, so a transition produces a new
instance rather than mutating one in place. Two concurrent readers of the
registry therefore never observe a half-applied change, and a snapshot that
captured an incident keeps the incident it captured.
"""

from __future__ import annotations

import datetime as dt
from typing import Final

from app.core.errors import ResourceNotFoundError, ValidationFailedError
from app.domain.enums import IncidentStatus
from app.domain.incident import Incident
from app.domain.network import RoadNetwork

#: Which status may follow which. Absent keys are terminal.
INCIDENT_TRANSITIONS: Final[dict[IncidentStatus, frozenset[IncidentStatus]]] = {
    IncidentStatus.REPORTED: frozenset(
        {
            IncidentStatus.CONFIRMED,
            IncidentStatus.RESPONDING,
            IncidentStatus.FALSE_ALARM,
            IncidentStatus.CLEARED,
        }
    ),
    IncidentStatus.CONFIRMED: frozenset(
        {
            IncidentStatus.RESPONDING,
            IncidentStatus.CONTAINED,
            IncidentStatus.FALSE_ALARM,
            IncidentStatus.CLEARED,
        }
    ),
    IncidentStatus.RESPONDING: frozenset({IncidentStatus.CONTAINED, IncidentStatus.CLEARED}),
    IncidentStatus.CONTAINED: frozenset({IncidentStatus.CLEARED}),
    IncidentStatus.CLEARED: frozenset(),
    IncidentStatus.FALSE_ALARM: frozenset(),
}

#: Statuses that end an incident's life.
TERMINAL_STATUSES: Final[frozenset[IncidentStatus]] = frozenset(
    {IncidentStatus.CLEARED, IncidentStatus.FALSE_ALARM}
)


def revalidated(incident: Incident, updates: dict[str, object]) -> Incident:
    """Apply updates and re-run every model validator.

    ``model_copy(update=...)`` does **not** re-validate in pydantic v2: it
    writes the fields and returns. That is precisely wrong here, because the
    invariants being crossed are the safety-relevant ones -- a ``CLEARED``
    incident must record ``cleared_at``, a blocked segment must name the
    incident blocking it. Round-tripping through ``model_validate`` costs a
    dump-and-parse on a transition, which happens once per lifecycle change, and
    buys the guarantee that no invalid incident can exist in the registry.
    """
    return Incident.model_validate({**incident.model_dump(), **updates})


def validate_against_network(incident: Incident, network: RoadNetwork) -> None:
    """Reject an incident that names network elements which do not exist.

    Checked here rather than trusted, because the failure is silent: an incident
    blocking segment ``SEG-99`` on a network that has no ``SEG-99`` prunes
    nothing while appearing, in every listing and on every dashboard, to be an
    active blockage.
    """
    segment_ids = {segment.segment_id for segment in network.segments}
    junction_ids = {junction.junction_id for junction in network.junctions}

    unknown_segments: set[str] = set()
    unknown_junctions: set[str] = set()

    if incident.location.segment_id and incident.location.segment_id not in segment_ids:
        unknown_segments.add(incident.location.segment_id)
    if incident.location.junction_id and incident.location.junction_id not in junction_ids:
        unknown_junctions.add(incident.location.junction_id)

    unknown_segments |= incident.blocks_segment_ids - segment_ids
    unknown_segments |= incident.affected_segment_ids - segment_ids
    if incident.exclusion_zone is not None:
        unknown_segments |= incident.exclusion_zone.excluded_segment_ids - segment_ids
        unknown_junctions |= incident.exclusion_zone.excluded_junction_ids - junction_ids

    if unknown_segments or unknown_junctions:
        parts = []
        if unknown_segments:
            parts.append(f"segment(s) {sorted(unknown_segments)}")
        if unknown_junctions:
            parts.append(f"junction(s) {sorted(unknown_junctions)}")
        msg = (
            f"Incident {incident.incident_id!r} references unknown "
            f"{' and '.join(parts)} on network {network.network_id!r} "
            f"(version {network.version}). An incident that names an element the "
            "network does not have would appear active while constraining nothing."
        )
        raise ValidationFailedError(
            msg,
            incident_id=incident.incident_id,
            unknown_segment_ids=sorted(unknown_segments),
            unknown_junction_ids=sorted(unknown_junctions),
        )


class IncidentRegistry:
    """In-memory incident records for this process.

    In-memory and per-process, consistent with the rest of the system before a
    persistence schema has settled. A restart therefore loses declared
    incidents, which is stated plainly in the API description rather than left
    for an operator to discover.
    """

    __slots__ = ("_incidents",)

    def __init__(self) -> None:
        self._incidents: dict[str, Incident] = {}

    def declare(self, incident: Incident, network: RoadNetwork) -> Incident:
        """Record a new incident after validating it against the network."""
        if incident.incident_id in self._incidents:
            msg = (
                f"Incident {incident.incident_id!r} already exists. Re-declaring an "
                "id would overwrite the record of what was blocked and when."
            )
            raise ValidationFailedError(msg, incident_id=incident.incident_id)
        validate_against_network(incident, network)
        self._incidents[incident.incident_id] = incident
        return incident

    def get(self, incident_id: str) -> Incident:
        """Return one incident, or raise a 404."""
        incident = self._incidents.get(incident_id)
        if incident is None:
            msg = f"No incident {incident_id!r} has been declared."
            raise ResourceNotFoundError(msg, incident_id=incident_id)
        return incident

    def list(self, *, active_only: bool = False) -> tuple[Incident, ...]:
        """Return incidents, newest report first."""
        items = sorted(
            self._incidents.values(), key=lambda item: item.reported_at, reverse=True
        )
        if active_only:
            items = [item for item in items if item.is_active]
        return tuple(items)

    def active(self) -> tuple[Incident, ...]:
        """Incidents that still constrain the network."""
        return self.list(active_only=True)

    def transition(
        self,
        incident_id: str,
        new_status: IncidentStatus,
        *,
        at: dt.datetime,
        note: str | None = None,
    ) -> Incident:
        """Move an incident along the lifecycle, or explain why it cannot move."""
        current = self.get(incident_id)
        allowed = INCIDENT_TRANSITIONS.get(current.status, frozenset())

        if current.status in TERMINAL_STATUSES:
            msg = (
                f"Incident {incident_id!r} is {current.status.value} and cannot change "
                "status. A recurrence is a new incident with its own id, so that the "
                "record of what was blocked and when stays answerable."
            )
            raise ValidationFailedError(
                msg, incident_id=incident_id, status=current.status.value
            )

        if new_status not in allowed:
            msg = (
                f"Incident {incident_id!r} cannot move from {current.status.value} to "
                f"{new_status.value}. Permitted next states: "
                f"{sorted(status.value for status in allowed)}."
            )
            raise ValidationFailedError(
                msg,
                incident_id=incident_id,
                status=current.status.value,
                requested_status=new_status.value,
            )

        updates: dict[str, object] = {"status": new_status.value}
        if new_status is IncidentStatus.CONFIRMED and current.confirmed_at is None:
            updates["confirmed_at"] = at
        if new_status in TERMINAL_STATUSES:
            # ``cleared_at`` records when the incident stopped constraining the
            # network, which is as true of a false alarm as of a clearance. The
            # domain validator requires it on a CLEARED incident, so setting it
            # here means a caller cannot record an ending without its time.
            updates["cleared_at"] = at
        if note:
            updates["description"] = note

        updated = revalidated(current, updates)
        self._incidents[incident_id] = updated
        return updated

    def clear_all(self) -> None:
        """Forget every incident. Used when a new network is loaded."""
        self._incidents.clear()

    def __len__(self) -> int:
        """Number of incidents on record, in any status."""
        return len(self._incidents)
