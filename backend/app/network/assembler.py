"""Assembling a :class:`~app.domain.state.NetworkState` over the loaded network.

This is the join point of Phase 6: a surveyed graph on one side, declared
incidents on the other, producing the per-element snapshot that Phase 7's hard
pruning will consume.

Every element gets a state, including the silent ones
----------------------------------------------------
A state is emitted for every junction and every segment, not only for the ones
something is known about. The alternative -- omitting unobserved elements --
makes "is segment SEG-7 clear?" and "does segment SEG-7 exist?" the same lookup
failure, and route search would have to guess which it hit. An element with no
evidence is present with :attr:`~app.domain.enums.FreshnessStatus.UNKNOWN`
freshness, which is the literal truth: no observation has ever been received
for it.

Congestion is not populated here, and that is not an oversight
--------------------------------------------------------------
Phase 5 interprets congestion for an **observation area** -- one camera's field
of view. Attaching that verdict to a *segment* requires knowing which segment
the camera looks at, which is a survey fact nothing in this system has recorded.
So ``congestion_level`` stays ``UNKNOWN`` on every element rather than being
filled from whichever camera happened to be analysed most recently. A congestion
band on the wrong segment is worse than no band at all: it is a specific false
claim that a route may be planned around.

Blockage, by contrast, *is* populated, because an incident names the segments it
blocks explicitly. That is the hard-constraint input Phase 7 needs, and it is
available without any camera at all.
"""

from __future__ import annotations

import datetime as dt
import hashlib

from app.domain.enums import CongestionLevel, FreshnessStatus
from app.domain.incident import Incident
from app.domain.network import RoadNetwork
from app.domain.state import (
    JunctionState,
    NetworkState,
    RoadSegmentState,
    StateAssemblyPolicy,
)

#: Recorded on the snapshot so a reader can tell how elements were reconciled.
FUSION_STRATEGY = "network_graph_with_declared_incidents"


def _blocking_incidents(incidents: tuple[Incident, ...]) -> dict[str, set[str]]:
    """Map each blocked segment id to the incidents responsible for it.

    Built from :attr:`~app.domain.incident.Incident.hard_blocked_segment_ids`,
    which already unions an incident's explicit blockages with its exclusion
    zone and returns nothing at all once the incident is no longer active.
    """
    blocked: dict[str, set[str]] = {}
    for incident in incidents:
        for segment_id in incident.hard_blocked_segment_ids:
            blocked.setdefault(segment_id, set()).add(incident.incident_id)
    return blocked


def _affecting_incidents(incidents: tuple[Incident, ...]) -> dict[str, set[str]]:
    """Map each degraded-but-passable segment id to the incidents affecting it."""
    affected: dict[str, set[str]] = {}
    for incident in incidents:
        if not incident.is_active:
            continue
        for segment_id in incident.affected_segment_ids:
            affected.setdefault(segment_id, set()).add(incident.incident_id)
    return affected


def _junction_incidents(incidents: tuple[Incident, ...]) -> dict[str, set[str]]:
    """Map each junction id to the active incidents touching it."""
    touching: dict[str, set[str]] = {}
    for incident in incidents:
        if not incident.is_active:
            continue
        if incident.location.junction_id:
            touching.setdefault(incident.location.junction_id, set()).add(
                incident.incident_id
            )
        if incident.exclusion_zone is not None:
            for junction_id in incident.exclusion_zone.excluded_junction_ids:
                touching.setdefault(junction_id, set()).add(incident.incident_id)
    return touching


def assemble_network_state(
    network: RoadNetwork,
    *,
    incidents: tuple[Incident, ...],
    policy: StateAssemblyPolicy,
    schema_version: str,
    as_of: dt.datetime,
    assembled_at: dt.datetime,
    emergency_junction_ids: frozenset[str] = frozenset(),
) -> NetworkState:
    """Build the per-instant snapshot for the loaded network.

    Pure: every timestamp is supplied by the caller, so a test can pin the exact
    instant and a replay reproduces the snapshot it is checking against.
    """
    active = tuple(incident for incident in incidents if incident.is_active)
    blocked = _blocking_incidents(active)
    junction_touching = _junction_incidents(active)

    segment_states = tuple(
        RoadSegmentState(
            segment_id=segment.segment_id,
            freshness=FreshnessStatus.UNKNOWN,
            congestion_level=CongestionLevel.UNKNOWN,
            is_blocked=segment.segment_id in blocked,
            blocked_by_incident_ids=frozenset(blocked.get(segment.segment_id, set())),
        )
        for segment in network.segments
    )

    junction_states = tuple(
        JunctionState(
            junction_id=junction.junction_id,
            freshness=FreshnessStatus.UNKNOWN,
            active_incident_ids=frozenset(
                junction_touching.get(junction.junction_id, set())
            ),
            emergency_active=junction.junction_id in emergency_junction_ids,
        )
        for junction in network.junctions
    )

    fingerprint = hashlib.sha256(
        "|".join(
            [network.network_id, network.version, *sorted(i.incident_id for i in active)]
        ).encode("utf-8")
    ).hexdigest()[:12]

    return NetworkState(
        state_id=f"netstate-{fingerprint}-{int(as_of.timestamp() * 1000)}",
        network_id=network.network_id,
        network_version=network.version,
        schema_version=schema_version,
        as_of=as_of,
        assembled_at=assembled_at,
        policy=policy,
        junction_states=junction_states,
        segment_states=segment_states,
        active_incidents=active,
    )


def degraded_segment_ids(
    network: RoadNetwork, incidents: tuple[Incident, ...]
) -> frozenset[str]:
    """Segments an active incident degrades without blocking.

    Kept separate from blockage because the two feed different safety tiers: a
    blocked segment is pruned before search, a degraded one is priced. Merging
    them would either strand a responder on a passable road or send one down an
    impassable one.
    """
    affected = _affecting_incidents(
        tuple(incident for incident in incidents if incident.is_active)
    )
    known = {segment.segment_id for segment in network.segments}
    return frozenset(segment_id for segment_id in affected if segment_id in known)
