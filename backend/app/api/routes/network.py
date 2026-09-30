"""Road network and network-state endpoints.

Phase 6 scope: **the surveyed graph, and what is currently true of it**. Loading
and validating a road network, serving its topology, and assembling the
per-instant state that declared incidents impose on it.

Why loading is explicit
-----------------------
No network is loaded until someone loads one, and ``/api/network`` answers 404
until then. Auto-loading a bundled example would mean a deployment could plan
routes over synthetic geography without anyone having chosen it -- and a map is
exactly the kind of input whose provenance must be a decision, not a default.

A 404 here is a statement about our records ("no map has been loaded"), not
about the world, which is what makes it honest where an empty ``NetworkState``
would not be.

The safety catalog is not part of the network
---------------------------------------------
``/api/safety/catalog`` has served real rules since Phase 1, because the rules
are *policy* -- what is unsafe and at which tier -- rather than
measurements. It is mounted here for URL tidiness and has nothing to do with
whether a network is loaded.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import Field

from app.api.deps import (
    EmergencyRegistryDep,
    IncidentRegistryDep,
    NetworkStoreDep,
    SettingsDep,
)
from app.api.schemas import ApiModel, ErrorResponse
from app.core.logging import get_logger
from app.core.versions import SCHEMA_VERSION
from app.domain.network import RoadNetwork
from app.domain.safety import PHASE1_SAFETY_CATALOG, SafetyRuleCatalog
from app.domain.state import NetworkState, StateAssemblyPolicy
from app.ingestion.storage import resolve_within
from app.network.assembler import FUSION_STRATEGY, assemble_network_state
from app.network.loader import NetworkSource, list_available, load_network

logger = get_logger(__name__)
router = APIRouter(tags=["network"])

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "No network loaded, or file not found"},
    422: {"model": ErrorResponse, "description": "The file is not a valid road network"},
}


class NetworkStatusResponse(ApiModel):
    """Whether a network is loaded, and which one.

    Always answers 200, including when nothing is loaded. "No network is
    loaded" is a complete and true description of this process's state, and a
    client needs to be able to ask it without handling an error.
    """

    loaded: bool
    source: NetworkSource | None = None
    available_files: tuple[str, ...] = ()
    """Relative paths under the networks directory that could be loaded. Listed
    without opening them: appearing here means the file exists and ends in
    ``.json``, not that it parses as a network."""

    networks_dir_exists: bool = False
    note: str = ""


class LoadNetworkRequest(ApiModel):
    """Which network file to load."""

    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    """Path relative to the configured networks directory. Resolved with the
    same traversal guard the media library uses, on the *resolved* path, so
    ``..`` segments, absolute paths and symlinks are all covered."""


class LoadNetworkResponse(ApiModel):
    """What was loaded, and what loading it discarded."""

    source: NetworkSource
    junction_count: int
    segment_count: int
    facility_count: int
    cleared_incident_count: int
    cleared_emergency_count: int
    note: str


@router.get(
    "/network/status",
    response_model=NetworkStatusResponse,
    summary="Whether a road network is loaded",
)
def network_status(
    store: NetworkStoreDep, settings: SettingsDep
) -> NetworkStatusResponse:
    """Report the loaded network and the files that could be loaded."""
    available = list_available(settings.networks_dir)
    loaded = store.is_loaded
    if loaded and store.source is not None:
        note = (
            f"Network {store.source.network_id!r} version {store.source.version} "
            f"loaded from {store.source.relative_path!r}."
        )
    elif available:
        note = (
            "No network is loaded. Load one with POST /api/network/load naming a "
            "file from available_files."
        )
    else:
        note = (
            "No network is loaded and the networks directory contains no JSON "
            f"files. Place one under {settings.networks_dir} or set NETWORKS_DIR."
        )

    return NetworkStatusResponse(
        loaded=loaded,
        source=store.source,
        available_files=available,
        networks_dir_exists=settings.networks_dir.exists(),
        note=note,
    )


@router.post(
    "/network/load",
    response_model=LoadNetworkResponse,
    responses=_ERRORS,
    summary="Load a road network from the networks directory",
)
def load_network_endpoint(
    request: LoadNetworkRequest,
    store: NetworkStoreDep,
    settings: SettingsDep,
    incidents: IncidentRegistryDep,
    emergencies: EmergencyRegistryDep,
) -> LoadNetworkResponse:
    """Load and validate a network, replacing whatever was loaded before.

    Declared incidents and emergencies are **discarded** on load. They are
    expressed as segment and junction ids, and those ids mean something
    different on a different map: keeping a blockage on ``SEG-4`` across a
    network change would silently move it to whatever ``SEG-4`` is now. Dropping
    them is the only interpretation that cannot be wrong, and the counts are
    reported so the loss is visible rather than silent.
    """
    path = resolve_within(settings.networks_dir, request.relative_path)
    network, source = load_network(
        path, request.relative_path, loaded_at=dt.datetime.now(tz=dt.UTC)
    )

    cleared_incidents = len(incidents)
    cleared_emergencies = len(emergencies)
    incidents.clear_all()
    emergencies.clear_all()
    store.load(network, source)

    logger.info(
        "road network loaded",
        extra={
            "network_id": network.network_id,
            "version": network.version,
            "junctions": len(network.junctions),
            "segments": len(network.segments),
            "sha256": source.sha256,
        },
    )

    return LoadNetworkResponse(
        source=source,
        junction_count=len(network.junctions),
        segment_count=len(network.segments),
        facility_count=len(network.facilities),
        cleared_incident_count=cleared_incidents,
        cleared_emergency_count=cleared_emergencies,
        note=(
            f"Loaded {len(network.junctions)} junctions and {len(network.segments)} "
            f"segments. {cleared_incidents} incident(s) and {cleared_emergencies} "
            "emergency response(s) were discarded, because their segment and "
            "junction ids refer to the previous network."
        ),
    )


@router.get(
    "/network",
    response_model=RoadNetwork,
    responses=_ERRORS,
    summary="Road network topology",
)
def get_network(store: NetworkStoreDep) -> RoadNetwork:
    """Return the loaded road network, or 404 explaining how to load one."""
    return store.require()


@router.get(
    "/network/state",
    response_model=NetworkState,
    responses=_ERRORS,
    summary="Current fused network state",
)
def get_network_state(
    store: NetworkStoreDep,
    settings: SettingsDep,
    incidents: IncidentRegistryDep,
    emergencies: EmergencyRegistryDep,
) -> NetworkState:
    """Assemble the per-instant state of the loaded network.

    Blockages come from declared incidents and are the hard-constraint input
    route pruning will use. Congestion is deliberately **not** populated: Phase 5
    interprets congestion for an observation area -- one camera's field of view
    -- and binding that to a segment is a survey fact nothing here has recorded.
    A band attached to the wrong segment is a specific false claim that a route
    would be planned around, so every element reports ``unknown`` instead.
    """
    network = store.require()
    now = dt.datetime.now(tz=dt.UTC)

    emergency_junctions = frozenset(
        emergency.current_position.junction_id
        for emergency in emergencies.active()
        if emergency.current_position.junction_id is not None
    )

    return assemble_network_state(
        network,
        incidents=incidents.list(),
        policy=StateAssemblyPolicy(
            max_staleness_s=settings.max_observation_staleness_s,
            drop_after_s=settings.drop_observation_after_s,
            fusion_strategy=FUSION_STRATEGY,
        ),
        schema_version=SCHEMA_VERSION,
        as_of=now,
        assembled_at=now,
        emergency_junction_ids=emergency_junctions,
    )


class BlockedSegmentsResponse(ApiModel):
    """Segments that are currently inadmissible, and why.

    The hard-constraint input to route pruning, served on its own because it is
    the one question Phase 7 asks of this phase most often, and because reading
    it from the full snapshot means transferring every element to learn about a
    handful.
    """

    network_id: str
    network_version: str
    blocked_segment_ids: tuple[str, ...]
    degraded_segment_ids: tuple[str, ...]
    """Passable but degraded. Priced by a soft penalty, never pruned -- a jammed
    road is slow, a blocked road is impassable, and conflating them either
    strands a responder or discards a usable route."""

    active_incident_count: int
    as_of: dt.datetime


@router.get(
    "/network/blocked",
    response_model=BlockedSegmentsResponse,
    responses=_ERRORS,
    summary="Segments made inadmissible by active incidents",
)
def get_blocked_segments(
    store: NetworkStoreDep,
    incidents: IncidentRegistryDep,
) -> BlockedSegmentsResponse:
    """Return the segments active incidents block or degrade."""
    from app.network.assembler import degraded_segment_ids

    network = store.require()
    active = incidents.active()
    blocked: set[str] = set()
    for incident in active:
        blocked |= incident.hard_blocked_segment_ids

    known = {segment.segment_id for segment in network.segments}
    return BlockedSegmentsResponse(
        network_id=network.network_id,
        network_version=network.version,
        blocked_segment_ids=tuple(sorted(blocked & known)),
        degraded_segment_ids=tuple(sorted(degraded_segment_ids(network, active))),
        active_incident_count=len(active),
        as_of=dt.datetime.now(tz=dt.UTC),
    )


@router.get(
    "/network/facilities",
    responses=_ERRORS,
    summary="Emergency destinations on the loaded network",
)
def get_facilities(
    store: NetworkStoreDep,
    accepts: Annotated[str | None, Query(max_length=40)] = None,
) -> dict[str, object]:
    """Return the facilities attached to the network, optionally filtered.

    ``accepts`` narrows to facilities valid for one responder type. A fire
    station is not a destination for an ambulance carrying a patient, and
    filtering here means a caller cannot build a dispatch that the emergency
    validator will only reject later.
    """
    network = store.require()
    facilities = network.facilities
    if accepts:
        facilities = tuple(
            facility
            for facility in facilities
            if not facility.accepts_emergency_types
            or any(item.value == accepts for item in facility.accepts_emergency_types)
        )
    return {
        "network_id": network.network_id,
        "facilities": [facility.model_dump(mode="json") for facility in facilities],
        "count": len(facilities),
    }


@router.get(
    "/safety/catalog",
    response_model=SafetyRuleCatalog,
    tags=["safety"],
    summary="Versioned safety rule catalog",
)
def safety_catalog() -> SafetyRuleCatalog:
    """Return the safety rules in force.

    Genuinely implemented in Phase 1, because the rules are *policy* -- what is
    unsafe and at which tier -- rather than measurements. The numeric penalty
    weights for soft rules are deliberately absent: those are empirical tuning
    parameters calibrated against simulation in Phase 11, and inventing them now
    would place fabricated numbers at the centre of route selection.
    """
    return PHASE1_SAFETY_CATALOG
