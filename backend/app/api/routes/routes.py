"""Route optimization API.

Phase 7 scope: **which way to go, and why**. Ranked admissible routes, their cost
breakdowns, the evidence each cost rested on, and the reasons the winner won.

Facts, not prose
----------------
Every reason this endpoint returns is a code with numbers attached. It writes no
sentences for an operator to read as an explanation, because an explanation
generated in the same place as the decision cannot contradict it -- and the
ability to contradict is the only thing that makes an explanation worth having.
Narrative is Phase 12, generated against a fact table built from exactly these
numbers.

No approval, no execution
-------------------------
This endpoint recommends. It does not accept, dispatch, actuate or simulate, and
there is deliberately no endpoint here that does. Human approval and simulated
execution belong to the Decision Engine and Simulation phases, and collapsing
them into the optimizer would remove the step at which a person can say no.

Traffic is used only where it is bound
--------------------------------------
An observation affects a segment's cost only when the loaded mapping binds it
there. Unbound evidence is named in the response and priced at nothing, because
attributing a camera's verdict to a road nobody surveyed it against would be a
specific false claim that a route was then planned around.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

from fastapi import APIRouter
from pydantic import Field

from app.api.deps import (
    BindingRegistryDep,
    EmergencyRegistryDep,
    IncidentRegistryDep,
    MappingStoreDep,
    NetworkStoreDep,
    SettingsDep,
    StateStoreDep,
)
from app.api.routing_context import prepare_routing
from app.api.schemas import ApiModel, ErrorResponse
from app.core.logging import get_logger
from app.ingestion.storage import resolve_within
from app.routing.engine import optimize
from app.routing.mapping import (
    MappingSource,
    ObservationMapping,
    load_mapping,
    validate_against_network,
)
from app.routing.schemas import RouteOptimizationResult
from app.routing.search import ALGORITHM, ALGORITHM_VERSION
from app.routing.weights import RouteWeights, RoutingMode, weights_for

logger = get_logger(__name__)
router = APIRouter(prefix="/routes", tags=["routes"])

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "No network loaded, or unknown id"},
    422: {"model": ErrorResponse, "description": "The request does not fit the network"},
}


class RouteConfigResponse(ApiModel):
    """The routing policy in force, for both modes.

    Exposed for the same reason the analytics thresholds and congestion bands
    are: a ranking is meaningless without the weights that produced it, and an
    operator who disagrees with a recommendation needs something specific to
    disagree with.
    """

    algorithm: str
    algorithm_version: str
    normal: RouteWeights
    emergency: RouteWeights
    mapping_loaded: bool
    mapping: MappingSource | None = None
    note: str


class OptimizeRouteRequest(ApiModel):
    """What to route, and under which profile."""

    mode: RoutingMode = RoutingMode.NORMAL
    origin_junction_id: Annotated[str, Field(max_length=64)] | None = None
    destination_junction_id: Annotated[str, Field(max_length=64)] | None = None
    destination_facility_id: Annotated[str, Field(max_length=64)] | None = None

    emergency_id: Annotated[str, Field(max_length=64)] | None = None
    """When given, origin, destination and vehicle type are read from the
    declared response rather than supplied here, and the mode is forced to
    emergency. A dispatch's own record is the authority on where it is and where
    it is going; letting a caller restate them invites the two to disagree."""

    max_candidates: Annotated[int, Field(ge=1, le=20)] | None = None


class LoadMappingRequest(ApiModel):
    """Which observation-mapping file to load."""

    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    """Path relative to the networks directory, resolved with the same traversal
    guard the media and network libraries use."""


class MappingStatusResponse(ApiModel):
    """Whether an observation mapping is in force, and what it binds."""

    loaded: bool
    source: MappingSource | None = None
    mapping: ObservationMapping | None = None
    bound_segment_ids: tuple[str, ...] = ()
    bound_observation_area_ids: tuple[str, ...] = ()
    interpreted_area_ids: tuple[str, ...] = ()
    """Observation areas Phase 5 has actually interpreted. The intersection with
    ``bound_observation_area_ids`` is the evidence the optimizer can use; the
    difference is evidence it must ignore."""

    unmapped_observation_area_ids: tuple[str, ...] = ()

    confirmed_binding_count: int = 0
    """How many site-to-junction bindings an operator has confirmed on the
    loaded network. Reported beside the file-loaded mapping rather than merged
    into it: the two have different origins -- a surveyed file and a human
    accepting a proximity proposal -- and a reviewer must be able to tell which
    a binding came from."""

    confirmed_observation_area_ids: tuple[str, ...] = ()
    """Areas with a confirmed junction binding. These are junction-level and so
    contribute nothing to segment costing, exactly as a junction binding in a
    mapping file always has: a junction is not a directed segment, and binding
    a queue to the wrong direction is the failure this module exists to
    prevent."""

    confirmed_stale_version_area_ids: tuple[str, ...] = ()
    """Confirmed bindings made against a different version of the loaded
    network. Reported, never auto-invalidated."""

    note: str


def _interpreted_area_ids(states: dict[str, Any]) -> tuple[str, ...]:
    """Observation areas with an interpreted traffic state in this process."""
    return tuple(sorted({state.observation_area_id for state in states.values()}))


@router.get("/config", response_model=RouteConfigResponse, summary="Routing policy")
def route_config(
    settings: SettingsDep, mappings: MappingStoreDep
) -> RouteConfigResponse:
    """Return the algorithm and the weight sets behind every ranking."""
    return RouteConfigResponse(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        normal=weights_for(settings, RoutingMode.NORMAL),
        emergency=weights_for(settings, RoutingMode.EMERGENCY),
        mapping_loaded=mappings.is_loaded,
        mapping=mappings.source,
        note=(
            "Every weight is in equivalent seconds, so a total cost is a "
            "risk-adjusted arrival time rather than a score. Emergency weights "
            "default to the normal ones: what separates emergency routing is "
            "structural -- vehicle exclusions become hard constraints and an "
            "emergency-specific travel time is preferred where one exists -- not a "
            "multiplier invented here."
        ),
    )


@router.get(
    "/mapping",
    response_model=MappingStatusResponse,
    summary="Observation-to-network mapping in force",
)
def mapping_status(
    mappings: MappingStoreDep,
    states: StateStoreDep,
    bindings: BindingRegistryDep,
    store: NetworkStoreDep,
) -> MappingStatusResponse:
    """Report which cameras are bound to which roads, and which are not.

    Confirmed site-to-junction bindings are reported alongside the loaded
    mapping file, not folded into it. Merging them would make a human
    confirmation indistinguishable from a surveyed binding in every downstream
    reader, and they are not the same strength of evidence.
    """
    interpreted = _interpreted_area_ids(states)
    network = store.network
    confirmed = bindings.list(network.network_id) if network is not None else ()
    stale = bindings.stale_version_ids(network) if network is not None else ()
    mapping = mappings.mapping
    bound_areas = tuple(sorted(mapping.area_ids)) if mapping else ()
    bound_segments = tuple(sorted(mapping.bindings_by_segment())) if mapping else ()
    unmapped = tuple(sorted(set(interpreted) - set(bound_areas)))

    if mapping is None:
        note = mappings.empty_mapping_reason()
    elif unmapped:
        note = (
            f"{len(unmapped)} interpreted observation area(s) are bound to nothing, "
            "so their traffic evidence cannot affect any route cost."
        )
    else:
        note = (
            f"{len(bound_areas)} observation area(s) bound across "
            f"{len(bound_segments)} segment(s)."
        )

    return MappingStatusResponse(
        loaded=mappings.is_loaded,
        source=mappings.source,
        mapping=mapping,
        bound_segment_ids=bound_segments,
        bound_observation_area_ids=bound_areas,
        interpreted_area_ids=interpreted,
        unmapped_observation_area_ids=unmapped,
        confirmed_binding_count=len(confirmed),
        confirmed_observation_area_ids=tuple(
            sorted(item.observation_area_id for item in confirmed)
        ),
        confirmed_stale_version_area_ids=stale,
        note=note,
    )


@router.post(
    "/mapping/load",
    response_model=MappingStatusResponse,
    responses=_ERRORS,
    summary="Load an observation-to-network mapping",
)
def load_mapping_endpoint(
    request: LoadMappingRequest,
    mappings: MappingStoreDep,
    store: NetworkStoreDep,
    settings: SettingsDep,
    states: StateStoreDep,
    bindings: BindingRegistryDep,
) -> MappingStatusResponse:
    """Load and validate a mapping against the loaded network.

    Requires a network, because a binding names segment ids and those mean
    nothing until there is a map to name them on.
    """
    network = store.require()
    path = resolve_within(settings.networks_dir, request.relative_path)
    mapping, source = load_mapping(
        path, request.relative_path, loaded_at=dt.datetime.now(tz=dt.UTC)
    )
    validate_against_network(mapping, network)
    mappings.load(mapping, source)

    logger.info(
        "observation mapping loaded",
        extra={
            "mapping_id": mapping.mapping_id,
            "version": mapping.version,
            "bindings": len(mapping.bindings),
            "sha256": source.sha256,
        },
    )
    return mapping_status(mappings, states, bindings, store)


@router.post(
    "/optimize",
    response_model=RouteOptimizationResult,
    responses=_ERRORS,
    summary="Rank admissible routes from origin to destination",
)
def optimize_route(
    request: OptimizeRouteRequest,
    store: NetworkStoreDep,
    settings: SettingsDep,
    incidents: IncidentRegistryDep,
    emergencies: EmergencyRegistryDep,
    states: StateStoreDep,
    mappings: MappingStoreDep,
) -> RouteOptimizationResult:
    """Optimize one route request over the loaded network.

    Returns 200 with ``status = no_admissible_route`` when every path is cut by
    a hard constraint. That is a complete answer carrying its causes, not an
    error: the operator learns the network is severed and where. No route is
    invented in its place, and for an emergency none is downgraded to an unsafe
    one.
    """
    now = dt.datetime.now(tz=dt.UTC)
    context = prepare_routing(
        store=store,
        settings=settings,
        incidents=incidents,
        emergencies=emergencies,
        states=states,
        mappings=mappings,
        mode=request.mode,
        origin_junction_id=request.origin_junction_id,
        destination_junction_id=request.destination_junction_id,
        destination_facility_id=request.destination_facility_id,
        emergency_id=request.emergency_id,
        max_candidates=request.max_candidates,
        at=now,
    )
    return optimize(
        context.request,
        context.optimizer,
        destination_junction_id=context.destination_junction_id,
        at=context.at,
    )
