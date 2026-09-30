"""Traffic simulation API.

Phase 8 scope: **what would happen if we went that way**. A deterministic
point-queue model evolves the network over a configured horizon, drives one or
two routes through it, and reports the difference between them.

Recommend, simulate, stop
-------------------------
There is deliberately no endpoint here that approves, dispatches, actuates or
changes a signal. A simulation is the step that exists so a person can check a
recommendation *before* acting on it; putting an action button beside it would
remove exactly the check it was built to provide. Human approval is Phase 9 and
narrative is Phase 12.

Assumptions must justify themselves
-----------------------------------
A stated demand or capacity on a request carries ``note``, and the field is
required. That is the API-level expression of the domain rule that an
``ASSUMED`` value states why it was assumed: without it a caller could inject a
number that then appears in a result indistinguishable from a surveyed one.

Nothing here is asynchronous
----------------------------
The model is deterministic and bounded by the horizon, so a run finishes inside
the request that asked for it and its status is terminal from the moment it
exists. ``GET /runs/{id}`` retrieves a completed run rather than polling a
pending one, and the response says so.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Annotated, Any

from fastapi import APIRouter
from pydantic import Field

from app.api.deps import (
    AnalyticsStoreDep,
    EmergencyRegistryDep,
    IncidentRegistryDep,
    MappingStoreDep,
    NetworkStoreDep,
    SettingsDep,
    SimulationStoreDep,
    StateStoreDep,
)
from app.api.routing_context import (
    RoutingContext,
    current_network_state,
    prepare_routing,
)
from app.api.schemas import ApiModel, ErrorResponse
from app.core.config import Settings
from app.core.errors import ValidationFailedError
from app.core.logging import get_logger
from app.domain.network import RoadNetwork
from app.emergency.incidents import IncidentRegistry
from app.emergency.responses import EmergencyRegistry
from app.network.store import NetworkStore
from app.routing.engine import optimize
from app.routing.mapping import MappingStore
from app.routing.schemas import OptimizationStatus, RouteOptimizationResult
from app.routing.search import SearchLimitReachedError, k_shortest_paths
from app.routing.weights import RoutingMode
from app.simulation import runner
from app.simulation.scenario import (
    ObservedSegmentInput,
    StatedSegmentInput,
    build_links,
    build_scenario,
    observed_inputs,
)
from app.simulation.schemas import (
    MODEL_NAME,
    MODEL_VERSION,
    ArmRole,
    BaselineStrategy,
    LinkInputs,
    RouteArmInput,
    RouteSource,
    SimulationParameters,
    SimulationRejection,
    SimulationRun,
    UnavailableReason,
)
from app.simulation.store import SimulationStore

logger = get_logger(__name__)
router = APIRouter(prefix="/simulation", tags=["simulation"])

_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "No network loaded, or unknown id"},
    422: {"model": ErrorResponse, "description": "The request does not fit the network"},
}


class SegmentAssumption(ApiModel):
    """One operator-stated assumption about a segment, for this run only.

    Nothing here is written back to the road network. A capacity stated on a
    request is a what-if for one simulation; a capacity in the network file is a
    claim about a road. Conflating them would let a scenario silently re-survey
    the map.
    """

    segment_id: Annotated[str, Field(min_length=1, max_length=64)]
    note: Annotated[str, Field(min_length=1, max_length=400)]
    """Why this value is being assumed. Required: an assumption that does not
    justify itself is indistinguishable from a fabricated measurement, and this
    text travels into the result beside the number."""

    demand_veh_per_hour: Annotated[float, Field(ge=0, le=20_000)] | None = None
    capacity_veh_per_hour: Annotated[float, Field(gt=0, le=20_000)] | None = None
    initial_queue_veh: Annotated[float, Field(ge=0, le=10_000)] | None = None


class SimulationRunRequest(ApiModel):
    """What to simulate, over what horizon, under which assumptions."""

    horizon_s: Annotated[float, Field(gt=0, le=86_400)] | None = None
    step_s: Annotated[float, Field(gt=0, le=3600)] | None = None
    departure_offset_s: Annotated[float, Field(ge=0, le=86_400)] | None = None
    """When the evaluated vehicle departs, measured into the horizon. Non-zero
    evaluates a route against queues that have already built."""

    use_observed_demand: bool | None = None
    segment_inputs: tuple[SegmentAssumption, ...] = ()

    mode: RoutingMode = RoutingMode.NORMAL
    origin_junction_id: Annotated[str, Field(max_length=64)] | None = None
    destination_junction_id: Annotated[str, Field(max_length=64)] | None = None
    destination_facility_id: Annotated[str, Field(max_length=64)] | None = None
    emergency_id: Annotated[str, Field(max_length=64)] | None = None
    """Routes the declared response from its own record, exactly as the route
    optimizer does, so the two cannot disagree about where it is going."""

    route_segment_ids: tuple[str, ...] = ()
    """An explicit route to evaluate, segment by segment. When given, the
    optimizer is not consulted at all -- useful for evaluating a route an
    operator has in mind rather than one the system proposed."""


class SimulationCompareRequest(SimulationRunRequest):
    """A baseline route and a recommended one, run against identical conditions."""

    baseline: BaselineStrategy | None = None
    baseline_segment_ids: tuple[str, ...] = ()
    alternative_segment_ids: tuple[str, ...] = ()
    """Both optional. Given neither, the recommended arm is the optimizer's
    selection and the baseline follows the configured strategy."""


class SimulationConfigResponse(ApiModel):
    """The model and the parameters behind every simulated figure."""

    model_name: str
    model_version: str
    parameters_version: str
    horizon_s: float
    step_s: float
    step_count: int
    departure_offset_s: float
    emergency_queue_bypass_fraction: float
    use_observed_demand: bool
    default_baseline: BaselineStrategy
    max_steps: int
    max_stored_runs: int
    is_stochastic: bool
    stored_runs: int
    model_description: str
    limitations: tuple[str, ...]


class SimulationRunSummary(ApiModel):
    """One retained run, in list form."""

    run_id: str
    status: str
    computed_at: dt.datetime
    network_id: str
    fingerprint: str
    arm_count: int
    modelled_link_count: int
    has_comparison: bool


class SimulationRunListResponse(ApiModel):
    """Retained runs, newest first."""

    runs: tuple[SimulationRunSummary, ...] = ()
    count: int = 0
    max_runs: int = 0
    note: str


def _parameters(
    settings: Settings, request: SimulationRunRequest, baseline: BaselineStrategy
) -> SimulationParameters:
    """Resolve the run's parameters from configuration and request overrides."""
    horizon = request.horizon_s if request.horizon_s is not None else settings.sim_horizon_s
    step = request.step_s if request.step_s is not None else settings.sim_step_s
    offset = (
        request.departure_offset_s
        if request.departure_offset_s is not None
        else settings.sim_departure_offset_s
    )
    observed = (
        request.use_observed_demand
        if request.use_observed_demand is not None
        else settings.sim_use_observed_demand
    )

    if step > horizon:
        msg = (
            f"step_s ({step:g}) exceeds horizon_s ({horizon:g}); a simulation of "
            "less than one step produces no state to report."
        )
        raise ValidationFailedError(msg, step_s=step, horizon_s=horizon)

    steps = math.ceil(horizon / step)
    if steps > settings.sim_max_steps:
        msg = (
            f"horizon_s / step_s is {steps} steps, above this deployment's ceiling "
            f"of {settings.sim_max_steps}. Coarsen the step or shorten the horizon."
        )
        raise ValidationFailedError(msg, steps=steps, max_steps=settings.sim_max_steps)

    if offset >= horizon:
        msg = (
            f"departure_offset_s ({offset:g}) is at or beyond the horizon "
            f"({horizon:g}); the vehicle would never depart within the simulated "
            "period."
        )
        raise ValidationFailedError(msg, departure_offset_s=offset, horizon_s=horizon)

    return SimulationParameters(
        horizon_s=horizon,
        step_s=step,
        step_count=steps,
        departure_offset_s=offset,
        emergency_queue_bypass_fraction=settings.sim_emergency_queue_bypass_fraction,
        use_observed_demand=observed,
        baseline_strategy=baseline,
        parameters_version=settings.sim_parameters_version,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )


def _stated(request: SimulationRunRequest, network: RoadNetwork) -> tuple[
    StatedSegmentInput, ...
]:
    """Validate stated assumptions against the network before they are used."""
    known = {segment.segment_id for segment in network.segments}
    unknown = sorted(
        {item.segment_id for item in request.segment_inputs} - known
    )
    if unknown:
        msg = (
            f"Stated input(s) name segment(s) that are not on network "
            f"{network.network_id!r}: {', '.join(unknown)}."
        )
        raise ValidationFailedError(msg, segment_ids=unknown)

    duplicates = sorted(
        {
            item.segment_id
            for item in request.segment_inputs
            if sum(1 for other in request.segment_inputs if other.segment_id == item.segment_id)
            > 1
        }
    )
    if duplicates:
        msg = (
            "A segment may carry at most one stated input per run; two would leave "
            f"the run's inputs ambiguous: {', '.join(duplicates)}."
        )
        raise ValidationFailedError(msg, segment_ids=duplicates)

    return tuple(
        StatedSegmentInput(
            segment_id=item.segment_id,
            note=item.note,
            demand_veh_per_hour=item.demand_veh_per_hour,
            capacity_veh_per_hour=item.capacity_veh_per_hour,
            initial_queue_veh=item.initial_queue_veh,
        )
        for item in request.segment_inputs
    )


def _observed(
    analytics: dict[str, Any], mappings: MappingStore
) -> dict[str, ObservedSegmentInput]:
    """Observed contributions, reachable only through the loaded mapping."""
    return observed_inputs(tuple(analytics.values()), mappings.mapping)


def _explicit_arm(
    segment_ids: tuple[str, ...],
    network: RoadNetwork,
    *,
    role: ArmRole,
    label: str,
    mode: RoutingMode,
    route_id: str,
) -> RouteArmInput:
    """Build an arm from a caller-supplied segment sequence."""
    by_id = {segment.segment_id: segment for segment in network.segments}
    unknown = [item for item in segment_ids if item not in by_id]
    if unknown:
        msg = (
            f"Route segment(s) {', '.join(unknown)} are not on network "
            f"{network.network_id!r}."
        )
        raise ValidationFailedError(msg, segment_ids=unknown)
    first, last = by_id[segment_ids[0]], by_id[segment_ids[-1]]
    return RouteArmInput(
        role=role,
        route_id=route_id,
        label=label,
        source=RouteSource.EXPLICIT,
        mode=mode,
        segment_ids=segment_ids,
        origin_junction_id=first.from_junction_id,
        destination_junction_id=last.to_junction_id,
    )


def _shortest_distance_arm(
    context: RoutingContext, *, role: ArmRole, route_id: str
) -> RouteArmInput | None:
    """The cheapest route by distance over the same admissible subgraph.

    Run with the optimizer's own subgraph and its own distance edge cost, so the
    baseline is admissible under exactly the constraints the recommendation was
    made under. A baseline that crossed a blocked segment would flatter the
    recommendation by comparing it against something nobody could drive.
    """
    try:
        paths = k_shortest_paths(
            context.optimizer.index,
            context.optimizer.subgraph,
            origin_junction_id=context.request.origin_junction_id,
            destination_junction_id=context.destination_junction_id,
            edge_cost=context.optimizer.distance_edge_cost,
            k=1,
            max_settled=context.optimizer.weights.max_settled_nodes,
        )
    except SearchLimitReachedError:
        return None
    if not paths:
        return None
    return RouteArmInput(
        role=role,
        route_id=route_id,
        label="Shortest route by distance",
        source=RouteSource.SHORTEST_DISTANCE_SEARCH,
        mode=context.request.mode,
        segment_ids=paths[0][0],
        origin_junction_id=context.request.origin_junction_id,
        destination_junction_id=context.destination_junction_id,
    )


def _optimizer_arm(
    result: RouteOptimizationResult, rank: int, *, role: ArmRole, label: str
) -> RouteArmInput | None:
    """One of the optimizer's ranked candidates, as a simulation arm.

    The arm is named after its role rather than after the optimizer's candidate
    id. That id embeds a per-request identifier which is fresh on every call, so
    carrying it would make two runs of an identical scenario differ in their
    output for no reason connected to traffic -- and determinism is the property
    that makes a simulated comparison reviewable at all. Which candidate this
    was is recorded in ``source``, and the segments are in the arm itself.
    """
    route = next((item for item in result.routes if item.rank == rank), None)
    if route is None:
        return None
    return RouteArmInput(
        role=role,
        route_id=f"route-{role.value}",
        label=label,
        source=(
            RouteSource.OPTIMIZER_SELECTED
            if rank == 1
            else RouteSource.OPTIMIZER_ALTERNATIVE
        ),
        mode=result.request.mode,
        segment_ids=route.segment_sequence,
        origin_junction_id=route.junction_sequence[0],
        destination_junction_id=route.junction_sequence[-1],
    )


def _links_for(
    network: RoadNetwork,
    *,
    settings: Settings,
    incidents: IncidentRegistry,
    mode: RoutingMode,
    stated: tuple[StatedSegmentInput, ...],
    observed: dict[str, ObservedSegmentInput],
    use_observed_demand: bool,
    at: dt.datetime,
    context: RoutingContext | None,
) -> tuple[tuple[LinkInputs, ...], tuple[str, ...]]:
    """Resolve every link's inputs, reusing the routing snapshot where there is one."""
    active = context.incidents if context is not None else incidents.list()
    network_state = (
        context.network_state
        if context is not None
        else current_network_state(network, active, settings, at=at)
    )
    links = build_links(
        network,
        network_state=network_state,
        mode=mode,
        stated=stated,
        observed=observed,
        use_observed_demand=use_observed_demand,
    )
    return links, tuple(sorted(item.incident_id for item in active))


@router.get(
    "/config", response_model=SimulationConfigResponse, summary="Simulation model and parameters"
)
def simulation_config(
    settings: SettingsDep, runs: SimulationStoreDep
) -> SimulationConfigResponse:
    """Return the model in force and every parameter behind a simulated figure."""
    return SimulationConfigResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        parameters_version=settings.sim_parameters_version,
        horizon_s=settings.sim_horizon_s,
        step_s=settings.sim_step_s,
        step_count=math.ceil(settings.sim_horizon_s / settings.sim_step_s),
        departure_offset_s=settings.sim_departure_offset_s,
        emergency_queue_bypass_fraction=settings.sim_emergency_queue_bypass_fraction,
        use_observed_demand=settings.sim_use_observed_demand,
        default_baseline=BaselineStrategy(settings.sim_default_baseline),
        max_steps=settings.sim_max_steps,
        max_stored_runs=settings.sim_max_stored_runs,
        is_stochastic=False,
        stored_runs=runs.count,
        model_description=(
            "Deterministic point-queue link model. Each directed segment runs "
            "vehicles at its free-flow traversal time and discharges them at a "
            "stated exit capacity; demand above capacity forms a queue, and the "
            "evaluated vehicle waits for it. No random component and no seed: "
            "identical inputs produce identical output."
        ),
        limitations=(
            "Queues are vertical: they occupy no road space and cannot spill back "
            "into the segment upstream. A network at capacity is therefore modelled "
            "optimistically.",
            "Links are evolved independently. Vehicles discharged from a link leave "
            "the modelled area rather than moving to a successor, because no "
            "origin-destination matrix has been surveyed for this network.",
            "Demand is constant over the horizon unless stated otherwise. A peak "
            "arriving mid-horizon is not modelled.",
            "The evaluated vehicle is a probe: it experiences queues but neither "
            "adds to them nor consumes exit capacity.",
            "A segment with no capacity, no demand or no traversal time is reported "
            "unmodelled rather than simulated against a default, so a run over a "
            "sparsely surveyed network understates congestion rather than inventing "
            "it.",
        ),
    )


def _run_simulation(
    *,
    request: SimulationRunRequest,
    baseline_strategy: BaselineStrategy,
    arms_builder: Any,
    store: NetworkStore,
    settings: Settings,
    incidents: IncidentRegistry,
    emergencies: EmergencyRegistry,
    states: dict[str, Any],
    mappings: MappingStore,
    analytics: dict[str, Any],
    runs: SimulationStore,
) -> SimulationRun:
    """Shared body of the run and compare endpoints."""
    network = store.require()
    now = dt.datetime.now(tz=dt.UTC)
    parameters = _parameters(settings, request, baseline_strategy)
    stated = _stated(request, network)
    observed = _observed(analytics, mappings) if parameters.use_observed_demand else {}

    context: RoutingContext | None = None
    optimization: RouteOptimizationResult | None = None
    needs_optimizer = arms_builder.needs_optimizer(request)
    if needs_optimizer:
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
            max_candidates=None,
            at=now,
        )
        optimization = optimize(
            context.request,
            context.optimizer,
            destination_junction_id=context.destination_junction_id,
            at=now,
        )

    mode = context.request.mode if context is not None else request.mode
    links, incident_ids = _links_for(
        network,
        settings=settings,
        incidents=incidents,
        mode=mode,
        stated=stated,
        observed=observed,
        use_observed_demand=parameters.use_observed_demand,
        at=now,
        context=context,
    )

    arms, rejections, warnings = arms_builder.build(
        request=request,
        network=network,
        context=context,
        optimization=optimization,
        parameters=parameters,
    )

    scenario = build_scenario(
        network=network,
        parameters=parameters,
        links=links,
        arms=arms,
        traffic_state_id=(context.traffic_state.state_id if context else None),
        mapping_id=mappings.mapping.mapping_id if mappings.mapping else None,
        mapping_version=mappings.mapping.version if mappings.mapping else None,
        active_incident_ids=incident_ids,
    )

    if not arms:
        return runs.put(
            runner.unavailable_run(
                scenario, at=now, rejections=rejections, warnings=warnings
            )
        )

    return runs.put(
        runner.run(scenario, store.require_index(), at=now, warnings=warnings)
    )


class _SingleRouteArms:
    """Builds the one arm evaluated by ``POST /run``."""

    @staticmethod
    def needs_optimizer(request: SimulationRunRequest) -> bool:
        return not request.route_segment_ids and bool(
            request.origin_junction_id or request.emergency_id
        )

    @staticmethod
    def build(
        *,
        request: SimulationRunRequest,
        network: RoadNetwork,
        context: RoutingContext | None,
        optimization: RouteOptimizationResult | None,
        parameters: SimulationParameters,
    ) -> tuple[
        tuple[RouteArmInput, ...],
        tuple[SimulationRejection, ...],
        tuple[str, ...],
    ]:
        del context, parameters
        if request.route_segment_ids:
            return (
                (
                    _explicit_arm(
                        request.route_segment_ids,
                        network,
                        role=ArmRole.EVALUATED,
                        label="Route supplied on the request",
                        mode=request.mode,
                        route_id="route-evaluated",
                    ),
                ),
                (),
                (),
            )

        if optimization is None:
            return (
                (),
                (
                    SimulationRejection(
                        reason=UnavailableReason.NO_ROUTE_SUPPLIED,
                        message=(
                            "Nothing was given to evaluate. Supply route_segment_ids, "
                            "or an origin and destination for the optimizer to route "
                            "between, or an emergency_id to route from its own record."
                        ),
                    ),
                ),
                (),
            )

        if optimization.status is not OptimizationStatus.ROUTES_FOUND:
            return (
                (),
                (
                    SimulationRejection(
                        reason=UnavailableReason.NO_ADMISSIBLE_ROUTE,
                        message=(
                            "The optimizer found no admissible route, so there is "
                            "nothing to simulate. "
                            + " ".join(item.message for item in optimization.rejections)
                        )[:500],
                        segment_ids=optimization.pruned_segment_ids,
                    ),
                ),
                (),
            )

        arm = _optimizer_arm(
            optimization, 1, role=ArmRole.EVALUATED, label="Recommended route"
        )
        return ((arm,) if arm else (), (), ())


class _ComparisonArms:
    """Builds the baseline and recommended arms evaluated by ``POST /compare``."""

    @staticmethod
    def needs_optimizer(request: SimulationCompareRequest) -> bool:
        if request.baseline_segment_ids and request.alternative_segment_ids:
            return False
        return bool(request.origin_junction_id or request.emergency_id)

    @staticmethod
    def build(
        *,
        request: SimulationCompareRequest,
        network: RoadNetwork,
        context: RoutingContext | None,
        optimization: RouteOptimizationResult | None,
        parameters: SimulationParameters,
    ) -> tuple[
        tuple[RouteArmInput, ...],
        tuple[SimulationRejection, ...],
        tuple[str, ...],
    ]:
        compare = request
        if compare.baseline_segment_ids and compare.alternative_segment_ids:
            return (
                (
                    _explicit_arm(
                        compare.baseline_segment_ids,
                        network,
                        role=ArmRole.BASELINE,
                        label="Baseline route supplied on the request",
                        mode=compare.mode,
                        route_id="route-baseline",
                    ),
                    _explicit_arm(
                        compare.alternative_segment_ids,
                        network,
                        role=ArmRole.RECOMMENDED,
                        label="Alternative route supplied on the request",
                        mode=compare.mode,
                        route_id="route-recommended",
                    ),
                ),
                (),
                (),
            )

        if optimization is None or context is None:
            return (
                (),
                (
                    SimulationRejection(
                        reason=UnavailableReason.NO_ROUTE_SUPPLIED,
                        message=(
                            "Nothing was given to compare. Supply both "
                            "baseline_segment_ids and alternative_segment_ids, or an "
                            "origin and destination for the optimizer to route "
                            "between, or an emergency_id."
                        ),
                    ),
                ),
                (),
            )

        if optimization.status is not OptimizationStatus.ROUTES_FOUND:
            return (
                (),
                (
                    SimulationRejection(
                        reason=UnavailableReason.NO_ADMISSIBLE_ROUTE,
                        message=(
                            "The optimizer found no admissible route, so there is "
                            "nothing to compare. "
                            + " ".join(item.message for item in optimization.rejections)
                        )[:500],
                        segment_ids=optimization.pruned_segment_ids,
                    ),
                ),
                (),
            )

        recommended = _optimizer_arm(
            optimization, 1, role=ArmRole.RECOMMENDED, label="Recommended route"
        )
        if recommended is None:  # pragma: no cover - guarded by ROUTES_FOUND
            return ((), (), ())

        warnings: list[str] = []
        baseline: RouteArmInput | None = None
        strategy = parameters.baseline_strategy

        if strategy is BaselineStrategy.EXPLICIT and compare.baseline_segment_ids:
            baseline = _explicit_arm(
                compare.baseline_segment_ids,
                network,
                role=ArmRole.BASELINE,
                label="Baseline route supplied on the request",
                mode=compare.mode,
                route_id="route-baseline",
            )
        elif strategy is BaselineStrategy.RUNNER_UP:
            baseline = _optimizer_arm(
                optimization, 2, role=ArmRole.BASELINE, label="Runner-up route"
            )
        else:
            baseline = _shortest_distance_arm(
                context, role=ArmRole.BASELINE, route_id="route-baseline"
            )

        if baseline is None:
            return (
                (),
                (
                    SimulationRejection(
                        reason=UnavailableReason.ONLY_ONE_CANDIDATE,
                        message=(
                            "No baseline route could be established under strategy "
                            f"{strategy.value!r}, so there is nothing to compare the "
                            "recommendation against. The recommended route is "
                            "reported by the optimizer; no improvement figure exists."
                        ),
                        route_ids=(recommended.route_id,),
                    ),
                ),
                (),
            )

        if baseline.segment_ids == recommended.segment_ids:
            warnings.append(
                "The baseline and the recommended route are the same road: the "
                "system is recommending exactly what an operator would have chosen "
                "anyway, so the comparison correctly reports no difference."
            )

        return ((baseline, recommended), (), tuple(warnings))


@router.post(
    "/run",
    response_model=SimulationRun,
    responses=_ERRORS,
    summary="Simulate the network and one evaluated route",
)
def run_simulation(
    request: SimulationRunRequest,
    store: NetworkStoreDep,
    settings: SettingsDep,
    incidents: IncidentRegistryDep,
    emergencies: EmergencyRegistryDep,
    states: StateStoreDep,
    mappings: MappingStoreDep,
    analytics: AnalyticsStoreDep,
    runs: SimulationStoreDep,
) -> SimulationRun:
    """Evolve the network over the horizon and drive one route through it.

    Returns 200 with ``status = unavailable`` when the run could not be built --
    no route to evaluate, or none admissible. That is a complete answer with its
    causes, not an error, and no journey is invented in its place.
    """
    return _run_simulation(
        request=request,
        baseline_strategy=BaselineStrategy(settings.sim_default_baseline),
        arms_builder=_SingleRouteArms,
        store=store,
        settings=settings,
        incidents=incidents,
        emergencies=emergencies,
        states=states,
        mappings=mappings,
        analytics=analytics,
        runs=runs,
    )


@router.post(
    "/compare",
    response_model=SimulationRun,
    responses=_ERRORS,
    summary="Compare a baseline route against the recommended one",
)
def compare_routes(
    request: SimulationCompareRequest,
    store: NetworkStoreDep,
    settings: SettingsDep,
    incidents: IncidentRegistryDep,
    emergencies: EmergencyRegistryDep,
    states: StateStoreDep,
    mappings: MappingStoreDep,
    analytics: AnalyticsStoreDep,
    runs: SimulationStoreDep,
) -> SimulationRun:
    """Run both routes against identical conditions and report the difference.

    Both arms meet the same queues, built from the same demand over the same
    horizon under the same parameters, so a difference between them is
    attributable to the routes rather than to the conditions. Where either arm
    produces no travel time, the comparison reports no improvement rather than a
    one-sided figure.
    """
    strategy = (
        request.baseline
        if request.baseline is not None
        else BaselineStrategy(settings.sim_default_baseline)
    )
    if request.baseline_segment_ids and request.alternative_segment_ids:
        strategy = BaselineStrategy.EXPLICIT
    return _run_simulation(
        request=request,
        baseline_strategy=strategy,
        arms_builder=_ComparisonArms,
        store=store,
        settings=settings,
        incidents=incidents,
        emergencies=emergencies,
        states=states,
        mappings=mappings,
        analytics=analytics,
        runs=runs,
    )


@router.get(
    "/runs", response_model=SimulationRunListResponse, summary="Retained simulation runs"
)
def list_runs(runs: SimulationStoreDep) -> SimulationRunListResponse:
    """List the runs this process still holds, newest first."""
    stored = runs.list()
    return SimulationRunListResponse(
        runs=tuple(
            SimulationRunSummary(
                run_id=item.run_id,
                status=item.status.value,
                computed_at=item.computed_at,
                network_id=item.network_id,
                fingerprint=item.scenario.input_fingerprint,
                arm_count=len(item.routes),
                modelled_link_count=(
                    item.network.modelled_link_count if item.network else 0
                ),
                has_comparison=item.comparison is not None,
            )
            for item in stored
        ),
        count=len(stored),
        max_runs=runs.max_runs,
        note=(
            "Runs are held in memory for this process only. A run's id is the "
            "fingerprint of its inputs, so re-requesting an identical scenario "
            "reproduces the same run under the same id rather than creating a "
            "second one."
        ),
    )


@router.get(
    "/runs/{run_id}",
    response_model=SimulationRun,
    responses=_ERRORS,
    summary="Retrieve one completed simulation run",
)
def get_run(run_id: str, runs: SimulationStoreDep) -> SimulationRun:
    """Return a retained run.

    There is no pending state to poll: the model is deterministic and bounded, so
    a run finishes inside the request that asked for it and its status is
    terminal from the moment it exists.
    """
    return runs.get(run_id)
