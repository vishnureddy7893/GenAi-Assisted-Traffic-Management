"""Running a scenario and assembling the structured result.

Order of operations
-------------------
Evolve the network first, then drive each evaluated route through it. Both arms
of a comparison therefore meet *the same* queues, built from the same demand
over the same horizon -- which is the only way a difference between them can be
attributed to the routes rather than to the conditions.

Where a figure cannot exist, it is absent
-----------------------------------------
An arm whose vehicle had not arrived when the horizon ended reports no travel
time, not a truncated one. A comparison with an unavailable arm reports no
improvement, not a one-sided number. A link with no capacity reports no queue,
not zero. Each refusal carries a code and a sentence, so the caller learns which
input to supply rather than being handed a plausible number.

No prose is written here
------------------------
Reasons are codes with numbers attached, exactly as in Phase 7. The sentence an
operator reads about *why* a route won in simulation is Phase 12's job, generated
against these facts and validated against them.
"""

from __future__ import annotations

import datetime as dt
from itertools import pairwise

from app.core.logging import get_logger
from app.domain.network import NetworkIndex
from app.domain.provenance import (
    FloatMeasurement,
    Provenance,
    Unit,
    ValueSource,
)
from app.routing.weights import RoutingMode
from app.simulation import model as engine
from app.simulation.scenario import modellable_links
from app.simulation.schemas import (
    MODEL_NAME,
    MODEL_VERSION,
    SIM_PRODUCER,
    ArmRole,
    ArmStatus,
    BetterArm,
    InputBasis,
    LegResult,
    LinkInputs,
    LinkResult,
    LinkStatus,
    NetworkSimulationSummary,
    RouteArmInput,
    RouteSimulationResult,
    SimulationComparison,
    SimulationRejection,
    SimulationRun,
    SimulationScenario,
    SimulationStatus,
    UnavailableReason,
)

logger = get_logger(__name__)

#: Two travel times closer than this are reported as no difference rather than
#: as a win. A millisecond of separation on a modelled journey is arithmetic
#: noise, and presenting it as an improvement would be false precision.
TIE_TOLERANCE_S = 1e-3


def _provenance(
    source: ValueSource,
    *,
    at: dt.datetime,
    input_refs: tuple[str, ...],
    notes: str | None = None,
) -> Provenance:
    """Provenance for one simulated or derived figure."""
    return Provenance(
        source=source,
        producer=SIM_PRODUCER,
        producer_version=MODEL_VERSION,
        observed_at=at,
        computed_at=at,
        input_refs=input_refs,
        notes=notes,
    )


def simulated(
    value: float,
    unit: Unit,
    *,
    at: dt.datetime,
    input_refs: tuple[str, ...],
    notes: str | None = None,
) -> FloatMeasurement:
    """Wrap a model output so it can never be read as a measurement.

    ``SIMULATED`` provenance makes :attr:`Measurement.is_factual` false, which is
    the mechanism -- not the convention -- that keeps a modelled outcome out of
    any sentence stating what the roads did.
    """
    return FloatMeasurement(
        value=round(value, 6),
        unit=unit,
        provenance=_provenance(
            ValueSource.SIMULATED, at=at, input_refs=input_refs, notes=notes
        ),
    )


def derived(
    value: float,
    unit: Unit,
    *,
    at: dt.datetime,
    input_refs: tuple[str, ...],
    notes: str | None = None,
) -> FloatMeasurement:
    """Wrap arithmetic over stated inputs, which is not a simulation outcome."""
    return FloatMeasurement(
        value=round(value, 6),
        unit=unit,
        provenance=_provenance(
            ValueSource.DERIVED, at=at, input_refs=input_refs, notes=notes
        ),
    )


def _link_result(
    link: LinkInputs, trace: engine.LinkTrace | None, *, at: dt.datetime
) -> LinkResult:
    """One segment's outcome, or an explicit statement that it has none."""
    common = {
        "segment_id": link.segment_id,
        "demand_veh_per_hour": link.demand_veh_per_hour,
        "demand_basis": link.demand_basis,
        "capacity_veh_per_hour": link.capacity_veh_per_hour,
        "capacity_basis": link.capacity_basis,
    }

    if trace is None:
        missing: list[str] = []
        if link.free_flow_traversal_s is None:
            missing.append(link.time_reason or "no traversal-time basis")
        if link.capacity_veh_per_hour is None:
            missing.append(link.capacity_reason or "no capacity basis")
        if link.demand_veh_per_hour is None:
            missing.append(link.demand_reason or "no demand basis")
        return LinkResult(
            **common,
            status=LinkStatus.NOT_MODELLED,
            reason=" ".join(missing)[:400],
        )

    refs = (link.segment_id,)
    mean_delay = trace.mean_delay_per_vehicle_s
    horizon_s = trace.step_s * len(trace.queue_veh)
    return LinkResult(
        **common,
        status=LinkStatus.MODELLED,
        saturation_ratio=derived(
            trace.saturation_ratio,
            Unit.RATIO,
            at=at,
            input_refs=refs,
            notes=(
                "Stated demand divided by stated capacity. A ratio of the inputs, "
                "not an outcome of the run."
            ),
        ),
        peak_queue_veh=simulated(
            trace.peak_queue_veh, Unit.VEHICLE, at=at, input_refs=refs
        ),
        final_queue_veh=simulated(
            trace.final_queue_veh, Unit.VEHICLE, at=at, input_refs=refs
        ),
        mean_queue_veh=simulated(
            trace.mean_queue_veh, Unit.VEHICLE, at=at, input_refs=refs
        ),
        throughput_veh=simulated(
            trace.throughput_veh, Unit.VEHICLE, at=at, input_refs=refs
        ),
        throughput_veh_per_hour=simulated(
            trace.throughput_veh / horizon_s * 3600.0 if horizon_s > 0 else 0.0,
            Unit.VEHICLE_PER_HOUR,
            at=at,
            input_refs=refs,
            notes="Vehicles discharged over the horizon, expressed as an hourly rate.",
        ),
        total_delay_veh_s=simulated(
            trace.total_delay_veh_s,
            Unit.SECOND,
            at=at,
            input_refs=refs,
            notes="Vehicle-seconds spent waiting in this link's exit queue.",
        ),
        mean_delay_per_vehicle_s=(
            simulated(mean_delay, Unit.SECOND, at=at, input_refs=refs)
            if mean_delay is not None
            else None
        ),
        queue_series_veh=trace.queue_veh,
    )


def _network_summary(
    links: tuple[LinkInputs, ...],
    traces: dict[str, engine.LinkTrace],
    *,
    at: dt.datetime,
) -> NetworkSimulationSummary:
    """Totals over the modelled links, named so they cannot be read as network-wide."""
    modelled = [traces[link.segment_id] for link in links if link.segment_id in traces]
    unmodelled = len(links) - len(modelled)
    refs = tuple(trace.segment_id for trace in modelled)
    oversaturated = tuple(
        sorted(trace.segment_id for trace in modelled if trace.saturation_ratio > 1.0)
    )

    if not modelled:
        return NetworkSimulationSummary(
            modelled_link_count=0,
            unmodelled_link_count=unmodelled,
            note=(
                f"No link could be modelled. All {unmodelled} segment(s) lack a "
                "demand basis, a capacity basis or a traversal time, each stated "
                "on the link. No queue, throughput or delay figure exists for this "
                "network, and none was substituted."
            ),
        )

    note = (
        f"Totals cover the {len(modelled)} link(s) that had every input; "
        f"{unmodelled} segment(s) could not be modelled and contribute nothing to "
        "these figures. They are a total over part of the network, not over it."
    )
    if oversaturated:
        note += (
            f" {len(oversaturated)} link(s) are oversaturated, so their queues grow "
            "for as long as the stated demand holds and their figures depend on the "
            "horizon rather than settling within it."
        )

    return NetworkSimulationSummary(
        modelled_link_count=len(modelled),
        unmodelled_link_count=unmodelled,
        modelled_throughput_veh=simulated(
            sum(trace.throughput_veh for trace in modelled),
            Unit.VEHICLE,
            at=at,
            input_refs=refs,
        ),
        modelled_peak_queue_veh=simulated(
            sum(trace.peak_queue_veh for trace in modelled),
            Unit.VEHICLE,
            at=at,
            input_refs=refs,
            notes=(
                "Sum of each link's own peak. The links do not necessarily peak at "
                "the same instant, so this is an upper bound on any single-instant "
                "network queue rather than a snapshot of one."
            ),
        ),
        modelled_final_queue_veh=simulated(
            sum(trace.final_queue_veh for trace in modelled),
            Unit.VEHICLE,
            at=at,
            input_refs=refs,
        ),
        modelled_total_delay_veh_s=simulated(
            sum(trace.total_delay_veh_s for trace in modelled),
            Unit.SECOND,
            at=at,
            input_refs=refs,
        ),
        oversaturated_segment_ids=oversaturated,
        note=note[:600],
    )


def _unavailable_arm(
    arm: RouteArmInput, rejections: tuple[SimulationRejection, ...]
) -> RouteSimulationResult:
    """An arm that produced no journey, carrying its causes."""
    return RouteSimulationResult(
        role=arm.role,
        route_id=arm.route_id,
        label=arm.label,
        source=arm.source,
        mode=arm.mode,
        status=ArmStatus.UNAVAILABLE,
        segment_ids=arm.segment_ids,
        rejections=rejections,
    )


def _validate_arm(
    arm: RouteArmInput,
    index: NetworkIndex,
    links: dict[str, LinkInputs],
) -> tuple[SimulationRejection, ...]:
    """Structural and admissibility checks, before anything is simulated."""
    unknown = tuple(
        segment_id for segment_id in arm.segment_ids if segment_id not in links
    )
    if unknown:
        return (
            SimulationRejection(
                reason=UnavailableReason.ROUTE_NOT_ON_NETWORK,
                message=(
                    f"Route {arm.route_id!r} names {len(unknown)} segment(s) that are "
                    "not on the loaded network, so there is nothing to simulate them "
                    "over."
                ),
                segment_ids=unknown,
                route_ids=(arm.route_id,),
            ),
        )

    segments = [index.segment(segment_id) for segment_id in arm.segment_ids]
    if segments[0].from_junction_id != arm.origin_junction_id:
        return (
            SimulationRejection(
                reason=UnavailableReason.ROUTE_DISCONTINUOUS,
                message=(
                    f"Route {arm.route_id!r} claims to start at "
                    f"{arm.origin_junction_id!r} but its first segment departs from "
                    f"{segments[0].from_junction_id!r}."
                ),
                route_ids=(arm.route_id,),
                junction_ids=(arm.origin_junction_id, segments[0].from_junction_id),
            ),
        )
    if segments[-1].to_junction_id != arm.destination_junction_id:
        return (
            SimulationRejection(
                reason=UnavailableReason.ROUTE_DISCONTINUOUS,
                message=(
                    f"Route {arm.route_id!r} claims to end at "
                    f"{arm.destination_junction_id!r} but its last segment arrives at "
                    f"{segments[-1].to_junction_id!r}."
                ),
                route_ids=(arm.route_id,),
                junction_ids=(arm.destination_junction_id, segments[-1].to_junction_id),
            ),
        )
    for previous, following in pairwise(segments):
        if previous.to_junction_id != following.from_junction_id:
            return (
                SimulationRejection(
                    reason=UnavailableReason.ROUTE_DISCONTINUOUS,
                    message=(
                        f"Route {arm.route_id!r} is discontinuous: segment "
                        f"{previous.segment_id!r} ends at {previous.to_junction_id!r} "
                        f"and {following.segment_id!r} starts at "
                        f"{following.from_junction_id!r}."
                    ),
                    route_ids=(arm.route_id,),
                    segment_ids=(previous.segment_id, following.segment_id),
                ),
            )

    blocked = tuple(
        segment_id for segment_id in arm.segment_ids if links[segment_id].is_blocked
    )
    if blocked:
        incidents = tuple(
            sorted(
                {
                    incident_id
                    for segment_id in blocked
                    for incident_id in links[segment_id].blocked_by_incident_ids
                }
            )
        )
        return (
            SimulationRejection(
                reason=UnavailableReason.SEGMENT_BLOCKED,
                message=(
                    f"Route {arm.route_id!r} crosses {len(blocked)} segment(s) that an "
                    "active incident blocks. Simulating a traversal of an impassable "
                    "road would produce a travel time for a journey that cannot be "
                    "made, so no figure is reported."
                ),
                segment_ids=blocked,
                incident_ids=incidents,
                route_ids=(arm.route_id,),
            ),
        )

    untimed = tuple(
        segment_id
        for segment_id in arm.segment_ids
        if links[segment_id].free_flow_traversal_s is None
    )
    if untimed:
        return (
            SimulationRejection(
                reason=UnavailableReason.NO_TRAVERSAL_TIME_BASIS,
                message=(
                    f"{len(untimed)} segment(s) on route {arm.route_id!r} have neither "
                    "a travel-time estimate nor a surveyed free-flow speed, so the "
                    "vehicle cannot be moved along them. No speed was assumed in "
                    "their place."
                ),
                segment_ids=untimed,
                route_ids=(arm.route_id,),
            ),
        )
    return ()


def _simulate_arm(
    arm: RouteArmInput,
    *,
    index: NetworkIndex,
    links: dict[str, LinkInputs],
    traces: dict[str, engine.LinkTrace],
    scenario: SimulationScenario,
    at: dt.datetime,
) -> RouteSimulationResult:
    """Drive one route through the evolved network and describe what happened."""
    rejections = _validate_arm(arm, index, links)
    if rejections:
        return _unavailable_arm(arm, rejections)

    parameters = scenario.parameters
    probe_legs = tuple(
        engine.ProbeLeg(
            segment_id=segment_id,
            # Present: _validate_arm rejected the arm otherwise.
            free_flow_traversal_s=links[segment_id].free_flow_traversal_s or 0.0,
        )
        for segment_id in arm.segment_ids
    )
    bypass = (
        parameters.emergency_queue_bypass_fraction
        if arm.mode is RoutingMode.EMERGENCY
        else 0.0
    )
    probe = engine.simulate_probe(
        probe_legs,
        traces,
        horizon_s=parameters.horizon_s,
        departure_offset_s=parameters.departure_offset_s,
        queue_bypass_fraction=bypass,
    )

    segments = [index.segment(segment_id) for segment_id in arm.segment_ids]
    legs = tuple(
        LegResult(
            segment_id=outcome.segment_id,
            from_junction_id=segment.from_junction_id,
            to_junction_id=segment.to_junction_id,
            entry_time_s=round(outcome.entry_time_s, 6),
            exit_time_s=round(outcome.exit_time_s, 6),
            free_flow_traversal_s=round(outcome.free_flow_traversal_s, 6),
            time_basis=links[outcome.segment_id].time_basis,
            queue_delay_s=(
                round(outcome.queue_delay_s, 6)
                if outcome.queue_delay_s is not None
                else None
            ),
            queue_modelled=outcome.modelled,
            queue_at_arrival_veh=(
                round(outcome.queue_at_arrival_veh, 6)
                if outcome.queue_at_arrival_veh is not None
                else None
            ),
            reason=(
                None
                if outcome.modelled
                else (
                    links[outcome.segment_id].capacity_reason
                    or links[outcome.segment_id].demand_reason
                    or "This link was not modelled, so no queueing is included here."
                )[:400]
            ),
        )
        for outcome, segment in zip(probe.legs, segments, strict=True)
    )

    unmodelled = tuple(
        outcome.segment_id for outcome in probe.legs if not outcome.modelled
    )
    distance_m = sum(segment.length_m.value for segment in segments)
    refs = (arm.route_id, *arm.segment_ids)
    junctions = (arm.origin_junction_id, *(s.to_junction_id for s in segments))

    warnings: list[str] = []
    if unmodelled:
        warnings.append(
            f"{len(unmodelled)} of {len(legs)} segment(s) on this route could not be "
            "modelled, so no queueing is included for them and the delay reported "
            f"here is a lower bound: {', '.join(unmodelled)}."
        )
    if bypass > 0.0:
        warnings.append(
            f"An emergency queue bypass of {bypass:.0%} was applied as a stated "
            "assumption: the responder is modelled as passing that fraction of each "
            "exit queue. Nothing in this system has measured that behaviour."
        )

    if not probe.completed_within_horizon:
        rejection = SimulationRejection(
            reason=UnavailableReason.HORIZON_EXCEEDED,
            message=(
                f"The vehicle had not arrived when the {parameters.horizon_s:g} s "
                "horizon ended, so no travel time is reported. A truncated journey "
                "presented as an arrival would understate it by exactly the part "
                "that was cut off; extend the horizon to obtain a figure."
            ),
            route_ids=(arm.route_id,),
        )
        return RouteSimulationResult(
            role=arm.role,
            route_id=arm.route_id,
            label=arm.label,
            source=arm.source,
            mode=arm.mode,
            status=ArmStatus.UNAVAILABLE,
            segment_ids=arm.segment_ids,
            junction_sequence=junctions,
            legs=legs,
            free_flow_travel_time_s=derived(
                probe.free_flow_travel_time_s,
                Unit.SECOND,
                at=at,
                input_refs=refs,
                notes="Sum of the legs' free-flow traversal times: an empty road.",
            ),
            total_distance_m=derived(
                distance_m, Unit.METER, at=at, input_refs=refs
            ),
            completed_within_horizon=False,
            unmodelled_segment_ids=unmodelled,
            modelled_leg_count=probe.modelled_leg_count,
            rejections=(rejection,),
            warnings=tuple(warnings),
        )

    # Present: completed_within_horizon is true.
    travel_time = probe.travel_time_s or 0.0
    completion = probe.completion_time_s or 0.0
    delay = round(travel_time - probe.free_flow_travel_time_s, 6)

    return RouteSimulationResult(
        role=arm.role,
        route_id=arm.route_id,
        label=arm.label,
        source=arm.source,
        mode=arm.mode,
        status=ArmStatus.SIMULATED,
        segment_ids=arm.segment_ids,
        junction_sequence=junctions,
        legs=legs,
        travel_time_s=simulated(
            travel_time,
            Unit.SECOND,
            at=at,
            input_refs=refs,
            notes=(
                "Modelled traversal under the scenario's demand: free-flow running "
                "plus waiting for each exit queue to discharge."
            ),
        ),
        free_flow_travel_time_s=derived(
            probe.free_flow_travel_time_s,
            Unit.SECOND,
            at=at,
            input_refs=refs,
            notes="Sum of the legs' free-flow traversal times: an empty road.",
        ),
        delay_s=simulated(
            max(delay, 0.0),
            Unit.SECOND,
            at=at,
            input_refs=refs,
            notes=(
                "Modelled travel time above the free-flow reference. A lower bound "
                "where some links could not be modelled."
            ),
        ),
        completion_time_s=simulated(
            completion,
            Unit.SECOND,
            at=at,
            input_refs=refs,
            notes=(
                "Arrival on the simulation clock: the departure offset plus the "
                "modelled travel time."
            ),
        ),
        mean_speed_kph=(
            simulated(
                distance_m / travel_time * 3.6,
                Unit.KILOMETER_PER_HOUR,
                at=at,
                input_refs=refs,
                notes="Route length over modelled travel time.",
            )
            if travel_time > 0
            else None
        ),
        total_distance_m=derived(distance_m, Unit.METER, at=at, input_refs=refs),
        completed_within_horizon=True,
        unmodelled_segment_ids=unmodelled,
        modelled_leg_count=probe.modelled_leg_count,
        warnings=tuple(warnings),
    )


def compare_arms(
    baseline: RouteSimulationResult,
    alternative: RouteSimulationResult,
    *,
    at: dt.datetime,
) -> SimulationComparison:
    """Difference the two arms, or state why the difference does not exist."""
    unavailable = [
        arm for arm in (baseline, alternative) if arm.status is not ArmStatus.SIMULATED
    ]
    if unavailable:
        names = ", ".join(f"{arm.role.value} ({arm.label})" for arm in unavailable)
        return SimulationComparison(
            baseline_route_id=baseline.route_id,
            alternative_route_id=alternative.route_id,
            comparable=False,
            reason=(
                f"No difference can be reported: {names} produced no travel time. "
                "A one-sided figure presented as an improvement is how a simulation "
                "invents a benefit."
            ),
            rejections=tuple(
                SimulationRejection(
                    reason=UnavailableReason.ARM_UNAVAILABLE,
                    message=(
                        f"Arm {arm.role.value!r} ({arm.route_id}) is unavailable: "
                        + " ".join(item.message for item in arm.rejections)
                    )[:500],
                    route_ids=(arm.route_id,),
                )
                for arm in unavailable
            ),
        )

    # Both arms are SIMULATED, so every figure below is present.
    base_travel = baseline.travel_time_s.value if baseline.travel_time_s else 0.0
    alt_travel = alternative.travel_time_s.value if alternative.travel_time_s else 0.0
    base_delay = baseline.delay_s.value if baseline.delay_s else 0.0
    alt_delay = alternative.delay_s.value if alternative.delay_s else 0.0
    base_completion = (
        baseline.completion_time_s.value if baseline.completion_time_s else 0.0
    )
    alt_completion = (
        alternative.completion_time_s.value if alternative.completion_time_s else 0.0
    )
    base_distance = baseline.total_distance_m.value if baseline.total_distance_m else 0.0
    alt_distance = (
        alternative.total_distance_m.value if alternative.total_distance_m else 0.0
    )

    delta = round(base_travel - alt_travel, 6)
    refs = (baseline.route_id, alternative.route_id)
    if abs(delta) < TIE_TOLERANCE_S:
        better = BetterArm.NO_DIFFERENCE
    elif delta > 0:
        better = BetterArm.ALTERNATIVE
    else:
        better = BetterArm.BASELINE

    return SimulationComparison(
        baseline_route_id=baseline.route_id,
        alternative_route_id=alternative.route_id,
        comparable=True,
        travel_time_delta_s=simulated(
            delta,
            Unit.SECOND,
            at=at,
            input_refs=refs,
            notes=(
                "Baseline minus alternative under identical demand, horizon and "
                "parameters. Positive means the alternative arrived sooner."
            ),
        ),
        delay_delta_s=simulated(
            round(base_delay - alt_delay, 6), Unit.SECOND, at=at, input_refs=refs
        ),
        completion_time_delta_s=simulated(
            round(base_completion - alt_completion, 6),
            Unit.SECOND,
            at=at,
            input_refs=refs,
        ),
        distance_delta_m=derived(
            round(base_distance - alt_distance, 6), Unit.METER, at=at, input_refs=refs
        ),
        improvement_pct=(
            round(delta / base_travel * 100.0, 3) if base_travel > 0 else None
        ),
        better=better,
    )


def run(
    scenario: SimulationScenario,
    index: NetworkIndex,
    *,
    at: dt.datetime,
    warnings: tuple[str, ...] = (),
) -> SimulationRun:
    """Evolve the network, drive every arm through it, and assemble the result.

    Pure with respect to time: ``at`` is supplied rather than read from a clock,
    so a test can pin the instant and two runs of identical inputs at the same
    instant are byte-identical.
    """
    links_by_id = {link.segment_id: link for link in scenario.links}
    runnable = modellable_links(scenario.links)
    traces = engine.simulate_network(
        runnable,
        step_s=scenario.parameters.step_s,
        step_count=scenario.parameters.step_count,
    )

    results = tuple(
        _simulate_arm(
            arm,
            index=index,
            links=links_by_id,
            traces=traces,
            scenario=scenario,
            at=at,
        )
        for arm in scenario.arms
    )

    link_results = tuple(
        _link_result(link, traces.get(link.segment_id), at=at) for link in scenario.links
    )

    comparison: SimulationComparison | None = None
    baseline = next((r for r in results if r.role is ArmRole.BASELINE), None)
    recommended = next((r for r in results if r.role is ArmRole.RECOMMENDED), None)
    if baseline is not None and recommended is not None:
        comparison = compare_arms(baseline, recommended, at=at)

    run_warnings = list(warnings)
    observed_links = [
        link.segment_id
        for link in scenario.links
        if link.demand_basis is InputBasis.OBSERVED
    ]
    if observed_links:
        run_warnings.append(
            f"{len(observed_links)} link(s) take their demand from an observed "
            "arrival rate into a bound observation area. That is an area appearance "
            "rate rather than a counting-line flow, so it estimates demand rather "
            "than measuring it."
        )
    unmodelled = [link.segment_id for link in scenario.links if not link.is_modellable]
    if unmodelled:
        run_warnings.append(
            f"{len(unmodelled)} of {len(scenario.links)} segment(s) could not be "
            "modelled for want of a demand, capacity or traversal-time basis. They "
            "carry no queue and contribute no delay, which understates congestion "
            "rather than reporting it as absent."
        )

    logger.info(
        "simulation run",
        extra={
            "run_id": f"sim-{scenario.input_fingerprint[:12]}",
            "fingerprint": scenario.input_fingerprint,
            "arms": len(results),
            "modelled_links": len(traces),
            "unmodelled_links": len(unmodelled),
        },
    )

    return SimulationRun(
        run_id=f"sim-{scenario.input_fingerprint[:12]}",
        status=SimulationStatus.COMPLETED,
        scenario=scenario,
        routes=results,
        links=link_results,
        network=_network_summary(scenario.links, traces, at=at),
        comparison=comparison,
        warnings=tuple(run_warnings),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        network_id=scenario.network_id,
        network_version=scenario.network_version,
        computed_at=at,
        provenance=_provenance(
            ValueSource.SIMULATED,
            at=at,
            input_refs=(scenario.scenario_id, scenario.input_fingerprint),
            notes=(
                "Modelled outcome of a hypothetical scenario. Not a measurement of "
                "any road at any time."
            ),
        ),
    )


def unavailable_run(
    scenario: SimulationScenario,
    *,
    at: dt.datetime,
    rejections: tuple[SimulationRejection, ...],
    warnings: tuple[str, ...] = (),
) -> SimulationRun:
    """A run that could not be built, carrying its causes rather than an empty result."""
    return SimulationRun(
        run_id=f"sim-{scenario.input_fingerprint[:12]}",
        status=SimulationStatus.UNAVAILABLE,
        scenario=scenario,
        rejections=rejections,
        warnings=warnings,
        network_id=scenario.network_id,
        network_version=scenario.network_version,
        computed_at=at,
        provenance=_provenance(
            ValueSource.SIMULATED,
            at=at,
            input_refs=(scenario.scenario_id,),
            notes="No modelled outcome was produced; the causes are in rejections.",
        ),
    )
