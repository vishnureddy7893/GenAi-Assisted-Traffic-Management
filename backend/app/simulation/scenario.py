"""Turning the system's existing facts into a scenario the model can run.

This is where the honesty of Phase 8 is actually enforced. The model in
:mod:`app.simulation.model` takes numbers; this module decides which numbers it
is allowed to have, and refuses to manufacture the ones nobody supplied.

Three sources, ranked and always recorded
-----------------------------------------
* **Surveyed** -- from the road network file. A segment's length, lane count and
  (where a survey recorded one) its capacity.
* **Observed** -- derived from measurements this system took, and attributed to a
  segment *only* through an explicit observation-to-network binding. An
  unbound camera contributes nothing here, exactly as it contributes nothing to
  a route cost.
* **Stated** -- an assumption supplied on the request. It must justify itself:
  the API requires a note, and the note travels into the result.

Where none of the three supplies a value, the link is reported unmodelled with
the reason. There is no fourth source called "reasonable default".

The traversal time comes from Phase 7, on purpose
-------------------------------------------------
:func:`app.routing.costing.traversal_time` already answers "how long does this
segment take, and on what basis?", including the refusal to invent a speed when
there is neither a measured travel time nor a surveyed free-flow one. Re-deriving
that here would create a second answer that could disagree with the optimizer's,
which is precisely what makes a simulated comparison worthless.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.analytics.schemas import TrafficAnalytics
from app.domain.network import RoadNetwork, RoadSegment
from app.domain.state import NetworkState, RoadSegmentState
from app.routing import costing
from app.routing.mapping import ObservationMapping
from app.routing.weights import RoutingMode
from app.simulation.model import Link
from app.simulation.schemas import (
    InputBasis,
    LinkInputs,
    RouteArmInput,
    SimulationParameters,
    SimulationScenario,
)


@dataclass(frozen=True, slots=True)
class StatedSegmentInput:
    """An operator's stated assumption about one segment.

    ``note`` is not optional. A stated demand is carried as ``ASSUMED``
    provenance downstream, and the domain rule for an assumption is that it
    justifies itself -- an unjustified assumption is indistinguishable from a
    fabricated measurement.
    """

    segment_id: str
    note: str
    demand_veh_per_hour: float | None = None
    capacity_veh_per_hour: float | None = None
    initial_queue_veh: float | None = None


@dataclass(frozen=True, slots=True)
class ObservedSegmentInput:
    """What an observation bound to a segment can contribute to a scenario."""

    segment_id: str
    observation_area_id: str
    demand_veh_per_hour: float | None = None
    demand_note: str | None = None
    initial_queue_veh: float | None = None
    initial_queue_note: str | None = None


#: The caveat that travels with every demand derived from an arrival rate. It is
#: repeated in full rather than summarised because the difference it names --
#: between an area appearance rate and a cross-section flow -- is exactly the
#: kind of thing that gets lost when a number is copied into a slide.
ARRIVAL_RATE_CAVEAT = (
    "Derived from the observed arrival rate into the bound observation area "
    "(objects first seen per minute, x60). That is an area appearance rate, not "
    "a counting-line flow: it is direction-agnostic and counts an object once "
    "however it moved. Used as link demand because the survey binds this camera "
    "to this segment; treat it as an estimate of demand, not a measured flow."
)


def observed_inputs(
    analytics: Iterable[TrafficAnalytics], mapping: ObservationMapping | None
) -> dict[str, ObservedSegmentInput]:
    """Resolve observed contributions per segment through the mapping.

    Returns an empty mapping when no mapping is loaded. That is a working state,
    not a broken one: every link is then simply unobserved, and the scenario says
    so rather than attributing a camera's numbers to a road nobody surveyed it
    against.
    """
    if mapping is None:
        return {}

    by_area = {item.observation_area_id: item for item in analytics}
    resolved: dict[str, ObservedSegmentInput] = {}

    for segment_id, bindings in sorted(mapping.bindings_by_segment().items()):
        # Deterministic when several cameras watch one segment: the lowest area
        # id wins, and the rest stay visible in the mapping for a reviewer.
        area_id = bindings[0].observation_area_id
        measured = by_area.get(area_id)
        if measured is None:
            continue

        demand: float | None = None
        demand_note: str | None = None
        rate = measured.arrival_rate_per_minute
        if rate.is_available and rate.value is not None:
            demand = round(rate.value * 60.0, 6)
            demand_note = ARRIVAL_RATE_CAVEAT

        queue: float | None = None
        queue_note: str | None = None
        stationary = measured.stationary_vehicle_count
        if stationary.is_available and stationary.value is not None:
            queue = float(stationary.value)
            queue_note = (
                "Initial queue taken from the observed count of stationary road "
                f"users in observation area {area_id!r}, bound to this segment by "
                "the loaded survey."
            )

        if demand is None and queue is None:
            continue

        resolved[segment_id] = ObservedSegmentInput(
            segment_id=segment_id,
            observation_area_id=area_id,
            demand_veh_per_hour=demand,
            demand_note=demand_note,
            initial_queue_veh=queue,
            initial_queue_note=queue_note,
        )

    return resolved


def _capacity_for(
    segment: RoadSegment, stated: StatedSegmentInput | None
) -> tuple[float | None, InputBasis, str | None]:
    """Resolve a link's exit capacity, preferring a stated override."""
    surveyed = segment.capacity_veh_per_hour
    if stated is not None and stated.capacity_veh_per_hour is not None:
        note = f"Capacity stated on the request: {stated.note}"
        if surveyed is not None:
            note = (
                f"Capacity stated on the request, overriding the surveyed "
                f"{surveyed.value:.0f} veh/h: {stated.note}"
            )
        return stated.capacity_veh_per_hour, InputBasis.STATED, note[:400]
    if surveyed is not None:
        return (
            surveyed.value,
            InputBasis.SURVEYED,
            "Capacity as recorded in the road network survey.",
        )
    return (
        None,
        InputBasis.UNAVAILABLE,
        "This segment has no surveyed capacity and none was stated on the "
        "request, so its exit cannot discharge at a known rate and no queue is "
        "modelled here. No capacity was assumed in its place.",
    )


def _demand_for(
    stated: StatedSegmentInput | None,
    observed: ObservedSegmentInput | None,
    *,
    use_observed: bool,
) -> tuple[float | None, InputBasis, str | None]:
    """Resolve a link's demand: stated first, then observed, then nothing."""
    if stated is not None and stated.demand_veh_per_hour is not None:
        return (
            stated.demand_veh_per_hour,
            InputBasis.STATED,
            f"Demand stated on the request: {stated.note}"[:400],
        )
    if use_observed and observed is not None and observed.demand_veh_per_hour is not None:
        return observed.demand_veh_per_hour, InputBasis.OBSERVED, observed.demand_note
    if not use_observed and observed is not None:
        return (
            None,
            InputBasis.UNAVAILABLE,
            "An observation is bound to this segment but observed demand is "
            "switched off for this run, so no demand is applied here.",
        )
    return (
        None,
        InputBasis.UNAVAILABLE,
        "No demand is bound to this segment and none was stated on the request, "
        "so no traffic is released onto it. This is an absence of input, not a "
        "finding that the road is empty.",
    )


def _initial_queue_for(
    stated: StatedSegmentInput | None, observed: ObservedSegmentInput | None
) -> tuple[float | None, InputBasis, str | None]:
    """Resolve the queue standing on a link when the horizon opens."""
    if stated is not None and stated.initial_queue_veh is not None:
        return (
            stated.initial_queue_veh,
            InputBasis.STATED,
            f"Initial queue stated on the request: {stated.note}"[:400],
        )
    if observed is not None and observed.initial_queue_veh is not None:
        return observed.initial_queue_veh, InputBasis.OBSERVED, observed.initial_queue_note
    return (
        None,
        InputBasis.UNAVAILABLE,
        "No initial queue is known for this segment. The run opens it empty as "
        "a stated boundary condition of the model, which understates delay "
        "wherever a queue was in fact already standing.",
    )


def build_links(
    network: RoadNetwork,
    *,
    network_state: NetworkState | None,
    mode: RoutingMode,
    stated: Sequence[StatedSegmentInput] = (),
    observed: dict[str, ObservedSegmentInput] | None = None,
    use_observed_demand: bool = True,
) -> tuple[LinkInputs, ...]:
    """Resolve every segment's simulation inputs, and where each one came from."""
    stated_by_segment = {item.segment_id: item for item in stated}
    observed_by_segment = observed or {}
    segment_states: dict[str, RoadSegmentState] = {
        item.segment_id: item
        for item in (network_state.segment_states if network_state else ())
    }

    links: list[LinkInputs] = []
    for segment in sorted(network.segments, key=lambda item: item.segment_id):
        state = segment_states.get(segment.segment_id)
        time = costing.traversal_time(segment, state, mode=mode)
        stated_input = stated_by_segment.get(segment.segment_id)
        observed_input = observed_by_segment.get(segment.segment_id)

        capacity, capacity_basis, capacity_reason = _capacity_for(segment, stated_input)
        demand, demand_basis, demand_reason = _demand_for(
            stated_input, observed_input, use_observed=use_observed_demand
        )
        queue, queue_basis, queue_reason = _initial_queue_for(
            stated_input, observed_input
        )

        links.append(
            LinkInputs(
                segment_id=segment.segment_id,
                length_m=segment.length_m.value,
                lane_count=segment.lane_count,
                free_flow_traversal_s=(
                    round(time.seconds, 6) if time.seconds is not None else None
                ),
                time_basis=time.basis,
                time_reason=time.reason[:400] if time.reason else None,
                capacity_veh_per_hour=capacity,
                capacity_basis=capacity_basis,
                capacity_reason=capacity_reason,
                demand_veh_per_hour=demand,
                demand_basis=demand_basis,
                demand_reason=demand_reason,
                initial_queue_veh=queue,
                initial_queue_basis=queue_basis,
                initial_queue_reason=queue_reason,
                observation_area_id=(
                    observed_input.observation_area_id if observed_input else None
                ),
                is_blocked=bool(state.is_blocked) if state else False,
                blocked_by_incident_ids=(
                    tuple(sorted(state.blocked_by_incident_ids)) if state else ()
                ),
            )
        )
    return tuple(links)


def modellable_links(links: Iterable[LinkInputs]) -> dict[str, Link]:
    """The subset the model can actually evolve, keyed by segment id.

    A link missing any of its three inputs is simply absent from this mapping.
    The caller reports it as unmodelled with the reason already recorded on the
    :class:`~app.simulation.schemas.LinkInputs`; nothing is substituted.
    """
    out: dict[str, Link] = {}
    for link in links:
        traversal = link.free_flow_traversal_s
        capacity = link.capacity_veh_per_hour
        demand = link.demand_veh_per_hour
        if traversal is None or capacity is None or demand is None:
            continue
        out[link.segment_id] = Link(
            segment_id=link.segment_id,
            free_flow_traversal_s=traversal,
            capacity_veh_per_hour=capacity,
            demand_veh_per_hour=demand,
            # An unknown initial queue opens the link empty. Recorded on the
            # link as a stated boundary condition, never as a measurement.
            initial_queue_veh=link.initial_queue_veh or 0.0,
        )
    return out


def compute_fingerprint(payload: dict[str, object]) -> str:
    """SHA-256 over a canonical rendering of everything that changes the answer.

    Sorted keys and a compact separator, so two structurally identical scenarios
    hash identically regardless of the order their parts were assembled in.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_scenario(
    *,
    network: RoadNetwork,
    parameters: SimulationParameters,
    links: tuple[LinkInputs, ...],
    arms: tuple[RouteArmInput, ...],
    traffic_state_id: str | None = None,
    mapping_id: str | None = None,
    mapping_version: str | None = None,
    active_incident_ids: tuple[str, ...] = (),
) -> SimulationScenario:
    """Assemble the scenario and derive its identity from its content.

    The identifier is the fingerprint, not a random string. Two requests with
    identical inputs therefore address the same run, which is what makes
    "identical input produces identical output" checkable by a caller rather
    than only by a test.
    """
    # The traffic-state id is deliberately absent from the hashed body while
    # being recorded on the scenario. It is *context*, not an input: the
    # point-queue model never reads an interpreted congestion band. What an
    # observation contributes -- a demand, an initial queue -- has already been
    # resolved onto the links above and is hashed there. The id itself embeds
    # the instant its snapshot was assembled, so hashing it would make two
    # identical scenarios differ purely because a clock moved, which is exactly
    # the false negative the fingerprint exists to rule out.
    body = {
        "network_id": network.network_id,
        "network_version": network.version,
        "parameters": parameters.model_dump(mode="json"),
        "links": [item.model_dump(mode="json") for item in links],
        "arms": [item.model_dump(mode="json") for item in arms],
        "mapping_id": mapping_id,
        "mapping_version": mapping_version,
        "active_incident_ids": sorted(active_incident_ids),
    }
    digest = compute_fingerprint(body)
    return SimulationScenario(
        scenario_id=f"scn-{digest[:12]}",
        network_id=network.network_id,
        network_version=network.version,
        parameters=parameters,
        links=links,
        arms=arms,
        traffic_state_id=traffic_state_id,
        mapping_id=mapping_id,
        mapping_version=mapping_version,
        active_incident_ids=tuple(sorted(active_incident_ids)),
        input_fingerprint=digest,
    )
