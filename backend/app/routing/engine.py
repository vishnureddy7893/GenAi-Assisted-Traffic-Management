"""Route optimization: request in, structured result out.

The order of operations is the safety argument
----------------------------------------------
Prune, then search, then price, then rank. Hard constraints remove elements
before any path is considered, so no candidate can violate one -- not because
the ranking rejects it, but because it was never reachable. Everything soft is
priced into a cost the ranking can be argued with term by term.

Two ranking bases, never mixed
------------------------------
The primary search runs on the subgraph of segments that have a **time basis**,
and prices routes in equivalent seconds. If that subgraph has no path -- because
some segment on every route has neither a travel-time estimate nor a surveyed
free-flow speed -- the search is repeated over the full admissible subgraph
ranked by **distance**, every route reports no ETA, and the result says exactly
that. What does not happen is a fabricated seconds-per-metre bridging the two:
that is an invented speed under a different name, and it would make an
unmeasurable route look measured.

Nothing here writes prose
-------------------------
Reasons are codes with numbers attached. A sentence explaining the choice is
Phase 12's job, generated against a fact table; this module produces the facts.
"""

from __future__ import annotations

import datetime as dt
import uuid

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.enums import EmergencyVehicleType, MovementType
from app.domain.incident import Incident
from app.domain.network import NetworkIndex, RoadNetwork
from app.domain.provenance import FloatMeasurement, Provenance, Unit, ValueSource
from app.domain.routing import (
    RouteCandidate,
    RouteComparison,
    RouteCostBreakdown,
    RouteLeg,
    TimeBasis,
)
from app.domain.state import NetworkState
from app.routing import costing
from app.routing.mapping import ObservationMapping
from app.routing.pruning import AdmissibleSubgraph, build_subgraph
from app.routing.schemas import (
    EvidenceStatus,
    OptimizationStatus,
    RankedRoute,
    ReasonCode,
    RejectionReason,
    RouteOptimizationResult,
    RouteReason,
    RouteRecommendation,
    RouteRejection,
    RouteRequest,
    SegmentEvidence,
)
from app.routing.search import (
    ALGORITHM,
    ALGORITHM_VERSION,
    Path,
    SearchLimitReachedError,
    SearchNode,
    k_shortest_paths,
    shortest_path,
)
from app.routing.weights import RouteWeights, RoutingMode, weights_for
from app.state.schemas import IntersectionStateElement, NetworkTrafficState

logger = get_logger(__name__)

PRODUCER = "routing.optimizer"


class _LegCost:
    """The priced parts of one leg, retained so the total can be taken apart."""

    __slots__ = (
        "base_s",
        "congestion_s",
        "distance_s",
        "evidence",
        "incident_s",
        "segment_id",
        "time",
        "turn_s",
        "uncertainty_s",
    )

    def __init__(
        self,
        segment_id: str,
        *,
        time: costing.TraversalTime,
        evidence: SegmentEvidence,
        turn_s: float,
        distance_s: float,
    ) -> None:
        self.segment_id = segment_id
        self.time = time
        self.evidence = evidence
        self.base_s = time.seconds or 0.0
        self.congestion_s = evidence.congestion_penalty_s
        self.uncertainty_s = evidence.uncertainty_penalty_s
        self.incident_s = evidence.incident_penalty_s
        self.turn_s = turn_s
        self.distance_s = distance_s

    @property
    def priced_total_s(self) -> float:
        """Everything this leg contributes to a time-based ranking."""
        return (
            self.base_s
            + self.congestion_s
            + self.uncertainty_s
            + self.incident_s
            + self.turn_s
            + self.distance_s
        )


class RouteOptimizer:
    """Prices and ranks routes over one network snapshot.

    Constructed per request rather than held, because it closes over the network
    state, the incidents and the traffic evidence of one instant. Reusing an
    instance across snapshots would let a route be priced against conditions
    that had already changed.
    """

    def __init__(
        self,
        network: RoadNetwork,
        *,
        settings: Settings,
        mode: RoutingMode,
        network_state: NetworkState | None = None,
        incidents: tuple[Incident, ...] = (),
        traffic_state: NetworkTrafficState | None = None,
        mapping: ObservationMapping | None = None,
        vehicle_type: EmergencyVehicleType | None = None,
        index: NetworkIndex | None = None,
    ) -> None:
        self.network = network
        self.index = index or network.build_index()
        self.settings = settings
        self.weights: RouteWeights = weights_for(settings, mode)
        self.mode = mode
        self.network_state = network_state
        self.incidents = incidents
        self.traffic_state = traffic_state
        self.mapping = mapping
        self.subgraph: AdmissibleSubgraph = build_subgraph(
            network,
            network_state=network_state,
            incidents=incidents,
            vehicle_type=vehicle_type,
        )

        self._segment_states = {
            state.segment_id: state
            for state in (network_state.segment_states if network_state else ())
        }
        self._degraded = self._degraded_segments()
        self._evidence = self._build_evidence()

    # -- evidence -------------------------------------------------------
    def _degraded_segments(self) -> dict[str, tuple[str, ...]]:
        """Segments an active incident degrades without blocking."""
        degraded: dict[str, list[str]] = {}
        for incident in self.incidents:
            if not incident.is_active:
                continue
            for segment_id in incident.affected_segment_ids:
                degraded.setdefault(segment_id, []).append(incident.incident_id)
        return {key: tuple(sorted(value)) for key, value in degraded.items()}

    def _elements_by_area(self) -> dict[str, IntersectionStateElement]:
        if self.traffic_state is None:
            return {}
        return {
            element.state.observation_area_id: element
            for element in self.traffic_state.elements
        }

    def _build_evidence(self) -> dict[str, SegmentEvidence]:
        """Price every segment's traffic evidence once, up front.

        Computed for the whole network rather than lazily per candidate, so the
        cost function stays a pure lookup and two candidates sharing a segment
        cannot possibly price it differently.
        """
        elements = self._elements_by_area()
        by_segment = self.mapping.bindings_by_segment() if self.mapping else {}

        evidence: dict[str, SegmentEvidence] = {}
        for segment in self.network.segments:
            bindings = by_segment.get(segment.segment_id, ())
            element: IntersectionStateElement | None = None
            area_id: str | None = None
            if bindings:
                # Deterministic when a segment is watched by several cameras:
                # the lowest area id wins, and the others are still listed in
                # the mapping for a reviewer to see.
                area_id = bindings[0].observation_area_id
                element = elements.get(area_id)

            evidence[segment.segment_id] = costing.segment_evidence(
                segment.segment_id,
                element,
                area_id,
                weights=self.weights,
                degraded_by=self._degraded.get(segment.segment_id, ()),
            )
        return evidence

    def unmapped_area_ids(self) -> tuple[str, ...]:
        """Interpreted observation areas that are bound to no segment."""
        if self.traffic_state is None:
            return ()
        observed = {
            element.state.observation_area_id for element in self.traffic_state.elements
        }
        mapped = self.mapping.area_ids if self.mapping else frozenset()
        return tuple(sorted(observed - mapped))

    # -- costing --------------------------------------------------------
    def _turn_penalty(self, node: SearchNode, segment_id: str) -> float:
        junction_id, arriving = node
        if not arriving:
            return 0.0
        movement = self.subgraph.movement(junction_id, arriving, segment_id)
        if movement is None:
            return 0.0
        if movement.movement_type is MovementType.U_TURN:
            return self.weights.u_turn_penalty_s
        if movement.movement_type is MovementType.THROUGH:
            return 0.0
        return self.weights.turn_penalty_s

    def leg_cost(self, node: SearchNode, segment_id: str) -> _LegCost:
        """Everything one leg contributes, itemised."""
        segment = self.index.segment(segment_id)
        time = costing.traversal_time(
            segment, self._segment_states.get(segment_id), mode=self.mode
        )
        return _LegCost(
            segment_id,
            time=time,
            evidence=self._evidence[segment_id],
            turn_s=self._turn_penalty(node, segment_id),
            distance_s=costing.distance_penalty(segment, self.weights),
        )

    def time_edge_cost(self, node: SearchNode, segment_id: str) -> float | None:
        """Edge weight in equivalent seconds, or ``None`` with no time basis."""
        cost = self.leg_cost(node, segment_id)
        if not cost.time.is_available:
            return None
        return cost.priced_total_s

    def distance_edge_cost(self, node: SearchNode, segment_id: str) -> float | None:
        """Edge weight in metres, for the fallback ranking."""
        del node
        return self.index.segment(segment_id).length_m.value

    # -- candidate construction -----------------------------------------
    def build_candidate(
        self,
        path: Path,
        *,
        origin_junction_id: str,
        rank: int,
        request_id: str,
        at: dt.datetime,
    ) -> tuple[RouteCandidate, RankedRoute]:
        """Turn a path into a priced candidate and its derived view."""
        legs: list[RouteLeg] = []
        costs: list[_LegCost] = []
        node: SearchNode = (origin_junction_id, "")

        for segment_id in path:
            segment = self.index.segment(segment_id)
            cost = self.leg_cost(node, segment_id)
            costs.append(cost)
            movement = self.subgraph.movement(node[0], node[1], segment_id) if node[1] else None
            legs.append(
                RouteLeg(
                    segment_id=segment_id,
                    from_junction_id=segment.from_junction_id,
                    to_junction_id=segment.to_junction_id,
                    entry_movement_id=movement.movement_id if movement else None,
                    length_m=segment.length_m,
                    traversal_time_s=costing.time_measurement(cost.time, segment, at=at),
                    time_basis=cost.time.basis,
                )
            )
            node = (segment.to_junction_id, segment_id)

        breakdown = RouteCostBreakdown(
            base_travel_time_s=round(sum(c.base_s for c in costs), 6),
            congestion_penalty_s=round(sum(c.congestion_s for c in costs), 6),
            incident_penalty_s=round(sum(c.incident_s for c in costs), 6),
            turn_penalty_s=round(sum(c.turn_s for c in costs), 6),
            uncertainty_penalty_s=round(sum(c.uncertainty_s for c in costs), 6),
            distance_penalty_s=round(sum(c.distance_s for c in costs), 6),
            weights_version=self.weights.weights_version,
        )

        unpriced = tuple(c.segment_id for c in costs if not c.time.is_available)
        warnings = self._route_warnings(costs, unpriced)

        candidate = RouteCandidate(
            route_id=f"route-{request_id}-{rank}",
            legs=tuple(legs),
            cost=breakdown,
            soft_constraints=(),
            warnings=warnings,
            emergency_assumptions=self._emergency_assumptions(costs),
            computed_at=at,
            provenance=Provenance(
                source=ValueSource.DERIVED,
                producer=PRODUCER,
                producer_version=ALGORITHM_VERSION,
                observed_at=at,
                computed_at=at,
                input_refs=(
                    self.network.network_id,
                    *(c.segment_id for c in costs),
                ),
            ),
        )

        evidence = tuple(c.evidence for c in costs)
        ranked = RankedRoute(
            rank=rank,
            candidate=candidate,
            estimated_travel_time_s=(
                round(sum(c.base_s for c in costs), 3) if not unpriced else None
            ),
            travel_time_available=not unpriced,
            time_bases=tuple(c.time.basis for c in costs),
            unpriced_segment_ids=unpriced,
            total_distance_m=round(candidate.total_length_m, 3),
            junction_sequence=candidate.junction_sequence,
            segment_sequence=path,
            evidence=evidence,
            mapped_segment_count=sum(
                1 for item in evidence if item.status is not EvidenceStatus.UNMAPPED
            ),
        )
        return candidate, ranked

    def _route_warnings(
        self, costs: list[_LegCost], unpriced: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Operator-facing warnings for one route."""
        warnings: list[str] = []
        if unpriced:
            warnings.append(
                f"{len(unpriced)} segment(s) have no travel-time basis "
                f"({', '.join(unpriced)}), so this route has no estimated travel "
                "time. No speed was assumed in their place."
            )
        derived = [c for c in costs if c.time.basis is TimeBasis.DERIVED_FROM_FREE_FLOW]
        if derived:
            warnings.append(
                f"{len(derived)} of {len(costs)} segment(s) are timed from surveyed "
                "free-flow speed rather than a measured travel time. That describes "
                "an empty road and is optimistic wherever there is traffic."
            )
        unmapped = [c for c in costs if c.evidence.status is EvidenceStatus.UNMAPPED]
        if unmapped:
            warnings.append(
                f"{len(unmapped)} of {len(costs)} segment(s) have no observation "
                "bound to them and are priced as unobserved, not as clear."
            )
        stale = [c for c in costs if c.evidence.status is EvidenceStatus.MAPPED_STALE]
        if stale:
            warnings.append(
                f"{len(stale)} segment(s) rest on evidence past the staleness budget."
            )
        degraded = [c for c in costs if c.incident_s > 0]
        if degraded:
            warnings.append(
                f"{len(degraded)} segment(s) are degraded by an active incident but "
                "remain passable, so they are priced rather than pruned."
            )
        return tuple(warnings)

    def _emergency_assumptions(self, costs: list[_LegCost]) -> tuple[str, ...]:
        """Conditions this route's timing depends on."""
        if self.mode is not RoutingMode.EMERGENCY:
            return ()
        has_emergency_basis = any(
            c.time.basis is TimeBasis.MEASURED_EMERGENCY for c in costs
        )
        if has_emergency_basis:
            return (
                "Timed partly from emergency-specific travel-time estimates, whose "
                "own assumptions travel with those estimates.",
            )
        return (
            "No emergency-specific travel time exists for any segment on this route. "
            "Times are the same as for general traffic, so no emergency advantage is "
            "modelled and the estimate is conservative: a responder receiving "
            "priority may arrive sooner than stated. Nominal time was not scaled by "
            "a constant, because that is optimistic in free flow and badly "
            "pessimistic in the congestion that decides the route.",
        )


def optimize(
    request: RouteRequest,
    optimizer: RouteOptimizer,
    *,
    destination_junction_id: str,
    at: dt.datetime,
) -> RouteOptimizationResult:
    """Run the search and assemble the structured result."""
    network = optimizer.network
    index = optimizer.index
    subgraph = optimizer.subgraph
    rejections: list[RouteRejection] = list(subgraph.rejections)
    warnings: list[str] = []

    max_candidates = request.max_candidates or optimizer.weights.max_candidates

    def result(
        status: OptimizationStatus,
        *,
        routes: tuple[RankedRoute, ...] = (),
        comparison: RouteComparison | None = None,
        recommendation: RouteRecommendation | None = None,
    ) -> RouteOptimizationResult:
        return RouteOptimizationResult(
            request=request,
            status=status,
            routes=routes,
            comparison=comparison,
            recommendation=recommendation,
            rejections=tuple(rejections),
            pruned_segment_ids=tuple(sorted(subgraph.pruned_segment_ids)),
            unmapped_observation_area_ids=optimizer.unmapped_area_ids(),
            warnings=tuple(warnings),
            weights=optimizer.weights,
            algorithm=ALGORITHM,
            algorithm_version=ALGORITHM_VERSION,
            network_id=network.network_id,
            network_version=network.version,
            mapping_id=optimizer.mapping.mapping_id if optimizer.mapping else None,
            mapping_version=optimizer.mapping.version if optimizer.mapping else None,
            traffic_state_id=(
                optimizer.traffic_state.state_id if optimizer.traffic_state else None
            ),
            computed_at=at,
            provenance=Provenance(
                source=ValueSource.DERIVED,
                producer=PRODUCER,
                producer_version=ALGORITHM_VERSION,
                observed_at=at,
                computed_at=at,
                input_refs=(network.network_id, network.version),
            ),
        )

    if optimizer.mapping is None:
        warnings.append(
            "No observation mapping is loaded, so no camera is bound to a road "
            "segment. Traffic evidence, where it exists, cannot be attributed to any "
            "element of the network and every segment is priced as unobserved."
        )
    if unmapped := optimizer.unmapped_area_ids():
        warnings.append(
            f"{len(unmapped)} observation area(s) have interpreted traffic state but "
            "are bound to no segment, so their evidence was not used: "
            f"{', '.join(unmapped)}."
        )

    # Primary: rank in equivalent seconds over segments that have a time basis.
    try:
        paths = k_shortest_paths(
            index,
            subgraph,
            origin_junction_id=request.origin_junction_id,
            destination_junction_id=destination_junction_id,
            edge_cost=optimizer.time_edge_cost,
            k=max_candidates,
            max_settled=optimizer.weights.max_settled_nodes,
        )
    except SearchLimitReachedError:
        rejections.append(
            RouteRejection(
                reason=RejectionReason.SEARCH_LIMIT_REACHED,
                message=(
                    f"The search settled more than {optimizer.weights.max_settled_nodes} "
                    "nodes without reaching the destination and was stopped."
                ),
            )
        )
        return result(OptimizationStatus.NO_ADMISSIBLE_ROUTE)

    ranked_by_distance = False
    if not paths:
        # Fallback: some segment on every route has no time basis. Rank by
        # distance and report no ETA, rather than inventing a speed.
        try:
            paths = k_shortest_paths(
                index,
                subgraph,
                origin_junction_id=request.origin_junction_id,
                destination_junction_id=destination_junction_id,
                edge_cost=optimizer.distance_edge_cost,
                k=max_candidates,
                max_settled=optimizer.weights.max_settled_nodes,
            )
        except SearchLimitReachedError:
            paths = []
        if paths:
            ranked_by_distance = True
            warnings.append(
                "No route exists on which every segment has a travel-time basis, so "
                "routes are ranked by distance and no estimated travel time is "
                "reported. No speed was assumed to bridge the gap."
            )

    if not paths:
        rejections.append(_diagnose(optimizer, request, destination_junction_id))
        return result(OptimizationStatus.NO_ADMISSIBLE_ROUTE)

    candidates: list[RouteCandidate] = []
    routes: list[RankedRoute] = []
    for rank, (path, _cost) in enumerate(paths, start=1):
        candidate, ranked = optimizer.build_candidate(
            path,
            origin_junction_id=request.origin_junction_id,
            rank=rank,
            request_id=request.request_id,
            at=at,
        )
        candidates.append(candidate)
        routes.append(ranked)

    selected = candidates[0]
    comparison = RouteComparison(
        candidates=tuple(candidates),
        selected_route_id=selected.route_id,
        ranking_criterion=(
            "minimum total distance (no travel-time basis available)"
            if ranked_by_distance
            else "minimum risk-adjusted arrival time (equivalent seconds)"
        ),
        pruned_segment_ids=subgraph.pruned_segment_ids,
    )
    recommendation = _recommend(routes, ranked_by_distance=ranked_by_distance)

    logger.info(
        "route optimized",
        extra={
            "request_id": request.request_id,
            "mode": request.mode.value,
            "candidates": len(routes),
            "selected": selected.route_id,
            "total_cost_s": selected.cost.total_s,
            "ranked_by_distance": ranked_by_distance,
        },
    )
    return result(
        OptimizationStatus.ROUTES_FOUND,
        routes=tuple(routes),
        comparison=comparison,
        recommendation=recommendation,
    )


def _diagnose(
    optimizer: RouteOptimizer, request: RouteRequest, destination_junction_id: str
) -> RouteRejection:
    """Work out *why* nothing was found, which is the actionable part.

    "Disconnected by pruning" and "disconnected map" look identical to a caller
    and call for opposite responses: clear the incident, or fix the survey.
    """
    unrestricted = build_subgraph(optimizer.network)
    try:
        raw = shortest_path(
            optimizer.index,
            unrestricted,
            origin_junction_id=request.origin_junction_id,
            destination_junction_id=destination_junction_id,
            edge_cost=lambda node, segment_id: optimizer.distance_edge_cost(
                node, segment_id
            ),
            max_settled=optimizer.weights.max_settled_nodes,
        )
    except SearchLimitReachedError:
        raw = None

    if raw is None:
        return RouteRejection(
            reason=RejectionReason.NETWORK_DISCONNECTED,
            message=(
                f"No path exists from {request.origin_junction_id!r} to "
                f"{destination_junction_id!r} even with every constraint lifted. The "
                "network as surveyed does not connect these points."
            ),
            junction_ids=(request.origin_junction_id, destination_junction_id),
        )

    blocking = tuple(
        segment_id
        for segment_id in raw[0]
        if segment_id in optimizer.subgraph.pruned_segment_ids
    )
    incident_ids = tuple(
        sorted(
            {
                incident_id
                for segment_id in blocking
                for incident_id in optimizer.subgraph.blocking_incident_ids(segment_id)
            }
        )
    )
    return RouteRejection(
        reason=RejectionReason.DISCONNECTED_BY_PRUNING,
        message=(
            "A path exists on the network as surveyed, but every route crosses a "
            "segment removed by a hard constraint. No unsafe route is returned in "
            "its place; clearing the cause restores the connection."
        ),
        segment_ids=tuple(sorted(set(blocking) or optimizer.subgraph.pruned_segment_ids)),
        junction_ids=(request.origin_junction_id, destination_junction_id),
        incident_ids=incident_ids,
    )


def _recommend(
    routes: list[RankedRoute], *, ranked_by_distance: bool
) -> RouteRecommendation:
    """Explain the winner in structured, checkable terms."""
    winner = routes[0]
    runner_up = routes[1] if len(routes) > 1 else None
    reasons: list[RouteReason] = [
        RouteReason(
            code=ReasonCode.NO_HARD_BLOCKED_SEGMENTS,
            detail=(
                "Every candidate was generated on the subgraph left after hard "
                "constraints were applied, so no candidate crosses a blocked or "
                "excluded segment."
            ),
        )
    ]

    if ranked_by_distance:
        reasons.append(
            RouteReason(
                code=ReasonCode.TRAVEL_TIME_UNAVAILABLE_RANKED_BY_DISTANCE,
                detail=(
                    "No route had a travel-time basis on every segment, so ranking "
                    "fell back to total distance and no ETA is reported."
                ),
                value=winner.total_distance_m,
                comparison_value=runner_up.total_distance_m if runner_up else None,
            )
        )

    if runner_up is None:
        reasons.append(
            RouteReason(
                code=ReasonCode.ONLY_ADMISSIBLE_ROUTE,
                detail=(
                    "This was the only admissible route found; it wins by default "
                    "rather than by comparison."
                ),
            )
        )
        return RouteRecommendation(
            selected_route_id=winner.candidate.route_id,
            reasons=tuple(reasons),
            alternatives_considered=0,
            cost_margin_s=None,
        )

    win_cost = winner.candidate.cost
    other_cost = runner_up.candidate.cost
    margin = round(other_cost.total_s - win_cost.total_s, 6)

    if not ranked_by_distance:
        reasons.append(
            RouteReason(
                code=ReasonCode.LOWER_TOTAL_COST,
                detail=(
                    "Lowest risk-adjusted arrival time among the admissible routes, "
                    "in equivalent seconds."
                ),
                value=round(win_cost.total_s, 3),
                comparison_value=round(other_cost.total_s, 3),
            )
        )
    if win_cost.congestion_penalty_s < other_cost.congestion_penalty_s:
        reasons.append(
            RouteReason(
                code=ReasonCode.LOWER_CONGESTION_COST,
                detail="Less congestion priced against it than the runner-up.",
                value=round(win_cost.congestion_penalty_s, 3),
                comparison_value=round(other_cost.congestion_penalty_s, 3),
            )
        )
    if win_cost.incident_penalty_s < other_cost.incident_penalty_s:
        reasons.append(
            RouteReason(
                code=ReasonCode.LOWER_INCIDENT_COST,
                detail="Less incident degradation priced against it.",
                value=round(win_cost.incident_penalty_s, 3),
                comparison_value=round(other_cost.incident_penalty_s, 3),
            )
        )
    if (
        winner.travel_time_available
        and runner_up.travel_time_available
        and winner.estimated_travel_time_s is not None
        and runner_up.estimated_travel_time_s is not None
        and winner.estimated_travel_time_s < runner_up.estimated_travel_time_s
    ):
        reasons.append(
            RouteReason(
                code=ReasonCode.LOWER_TRAVEL_TIME,
                detail="Shorter estimated traversal before any penalty is applied.",
                value=winner.estimated_travel_time_s,
                comparison_value=runner_up.estimated_travel_time_s,
            )
        )
    if winner.total_distance_m < runner_up.total_distance_m:
        reasons.append(
            RouteReason(
                code=ReasonCode.SHORTER_DISTANCE,
                detail="Shorter in metres. Reported, not optimised for.",
                value=winner.total_distance_m,
                comparison_value=runner_up.total_distance_m,
            )
        )
    if winner.mapped_segment_count > runner_up.mapped_segment_count:
        reasons.append(
            RouteReason(
                code=ReasonCode.BETTER_EVIDENCE_COVERAGE,
                detail=(
                    "More of this route is actually observed, so less of its cost "
                    "rests on uncertainty weights."
                ),
                value=float(winner.mapped_segment_count),
                comparison_value=float(runner_up.mapped_segment_count),
            )
        )

    return RouteRecommendation(
        selected_route_id=winner.candidate.route_id,
        reasons=tuple(reasons),
        alternatives_considered=len(routes) - 1,
        cost_margin_s=margin,
    )


def new_request_id() -> str:
    """A fresh request identifier."""
    return uuid.uuid4().hex[:12]


def measure_seconds(value: float, at: dt.datetime) -> FloatMeasurement:
    """Wrap a second count in a measurement, for callers building fixtures."""
    return FloatMeasurement(
        value=value,
        unit=Unit.SECOND,
        provenance=Provenance(
            source=ValueSource.DERIVED,
            producer=PRODUCER,
            producer_version=ALGORITHM_VERSION,
            observed_at=at,
            computed_at=at,
            input_refs=("caller",),
        ),
    )
