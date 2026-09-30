"""Route optimization request and result schemas.

These wrap, rather than replace, the Phase 1 routing contract. `RouteCandidate`,
`RouteCostBreakdown` and `RouteComparison` in :mod:`app.domain.routing` are what
the Decision Engine will consume, and their shape was an architectural
commitment made before any of this existed. What Phase 7 adds is everything a
*request* and an *explanation* need that a candidate alone does not carry: what
was asked, what was rejected and why, what evidence each segment rested on, and
which weight set produced the ranking.

An answer of "no route" is a result, not an error
-------------------------------------------------
:class:`RouteOptimizationResult` carries a status. When every path is cut by a
hard constraint the status says so and the rejections say which segments did it.
That is a complete, useful answer -- the operator learns the network is severed
and where -- and it is emphatically not an occasion to relax a constraint until
something comes back.

Nothing here is prose
---------------------
Reasons are codes with structured detail, not sentences to be shown verbatim as
an explanation. Phase 12 turns them into narrative against a fact table; this
layer supplies the facts and stops.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import CongestionLevel, EmergencyVehicleType, FreshnessStatus
from app.domain.provenance import Provenance
from app.domain.routing import RouteCandidate, RouteComparison, TimeBasis
from app.routing.weights import RouteWeights, RoutingMode


class OptimizationStatus(StrEnum):
    """Whether the search produced anything usable."""

    ROUTES_FOUND = "routes_found"
    NO_ADMISSIBLE_ROUTE = "no_admissible_route"
    """Every path was cut by a hard constraint, or none existed. A first-class
    outcome: the correct response is to report it with the causes, never to fall
    back to a route that violates a constraint."""


class EvidenceStatus(StrEnum):
    """How much a segment's traffic evidence could be trusted, if any existed."""

    MAPPED_CLASSIFIED = "mapped_classified"
    """An observation area is bound to this segment and produced a congestion
    band from fresh evidence. The only status whose congestion is priced."""

    MAPPED_STALE = "mapped_stale"
    """Bound and classified, but the evidence is past the staleness budget. The
    band is still used -- withholding a route because one camera lagged would be
    its own failure -- and an uncertainty penalty is added alongside."""

    MAPPED_UNCLASSIFIED = "mapped_unclassified"
    """Bound, but Phase 5 refused to classify: poor quality, tracking failure or
    no usable signal. Priced as uncertainty, never as free-flowing."""

    UNMAPPED = "unmapped"
    """No observation area is bound to this segment. Traffic may well have been
    observed nearby; without a binding it cannot be attributed here."""


class RejectionReason(StrEnum):
    """Why a network element or the whole request was refused."""

    SEGMENT_BLOCKED = "segment_blocked"
    EXCLUSION_ZONE = "exclusion_zone"
    VEHICLE_PHYSICALLY_EXCLUDED = "vehicle_physically_excluded"
    MOVEMENT_PROHIBITED = "movement_prohibited"
    ORIGIN_NOT_ON_NETWORK = "origin_not_on_network"
    DESTINATION_NOT_ON_NETWORK = "destination_not_on_network"
    NETWORK_DISCONNECTED = "network_disconnected"
    """The admissible subgraph has no path at all. Distinct from a pruned one:
    it means the map itself does not connect these points."""

    DISCONNECTED_BY_PRUNING = "disconnected_by_pruning"
    """A path exists on the raw graph but every one crosses a pruned element.
    The most important rejection to distinguish, because it is the one an
    operator can act on by clearing an incident."""

    SEARCH_LIMIT_REACHED = "search_limit_reached"


class ReasonCode(StrEnum):
    """Why the recommended route won."""

    ONLY_ADMISSIBLE_ROUTE = "only_admissible_route"
    LOWER_TOTAL_COST = "lower_total_cost"
    LOWER_TRAVEL_TIME = "lower_travel_time"
    LOWER_CONGESTION_COST = "lower_congestion_cost"
    LOWER_INCIDENT_COST = "lower_incident_cost"
    SHORTER_DISTANCE = "shorter_distance"
    BETTER_EVIDENCE_COVERAGE = "better_evidence_coverage"
    NO_HARD_BLOCKED_SEGMENTS = "no_hard_blocked_segments"
    """True of every candidate by construction -- candidates are generated on the
    pruned subgraph. Stated anyway, because it is the first thing an operator
    checks and its absence from the list would read as an omission."""

    TRAVEL_TIME_UNAVAILABLE_RANKED_BY_DISTANCE = (
        "travel_time_unavailable_ranked_by_distance"
    )


class RouteRequest(DomainModel):
    """What was asked for."""

    request_id: Annotated[str, Field(min_length=1, max_length=64)]
    mode: RoutingMode
    origin_junction_id: Annotated[str, Field(min_length=1, max_length=64)]

    destination_junction_id: Annotated[str, Field(max_length=64)] | None = None
    destination_facility_id: Annotated[str, Field(max_length=64)] | None = None
    """A facility is resolved to its access junction. Kept in the request so the
    result can say "to Example General Hospital" rather than "to J3"."""

    emergency_id: Annotated[str, Field(max_length=64)] | None = None
    vehicle_type: EmergencyVehicleType | None = None
    max_candidates: Annotated[int, Field(ge=1, le=20)] | None = None
    requested_at: dt.datetime

    @model_validator(mode="after")
    def _exactly_one_destination(self) -> RouteRequest:
        provided = [
            name
            for name, value in (
                ("destination_junction_id", self.destination_junction_id),
                ("destination_facility_id", self.destination_facility_id),
            )
            if value is not None
        ]
        if len(provided) != 1:
            msg = (
                "a route request must name exactly one of destination_junction_id "
                f"or destination_facility_id; got {provided or 'none'}"
            )
            raise ValueError(msg)
        if self.requested_at.tzinfo is None:
            msg = "requested_at must be timezone-aware"
            raise ValueError(msg)
        return self


class SegmentEvidence(DomainModel):
    """What was known about one segment, and where that knowledge came from.

    Retained for every segment a candidate traverses, so a cost can be traced
    back to an observation, a media file and a frame range -- or shown to rest
    on nothing at all.
    """

    segment_id: Annotated[str, Field(min_length=1, max_length=64)]
    status: EvidenceStatus
    congestion_level: CongestionLevel = CongestionLevel.UNKNOWN
    congestion_index: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    observation_area_id: Annotated[str, Field(max_length=120)] | None = None
    traffic_state_id: Annotated[str, Field(max_length=64)] | None = None
    freshness: FreshnessStatus | None = None
    evidence_age_s: Annotated[float, Field(ge=0.0)] | None = None

    media_id: Annotated[str, Field(max_length=64)] | None = None
    """The asset the evidence ultimately came from. The end of the chain that
    starts at a route cost and ends at a file on disk."""

    frame_range: tuple[int, int] | None = None
    time_range: tuple[dt.datetime, dt.datetime] | None = None
    detector_model: Annotated[str, Field(max_length=120)] | None = None

    congestion_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    uncertainty_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    incident_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    reason: Annotated[str, Field(max_length=400)] | None = None
    """Why the evidence is in the state it is. Required reading when the status
    is anything other than ``MAPPED_CLASSIFIED``."""


class RouteRejection(DomainModel):
    """One reason a route, an element or the whole request was refused."""

    reason: RejectionReason
    message: Annotated[str, Field(min_length=1, max_length=500)]
    segment_ids: tuple[str, ...] = ()
    junction_ids: tuple[str, ...] = ()
    movement_ids: tuple[str, ...] = ()
    constraint_ids: tuple[str, ...] = ()
    incident_ids: tuple[str, ...] = ()


class RouteReason(DomainModel):
    """One structured reason the recommended route was selected."""

    code: ReasonCode
    detail: Annotated[str, Field(min_length=1, max_length=400)]
    value: float | None = None
    comparison_value: float | None = None
    """The runner-up's figure, where the reason is comparative. "Lower cost" is
    unfalsifiable on its own; "412 s against the runner-up's 530 s" can be
    checked against the breakdowns."""


class RankedRoute(DomainModel):
    """One candidate with the derived figures a consumer needs.

    Everything here is computed from the candidate rather than stored beside it,
    so the two cannot disagree. It exists because a `RouteCandidate` deliberately
    carries no ETA field: whether a route *has* an ETA depends on whether every
    one of its legs had a time basis, which is a property of the whole route.
    """

    rank: Annotated[int, Field(ge=1)]
    candidate: RouteCandidate

    estimated_travel_time_s: Annotated[float, Field(ge=0.0)] | None = None
    """Sum of leg traversal times. ``None`` whenever any leg had no time basis:
    a partial sum presented as an ETA would understate arrival by exactly the
    legs it silently omitted."""

    travel_time_available: bool = False
    time_bases: tuple[TimeBasis, ...] = ()
    """The basis of every leg, in order. A route timed entirely from free-flow
    derivation is a weaker claim than one timed from measurements, and this is
    where a reader sees which they have."""

    unpriced_segment_ids: tuple[str, ...] = ()
    total_distance_m: Annotated[float, Field(ge=0.0)]
    junction_sequence: tuple[str, ...]
    segment_sequence: tuple[str, ...]

    evidence: tuple[SegmentEvidence, ...] = ()
    mapped_segment_count: Annotated[int, Field(ge=0)] = 0
    """How many of this route's segments had traffic evidence bound to them.
    The honest measure of how much of this ranking rests on observation."""

    @model_validator(mode="after")
    def _eta_matches_availability(self) -> RankedRoute:
        if self.travel_time_available and self.estimated_travel_time_s is None:
            msg = "travel_time_available is true but no estimated_travel_time_s is given"
            raise ValueError(msg)
        if not self.travel_time_available and self.estimated_travel_time_s is not None:
            msg = (
                "estimated_travel_time_s is present while travel_time_available is "
                "false; a partial sum must not be presented as an ETA"
            )
            raise ValueError(msg)
        return self


class RouteRecommendation(DomainModel):
    """Which route won, and the structured reasons why."""

    selected_route_id: Annotated[str, Field(min_length=1, max_length=64)]
    reasons: tuple[RouteReason, ...] = ()
    alternatives_considered: Annotated[int, Field(ge=0)] = 0
    cost_margin_s: float | None = None
    """How much cheaper the winner is than the runner-up, in equivalent seconds.
    ``None`` when there was no runner-up. A small margin is worth surfacing: it
    means the ranking could flip on a weight change."""


class RouteOptimizationResult(DomainModel):
    """The complete, structured answer to one route request."""

    request: RouteRequest
    status: OptimizationStatus

    routes: tuple[RankedRoute, ...] = ()
    comparison: RouteComparison | None = None
    """The Phase 1 contract shape, for the Decision Engine. Built from the same
    candidates as :attr:`routes`, so the two cannot diverge."""

    recommendation: RouteRecommendation | None = None
    rejections: tuple[RouteRejection, ...] = ()

    pruned_segment_ids: tuple[str, ...] = ()
    unmapped_observation_area_ids: tuple[str, ...] = ()
    """Areas whose traffic was interpreted but bound to nothing. Named so an
    operator can see that evidence exists and was deliberately not used."""

    warnings: tuple[str, ...] = ()

    weights: RouteWeights
    algorithm: Annotated[str, Field(min_length=1, max_length=80)]
    algorithm_version: Annotated[str, Field(min_length=1, max_length=40)]

    network_id: Annotated[str, Field(min_length=1, max_length=64)]
    network_version: Annotated[str, Field(min_length=1, max_length=50)]
    mapping_id: Annotated[str, Field(max_length=64)] | None = None
    mapping_version: Annotated[str, Field(max_length=50)] | None = None
    traffic_state_id: Annotated[str, Field(max_length=64)] | None = None

    computed_at: dt.datetime
    provenance: Provenance

    @model_validator(mode="after")
    def _status_matches_content(self) -> RouteOptimizationResult:
        if self.computed_at.tzinfo is None:
            msg = "computed_at must be timezone-aware"
            raise ValueError(msg)
        if self.status is OptimizationStatus.ROUTES_FOUND:
            if not self.routes:
                msg = "status is routes_found but no route is present"
                raise ValueError(msg)
            if self.recommendation is None:
                msg = "status is routes_found but nothing was recommended"
                raise ValueError(msg)
        else:
            if self.routes:
                msg = (
                    "status is no_admissible_route but routes are present; a refusal "
                    "that still carries a route is the failure this status exists to "
                    "prevent"
                )
                raise ValueError(msg)
            if not self.rejections:
                msg = (
                    "no_admissible_route must state why: an unexplained refusal is "
                    "indistinguishable from a crash"
                )
                raise ValueError(msg)
        return self

    @property
    def selected(self) -> RankedRoute | None:
        """The recommended route, if there is one."""
        if self.recommendation is None:
            return None
        return next(
            (
                route
                for route in self.routes
                if route.candidate.route_id == self.recommendation.selected_route_id
            ),
            None,
        )
