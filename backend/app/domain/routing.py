"""Route candidates and their cost structure.

Defined in Phase 1 and implemented by :mod:`app.routing` in Phase 7. The
schemas came first because the *shape* of the cost function is an
architectural commitment, not an implementation detail, and the wrong shape is
not fixable later without rewriting everything that consumes it. The optimizer
was written against this contract rather than the contract being retrofitted
to it.

Everything is in equivalent seconds
-----------------------------------
The specification's cost formula is

    travel time + congestion penalty + incident penalty + safety penalty
    + accessibility penalty

which is dimensionally incoherent if the penalties are unitless points: adding
points to seconds produces a quantity that is neither. Here every term is in
seconds, so the total is a **risk-adjusted arrival time** -- an interpretable
quantity a traffic engineer can challenge term by term, and the honest
formalisation of "minimum safe emergency arrival time" rather than "minimum
distance".

The breakdown is always retained alongside the total. A recommendation that can
only say "cost 412" cannot be explained; one that can say "310 s predicted
traversal, plus 60 s for a crowded crossing, plus 42 s for incident proximity"
can be audited, and gives Phase 12 real numbers to narrate instead of inventing
justifications.

Why not a Pareto front
----------------------
Multi-objective routing returning a non-dominated set is defensible for planning
tools, where a human can browse trade-offs. For an emergency dispatch decision
under time pressure it is the wrong output: somebody must still collapse the set
to one route, and doing that implicitly at the UI layer hides the trade-off
instead of recording it. Scalarising in equivalent seconds makes the trade-off
explicit, versioned (:attr:`RouteCostBreakdown.weights_version`) and reviewable.
Alternatives are still returned ranked, so a human can override with the full
breakdown in view.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.provenance import FloatMeasurement, Provenance
from app.domain.safety import SafetyConstraint


class TimeBasis(StrEnum):
    """Where a leg's traversal time came from.

    Reported rather than assumed, because these are not equally trustworthy and
    a consumer that cannot tell them apart will treat a derivation as a
    measurement.
    """

    MEASURED_EMERGENCY = "measured_emergency"
    """An emergency-specific estimate, produced by a model of its own against
    its own ground truth. Never the nominal time scaled by a constant."""

    MEASURED_NOMINAL = "measured_nominal"
    """A measured or modelled travel time for general traffic."""

    DERIVED_FROM_FREE_FLOW = "derived_from_free_flow"
    """Length divided by the segment's surveyed free-flow speed. A derivation
    from two measured quantities, not an assumption -- but it describes an empty
    road, so it is optimistic wherever there is traffic."""

    UNAVAILABLE = "unavailable"
    """No basis exists. The leg carries no time and its route has no ETA."""


RouteId = Annotated[str, Field(min_length=1, max_length=64)]


class RouteLeg(DomainModel):
    """One segment traversal within a route, plus the turn taken to enter it."""

    segment_id: Annotated[str, Field(min_length=1, max_length=64)]
    from_junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    to_junction_id: Annotated[str, Field(min_length=1, max_length=64)]
    entry_movement_id: Annotated[str, Field(max_length=64)] | None = None
    """The turn taken at ``from_junction_id`` to enter this segment. Present when
    the route was computed on a movement-expanded graph; ``None`` for a
    node-level route where turn feasibility was not checked."""

    length_m: FloatMeasurement
    traversal_time_s: FloatMeasurement | None = None
    """Predicted time for *this* response to traverse this leg. For an emergency
    route this is the emergency estimate, with its assumptions attached to the
    parent route -- never the nominal time scaled by a constant.

    ``None`` when the segment offers no basis for a time at all: no measured
    travel time, and no free-flow speed to derive one from. It is optional
    rather than defaulted because the alternatives are both worse. A zero would
    make the segment look instantaneous, and an invented speed would make the
    whole route's ETA a fabrication. A leg with no time basis makes its route's
    ETA unavailable, which is a fact the consumer must be able to see."""

    time_basis: TimeBasis = TimeBasis.UNAVAILABLE
    """How ``traversal_time_s`` was arrived at. Two routes whose times come from
    a measurement and from a length-over-speed derivation are not equally
    trustworthy, and collapsing that difference would hide it."""


class RouteCostBreakdown(DomainModel):
    """Additive cost decomposition, every term in equivalent seconds."""

    base_travel_time_s: Annotated[float, Field(ge=0.0)]
    congestion_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    incident_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    safety_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    accessibility_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    turn_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    """Cost of turns themselves -- a heavy vehicle turning across traffic is not
    free. Zero on a node-level route, which is itself a signal that turn
    feasibility was not modelled."""

    uncertainty_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    """Cost of routing on stale or absent evidence. Makes the well-observed
    route win ties against an unobserved one, rather than the optimiser
    preferring unmonitored roads precisely because nothing was seen on them."""

    distance_penalty_s: Annotated[float, Field(ge=0.0)] = 0.0
    """Distance priced as a stated policy preference, in equivalent seconds.

    Zero by default, because distance already enters the cost through travel
    time and pricing it twice would double-count. A non-zero weight is an
    operator saying "I will trade N seconds to save 100 m" -- a preference, not
    a claim about how fast anything moves. It is emphatically *not* a way to
    infer a time for a segment whose speed is unknown: that would be an invented
    speed wearing a different name."""

    weights_version: Annotated[str, Field(min_length=1, max_length=50)]
    """Pinned weight-set version. Two runs with different weights are not
    comparable, and without this pin that difference is invisible."""

    @property
    def total_s(self) -> float:
        """Risk-adjusted arrival time in seconds."""
        return (
            self.base_travel_time_s
            + self.congestion_penalty_s
            + self.incident_penalty_s
            + self.safety_penalty_s
            + self.accessibility_penalty_s
            + self.turn_penalty_s
            + self.uncertainty_penalty_s
            + self.distance_penalty_s
        )

    @property
    def penalty_share(self) -> float:
        """Fraction of total cost that is penalty rather than travel time.

        A high share means the ranking is dominated by risk weighting rather than
        by predicted time. That is sometimes correct and always worth surfacing:
        it tells a reviewer the recommendation rests on policy weights, not on
        measured traffic.
        """
        total = self.total_s
        return 0.0 if total <= 0 else (total - self.base_travel_time_s) / total


class RouteCandidate(DomainModel):
    """One admissible route from origin to destination.

    Admissible by construction: candidates are generated on the subgraph that
    remains after hard constraints prune it, so a candidate can never violate a
    hard constraint. ``soft_constraints`` records what was priced.
    """

    route_id: RouteId
    legs: tuple[RouteLeg, ...]
    cost: RouteCostBreakdown
    soft_constraints: tuple[SafetyConstraint, ...] = ()
    warnings: tuple[str, ...] = ()
    emergency_assumptions: tuple[str, ...] = ()
    """Conditions the ETA depends on, propagated from the per-leg emergency
    estimates so the narrative can state them."""

    computed_at: dt.datetime
    provenance: Provenance

    @model_validator(mode="after")
    def _validate(self) -> RouteCandidate:
        if not self.legs:
            msg = "a route must contain at least one leg"
            raise ValueError(msg)
        if self.computed_at.tzinfo is None:
            msg = "computed_at must be timezone-aware"
            raise ValueError(msg)
        for prev, nxt in zip(self.legs, self.legs[1:], strict=False):
            if prev.to_junction_id != nxt.from_junction_id:
                msg = (
                    f"route {self.route_id!r} is discontinuous: leg ending at "
                    f"{prev.to_junction_id!r} is followed by a leg starting at "
                    f"{nxt.from_junction_id!r}"
                )
                raise ValueError(msg)
        return self

    @property
    def origin_junction_id(self) -> str:
        """Junction the route starts from."""
        return self.legs[0].from_junction_id

    @property
    def destination_junction_id(self) -> str:
        """Junction the route ends at."""
        return self.legs[-1].to_junction_id

    @property
    def junction_sequence(self) -> tuple[str, ...]:
        """Ordered junctions traversed, for display and for corridor planning."""
        return (self.legs[0].from_junction_id, *(leg.to_junction_id for leg in self.legs))

    @property
    def total_length_m(self) -> float:
        """Total route length. Reported alongside, never optimised for."""
        return sum(leg.length_m.value for leg in self.legs)


class RouteComparison(DomainModel):
    """Ranked candidates and the selection, with the reason for the ranking.

    The rejected alternatives are retained deliberately. "Why not the short
    route?" is the first question anyone asks of an emergency recommendation,
    and it can only be answered if the alternatives and their costs survive.
    """

    candidates: tuple[RouteCandidate, ...]
    selected_route_id: RouteId
    ranking_criterion: Annotated[str, Field(min_length=1, max_length=200)] = (
        "minimum risk-adjusted arrival time (equivalent seconds)"
    )
    pruned_segment_ids: frozenset[str] = frozenset()
    """Segments removed by hard constraints before search. Retained so the
    narrative can say *why* an obvious route never appeared as a candidate."""

    pruning_constraints: tuple[SafetyConstraint, ...] = ()

    @model_validator(mode="after")
    def _selection_is_a_candidate(self) -> RouteComparison:
        ids = {c.route_id for c in self.candidates}
        if self.selected_route_id not in ids:
            msg = (
                f"selected route {self.selected_route_id!r} is not among the "
                f"candidates {sorted(ids)}"
            )
            raise ValueError(msg)
        return self

    @property
    def selected(self) -> RouteCandidate:
        """The chosen route."""
        return next(c for c in self.candidates if c.route_id == self.selected_route_id)
