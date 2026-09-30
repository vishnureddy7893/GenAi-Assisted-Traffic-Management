"""Safety constraints, in two tiers.

The two-tier model
------------------
**Hard constraints** answer "may this element be used at all?" They are
*admissibility filters*, applied to segments, junctions and movements **before**
route search, so that an inadmissible element is never relaxed and never appears
in any candidate. A blocked road, an active fire exclusion zone, a structural
closure, a vehicle that physically cannot fit.

**Soft constraints** answer "how much worse is this element?" They contribute a
penalty to route cost and surface as warnings. Pedestrian density, adverse
weather, degraded surface, moderate congestion.

Why the ordering matters
------------------------
Selecting a route and then testing it for safety is the obvious design and it is
wrong in two ways. It wastes work proportional to the number of candidates, and
worse, when the best route fails the check there is no principled fallback --
the search has already discarded the alternatives. Filtering first means the
search only ever explores the admissible subgraph, so whatever it returns is
admissible by construction.

The corollary must be handled explicitly: hard filtering can disconnect the
graph. "No admissible route exists" is then a real, reportable outcome
(:class:`~app.core.errors.NoAdmissibleRouteError`) -- never a trigger for
quietly relaxing a hard constraint to produce *some* answer.

Penalties are in seconds
------------------------
Every soft penalty is expressed in **equivalent seconds**, not in arbitrary
points. Adding unitless penalties to a travel time produces a number that is not
a time and cannot be explained to anyone: "cost 47.3" means nothing to an
operator. "Nominal 180 s, plus 60 s equivalent for a crowded crossing" is a
statement a traffic engineer can argue with -- which is the property we want,
because a cost function nobody can argue with is a cost function nobody can
validate.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.enums import EmergencyVehicleType, IncidentSeverity
from app.domain.provenance import Provenance

RuleId = Annotated[str, Field(min_length=1, max_length=80)]
ConstraintId = Annotated[str, Field(min_length=1, max_length=64)]


class ConstraintTier(StrEnum):
    """Whether a constraint prunes or penalises."""

    HARD = "hard"
    """Absolute. Removes the element from the search graph. Never priced,
    never traded off, never overridden by a lower cost elsewhere."""

    SOFT = "soft"
    """Degradable. Adds an equivalent-seconds penalty and raises a warning."""


class ConstraintScope(StrEnum):
    """Which kind of network element a constraint binds to."""

    SEGMENT = "segment"
    JUNCTION = "junction"
    MOVEMENT = "movement"
    """A specific turn. The finest scope, and the reason movements are modelled
    as first-class: banning one left turn should not close the whole junction."""

    AREA = "area"
    NETWORK = "network"


class SafetyRule(DomainModel):
    """A rule in the versioned safety catalog.

    A rule is policy: it defines *what* is unsafe and *at which tier*. It carries
    no target -- binding to concrete network elements happens in
    :class:`SafetyConstraint`.

    Tier lives on the rule, not on the instance, so that "is a crowded pedestrian
    crossing a hard or a soft constraint?" is answered once, in a versioned,
    reviewable artifact, rather than case by case at runtime. Escalating a
    condition from soft to hard is a catalog change with a version bump and an
    audit trail -- which is exactly what a change of that consequence deserves.
    """

    rule_id: RuleId
    name: Annotated[str, Field(min_length=1, max_length=200)]
    tier: ConstraintTier
    scope: ConstraintScope
    description: Annotated[str, Field(min_length=1, max_length=1000)]
    rationale: Annotated[str, Field(min_length=1, max_length=1000)]
    """Why this rule exists. Required: an unexplained safety rule cannot be
    reviewed, and rules that cannot be reviewed accumulate until the system is
    refusing routes for reasons nobody remembers."""

    applies_to_vehicle_types: frozenset[EmergencyVehicleType] | None = None
    """``None`` means all traffic. A set scopes the rule to specific responders --
    a height restriction excludes a ladder truck but not an ambulance."""

    default_penalty_equivalent_s: float | None = None
    """Calibrated penalty for soft rules. Deliberately left unset in the Phase 1
    catalog: a penalty weight is an empirical tuning parameter, and inventing one
    now would put a fabricated number at the heart of route selection. These are
    calibrated against simulation in Phase 11 and versioned with the catalog."""

    @model_validator(mode="after")
    def _tier_obligations(self) -> SafetyRule:
        if self.tier is ConstraintTier.HARD and self.default_penalty_equivalent_s is not None:
            msg = (
                f"rule {self.rule_id!r} is HARD but defines a penalty; a hard "
                "constraint prunes and is never priced. If it is meant to be "
                "tradeable, it is a SOFT rule."
            )
            raise ValueError(msg)
        return self


class SafetyConstraint(DomainModel):
    """A rule bound to concrete network elements over a time window.

    Instances are produced by the safety engine from network state, incidents,
    environment and the responding vehicle -- and are attached to the decision
    record, so a recommendation can always be explained in terms of the specific
    constraints that shaped it.
    """

    constraint_id: ConstraintId
    rule_id: RuleId
    tier: ConstraintTier
    scope: ConstraintScope

    target_segment_ids: frozenset[str] = frozenset()
    target_junction_ids: frozenset[str] = frozenset()
    target_movement_ids: frozenset[str] = frozenset()

    active_from: dt.datetime
    active_until: dt.datetime | None = None
    """``None`` means open-ended: active until superseded or cleared."""

    severity: IncidentSeverity | None = None
    penalty_equivalent_s: Annotated[float, Field(ge=0.0)] | None = None
    """Soft only, and mandatory for soft. In equivalent seconds, so it is
    commensurate with travel time and can be explained additively."""

    message: Annotated[str, Field(min_length=1, max_length=500)]
    """Operator-facing text. For a soft constraint this becomes a warning on the
    recommendation; for a hard one it explains why a route was excluded."""

    applies_to_vehicle_types: frozenset[EmergencyVehicleType] | None = None
    provenance: Provenance
    """What evidence produced this constraint. A constraint asserted with no
    evidence is indistinguishable from an invented restriction."""

    @model_validator(mode="after")
    def _validate(self) -> SafetyConstraint:
        if self.active_from.tzinfo is None:
            msg = "active_from must be timezone-aware"
            raise ValueError(msg)
        if self.active_until is not None:
            if self.active_until.tzinfo is None:
                msg = "active_until must be timezone-aware"
                raise ValueError(msg)
            if self.active_until < self.active_from:
                msg = "active_until cannot precede active_from"
                raise ValueError(msg)

        has_targets = bool(
            self.target_segment_ids or self.target_junction_ids or self.target_movement_ids
        )
        if not has_targets and self.scope is not ConstraintScope.NETWORK:
            msg = (
                f"constraint {self.constraint_id!r} has scope {self.scope.value} but "
                "no targets; an unbound constraint silently affects nothing"
            )
            raise ValueError(msg)

        if self.tier is ConstraintTier.HARD and self.penalty_equivalent_s is not None:
            msg = (
                f"constraint {self.constraint_id!r} is HARD and must not carry a "
                "penalty: hard constraints prune, they are not traded off against "
                "travel time"
            )
            raise ValueError(msg)

        if self.tier is ConstraintTier.SOFT and self.penalty_equivalent_s is None:
            msg = (
                f"constraint {self.constraint_id!r} is SOFT and must carry an explicit "
                "penalty_equivalent_s; a soft constraint with no cost has no effect on "
                "ranking and would be a silently ignored safety signal"
            )
            raise ValueError(msg)

        return self

    def is_active_at(self, when: dt.datetime) -> bool:
        """Whether the constraint applies at instant ``when``."""
        if when.tzinfo is None:
            msg = "when must be timezone-aware"
            raise ValueError(msg)
        if when < self.active_from:
            return False
        return self.active_until is None or when <= self.active_until

    def applies_to(self, vehicle_type: EmergencyVehicleType | None) -> bool:
        """Whether the constraint binds for a given responder type.

        ``None`` means general traffic, for which every constraint applies:
        a vehicle-scoped restriction is a narrowing, never a widening.
        """
        if self.applies_to_vehicle_types is None:
            return True
        if vehicle_type is None:
            return True
        return vehicle_type in self.applies_to_vehicle_types


class SafetyRuleCatalog(DomainModel):
    """The versioned set of rules in force.

    Pinned by version into every decision record. Without that pin, a change in
    routing behaviour six weeks from now cannot be attributed to a rule change
    versus a model change versus a data change.
    """

    version: Annotated[str, Field(min_length=1, max_length=50)]
    rules: tuple[SafetyRule, ...]

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> SafetyRuleCatalog:
        ids = [r.rule_id for r in self.rules]
        if len(set(ids)) != len(ids):
            msg = "duplicate rule_id in safety catalog"
            raise ValueError(msg)
        return self

    def rule(self, rule_id: str) -> SafetyRule:
        """Return a rule by id, raising ``KeyError`` if absent."""
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        raise KeyError(rule_id)

    def rules_of_tier(self, tier: ConstraintTier) -> tuple[SafetyRule, ...]:
        """All rules at one tier."""
        return tuple(r for r in self.rules if r.tier is tier)

    def validate_constraint(self, constraint: SafetyConstraint) -> None:
        """Assert that a constraint instance is consistent with its rule.

        Guards against the most dangerous drift in a two-tier safety system: a
        constraint instantiated at SOFT from a rule the catalog defines as HARD.
        That would turn an absolute exclusion into a purchasable penalty, and a
        sufficiently congested alternative would then "outbid" a fire exclusion
        zone. Raises ``KeyError`` for an unknown rule and ``ValueError`` on a
        tier or scope mismatch.
        """
        rule = self.rule(constraint.rule_id)
        if rule.tier is not constraint.tier:
            msg = (
                f"constraint {constraint.constraint_id!r} declares tier "
                f"{constraint.tier.value} but catalog rule {rule.rule_id!r} (catalog "
                f"version {self.version}) defines tier {rule.tier.value}; a constraint "
                "may not re-tier its rule"
            )
            raise ValueError(msg)
        if rule.scope is not constraint.scope:
            msg = (
                f"constraint {constraint.constraint_id!r} declares scope "
                f"{constraint.scope.value} but rule {rule.rule_id!r} defines scope "
                f"{rule.scope.value}"
            )
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Phase 1 seed catalog
# ---------------------------------------------------------------------------
# Rule *definitions* only: identity, tier, scope and rationale. No penalty
# weights -- those are empirical parameters calibrated against simulation in
# Phase 11, and inventing them here would place fabricated numbers at the centre
# of route selection. Every soft constraint instance must therefore supply an
# explicit penalty, which keeps the gap visible instead of papering over it.

PHASE1_SAFETY_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        rule_id="hard.segment_blocked",
        name="Segment blocked",
        tier=ConstraintTier.HARD,
        scope=ConstraintScope.SEGMENT,
        description="The segment is impassable due to a recorded blockage or closure.",
        rationale=(
            "An impassable segment cannot be traversed at any cost. Pricing it "
            "would let a sufficiently attractive route send a responder into a "
            "dead end."
        ),
    ),
    SafetyRule(
        rule_id="hard.fire_exclusion_zone",
        name="Active fire exclusion zone",
        tier=ConstraintTier.HARD,
        scope=ConstraintScope.SEGMENT,
        description="The segment lies within the exclusion zone of an active fire.",
        rationale=(
            "Proximity to an active fire endangers the crew and may become "
            "impassable without warning. Exclusion is absolute, not tradeable."
        ),
    ),
    SafetyRule(
        rule_id="hard.structural_closure",
        name="Structural closure",
        tier=ConstraintTier.HARD,
        scope=ConstraintScope.SEGMENT,
        description="Bridge, tunnel or carriageway closed for structural reasons.",
        rationale="Structural failure risk is not a delay; it is a hazard to life.",
    ),
    SafetyRule(
        rule_id="hard.vehicle_physically_excluded",
        name="Vehicle physically excluded",
        tier=ConstraintTier.HARD,
        scope=ConstraintScope.SEGMENT,
        description=(
            "Segment height, width or weight restriction excludes the responding "
            "vehicle type."
        ),
        rationale=(
            "A physical restriction is independent of traffic conditions. It is "
            "vehicle-specific, so it must be evaluated per response rather than "
            "baked into the map."
        ),
    ),
    SafetyRule(
        rule_id="hard.movement_prohibited",
        name="Turn movement prohibited",
        tier=ConstraintTier.HARD,
        scope=ConstraintScope.MOVEMENT,
        description="The turn is banned or physically impossible at this junction.",
        rationale=(
            "Routing through a banned turn produces a route that cannot be "
            "driven. Scoped to the movement so the junction stays usable."
        ),
    ),
    SafetyRule(
        rule_id="soft.pedestrian_density",
        name="High pedestrian density at crossing",
        tier=ConstraintTier.SOFT,
        scope=ConstraintScope.JUNCTION,
        description="Pedestrian presence at a crossing on the recommended path.",
        rationale=(
            "Pedestrians raise risk and slow safe passage but do not make a route "
            "impossible. Kept soft so a marginally busier crossing does not "
            "eliminate an otherwise far faster route -- while still being priced, "
            "and surfaced as a precaution rather than silently ignored."
        ),
    ),
    SafetyRule(
        rule_id="soft.adverse_weather",
        name="Adverse weather",
        tier=ConstraintTier.SOFT,
        scope=ConstraintScope.SEGMENT,
        description="Rain, fog, snow or ice degrading traction and visibility.",
        rationale=(
            "Weather degrades safe speed and stopping distance continuously "
            "rather than absolutely, so it belongs in the cost function."
        ),
    ),
    SafetyRule(
        rule_id="soft.poor_road_surface",
        name="Poor road surface",
        tier=ConstraintTier.SOFT,
        scope=ConstraintScope.SEGMENT,
        description="Potholed, unpaved or partially flooded carriageway.",
        rationale=(
            "Degrades achievable speed and ride safety, particularly for a "
            "patient under transport, without preventing passage."
        ),
    ),
    SafetyRule(
        rule_id="soft.severe_congestion",
        name="Severe congestion",
        tier=ConstraintTier.SOFT,
        scope=ConstraintScope.SEGMENT,
        description="Congestion severe enough to impede emergency progress.",
        rationale=(
            "Congestion is already reflected in travel time; the additional "
            "penalty captures the variance and blocking risk that a mean travel "
            "time understates. Never hard: a jammed road is still a road."
        ),
    ),
    SafetyRule(
        rule_id="soft.stale_evidence",
        name="Stale or missing evidence",
        tier=ConstraintTier.SOFT,
        scope=ConstraintScope.SEGMENT,
        description="No fresh observation supports this element's state.",
        rationale=(
            "Routing over an unobserved element is a gamble on unknown conditions. "
            "Priced rather than pruned, because pruning every unobserved element "
            "would disconnect a sparsely instrumented network and deny a route "
            "outright -- a worse failure than an uncertain one."
        ),
    ),
    SafetyRule(
        rule_id="soft.incident_proximity",
        name="Proximity to an active incident",
        tier=ConstraintTier.SOFT,
        scope=ConstraintScope.SEGMENT,
        description="Segment is near, but not inside, an active incident's impact area.",
        rationale=(
            "Secondary-incident risk and rubbernecking delay are real but "
            "probabilistic. Distinguished from the exclusion zone, which is hard."
        ),
    ),
)

PHASE1_SAFETY_CATALOG = SafetyRuleCatalog(version="0.1.0", rules=PHASE1_SAFETY_RULES)
