"""Route cost weights, resolved from configuration into an inspectable value.

Every weight is in **equivalent seconds**, so a total cost is a risk-adjusted
arrival time rather than a score. "Cost 412" cannot be explained to an operator;
"310 s predicted traversal, plus 60 s for a crowded crossing, plus 42 s for
incident proximity" can be argued with, which is the property that makes the
cost function reviewable at all.

The weight set travels with the result
--------------------------------------
:class:`RouteWeights` is embedded in every optimization result, not merely read
at ranking time. Two runs under different weights are not comparable, and
without the weights recorded beside the ranking that difference is invisible.

Emergency weights default to the normal ones
--------------------------------------------
Not because the two are the same problem, but because inventing the difference
would be a fabrication of exactly the kind this system refuses elsewhere. What
separates emergency routing here is *structural*: vehicle-type physical
exclusions become hard constraints, and an emergency-specific travel time is
preferred where one genuinely exists. Any numeric divergence is an operator
decision recorded in configuration, not a constant chosen in this file.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from app.core.config import Settings
from app.domain.base import DomainModel
from app.domain.enums import CongestionLevel


class RoutingMode(StrEnum):
    """Which profile a request is routed under."""

    NORMAL = "normal"
    """General traffic. No vehicle-type exclusions, nominal travel time."""

    EMERGENCY = "emergency"
    """A declared response. Applies the responder's physical exclusions as hard
    constraints and prefers an emergency-specific travel time where one exists."""


class RouteWeights(DomainModel):
    """The priced policy in force for one optimization."""

    mode: RoutingMode
    weights_version: Annotated[str, Field(min_length=1, max_length=50)]

    congestion_s: dict[CongestionLevel, float]
    """Penalty per interpreted congestion band. ``UNKNOWN`` is deliberately
    absent: an unclassified segment is priced by the uncertainty weights, never
    as free-flowing, because reading "we did not see" as "it is clear" is what
    makes an optimizer prefer the roads nobody is watching."""

    uncertainty_unmapped_s: Annotated[float, Field(ge=0)]
    uncertainty_unclassified_s: Annotated[float, Field(ge=0)]
    uncertainty_stale_s: Annotated[float, Field(ge=0)]

    incident_degraded_s: Annotated[float, Field(ge=0)]
    turn_penalty_s: Annotated[float, Field(ge=0)]
    u_turn_penalty_s: Annotated[float, Field(ge=0)]
    distance_penalty_s_per_100m: Annotated[float, Field(ge=0)]

    max_candidates: Annotated[int, Field(ge=1)]
    max_settled_nodes: Annotated[int, Field(ge=1)]

    def congestion_penalty(self, level: CongestionLevel) -> float:
        """Penalty for a band, or zero for one that is not priced here.

        ``UNKNOWN`` returns zero *and* is expected to draw an uncertainty
        penalty from the caller. Returning zero without that companion charge
        would be the bug this whole design is built to avoid.
        """
        return self.congestion_s.get(level, 0.0)


def weights_for(settings: Settings, mode: RoutingMode) -> RouteWeights:
    """Resolve the weight set for a routing mode from configuration."""
    if mode is RoutingMode.EMERGENCY:
        return RouteWeights(
            mode=mode,
            weights_version=settings.route_weights_version,
            congestion_s={
                CongestionLevel.FREE_FLOW: settings.route_emergency_congestion_free_flow_s,
                CongestionLevel.LIGHT: settings.route_emergency_congestion_light_s,
                CongestionLevel.MODERATE: settings.route_emergency_congestion_moderate_s,
                CongestionLevel.HEAVY: settings.route_emergency_congestion_heavy_s,
                CongestionLevel.SEVERE: settings.route_emergency_congestion_severe_s,
                CongestionLevel.GRIDLOCK: settings.route_emergency_congestion_gridlock_s,
            },
            uncertainty_unmapped_s=settings.route_emergency_uncertainty_unmapped_s,
            uncertainty_unclassified_s=settings.route_emergency_uncertainty_unclassified_s,
            uncertainty_stale_s=settings.route_emergency_uncertainty_stale_s,
            incident_degraded_s=settings.route_emergency_incident_degraded_s,
            turn_penalty_s=settings.route_emergency_turn_penalty_s,
            u_turn_penalty_s=settings.route_emergency_u_turn_penalty_s,
            distance_penalty_s_per_100m=(
                settings.route_emergency_distance_penalty_s_per_100m
            ),
            max_candidates=settings.route_max_candidates,
            max_settled_nodes=settings.route_max_settled_nodes,
        )

    return RouteWeights(
        mode=mode,
        weights_version=settings.route_weights_version,
        congestion_s={
            CongestionLevel.FREE_FLOW: settings.route_congestion_free_flow_s,
            CongestionLevel.LIGHT: settings.route_congestion_light_s,
            CongestionLevel.MODERATE: settings.route_congestion_moderate_s,
            CongestionLevel.HEAVY: settings.route_congestion_heavy_s,
            CongestionLevel.SEVERE: settings.route_congestion_severe_s,
            CongestionLevel.GRIDLOCK: settings.route_congestion_gridlock_s,
        },
        uncertainty_unmapped_s=settings.route_uncertainty_unmapped_s,
        uncertainty_unclassified_s=settings.route_uncertainty_unclassified_s,
        uncertainty_stale_s=settings.route_uncertainty_stale_s,
        incident_degraded_s=settings.route_incident_degraded_s,
        turn_penalty_s=settings.route_turn_penalty_s,
        u_turn_penalty_s=settings.route_u_turn_penalty_s,
        distance_penalty_s_per_100m=settings.route_distance_penalty_s_per_100m,
        max_candidates=settings.route_max_candidates,
        max_settled_nodes=settings.route_max_settled_nodes,
    )
