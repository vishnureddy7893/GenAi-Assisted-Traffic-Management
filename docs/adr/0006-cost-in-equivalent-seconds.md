# ADR 0006 — Route cost is expressed in equivalent seconds

**Status:** accepted (Phase 1)

## Context

The specified cost is travel time plus congestion, incident, safety and
accessibility penalties.

## Decision

Express every term in **equivalent seconds**. Retain the full
`RouteCostBreakdown` alongside the total, and pin the weight-set version into
every breakdown.

## Rationale

Adding unitless penalties to a travel time yields a quantity that is neither a
time nor a score. "Cost 412" cannot be explained to an operator. "310 s
predicted traversal, plus 60 s for a crowded crossing, plus 42 s for incident
proximity" can be argued with term by term — and a cost function nobody can argue
with is a cost function nobody can validate.

The total is then a *risk-adjusted arrival time*, which is the honest
formalisation of "minimum safe emergency arrival time".

A Pareto front was considered and rejected. Returning a non-dominated set is
defensible for a planning tool where a human browses trade-offs, but under
dispatch time pressure somebody must still collapse it to one route. Doing that
implicitly at the UI layer hides the trade-off instead of recording it.

## Consequences

Weights are empirical parameters and are calibrated against simulation in Phase
11. The Phase 1 catalog therefore ships rule definitions with **no** penalty
values — inventing them now would place fabricated numbers at the centre of route
selection. Every soft constraint instance must supply an explicit penalty, which
keeps the gap visible rather than papering over it.

`penalty_share` surfaces when a ranking rests mostly on policy weights rather
than on measured traffic.
