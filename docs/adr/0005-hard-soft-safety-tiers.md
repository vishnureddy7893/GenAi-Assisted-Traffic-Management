# ADR 0005 — Hard constraints prune before search; soft constraints price

**Status:** accepted (Phase 1)

## Context

Safety conditions range from absolute (an active fire exclusion zone) to
degradable (pedestrian density, adverse weather).

## Decision

Two tiers. **Hard** constraints are admissibility filters applied to segments,
junctions and movements *before* route search. **Soft** constraints add
penalties in equivalent seconds and raise warnings. Tier is a property of the
versioned catalog **rule**, not of the constraint instance.

Validators enforce three invariants:

- a hard constraint must not carry a penalty;
- a soft constraint must carry an explicit penalty;
- a constraint may not re-tier its catalog rule.

## Rationale

Select-then-validate is the obvious design and it is wrong twice. It wastes work
proportional to the number of candidates, and when the best route fails the
check there is no principled fallback, because search has already discarded the
alternatives. Filtering first means whatever search returns is admissible by
construction.

Pricing a hard constraint would let a sufficiently congested alternative outbid
a fire exclusion zone. A soft constraint with no penalty is a silently ignored
safety signal.

Putting tier on the rule means "is a crowded pedestrian crossing hard or soft?"
is answered once, in a reviewable artifact, and escalating a condition from soft
to hard is a catalog version bump with an audit trail — which is what a change of
that consequence deserves.

## Consequences

Hard filtering can disconnect the graph. `NoAdmissibleRouteError` is therefore a
first-class outcome, never a trigger for quietly relaxing a hard constraint to
produce *some* answer.
