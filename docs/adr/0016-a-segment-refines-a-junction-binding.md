# ADR 0016 — A segment refines a junction binding, and still prices nothing

**Status:** accepted (Phase 8+, step 3 of real-world routing)

## Context

ADR-0015 established a human-confirmed site→junction binding and noted that it
prices nothing: `ObservationMapping.bindings_by_segment()` groups only bindings
that name a *segment*, so a junction binding cannot reach the optimizer's
costing. Knowing which intersection a camera watches is not knowing which
directed road its queue belongs to.

This step closes that gap in the record — an operator can now say which directed
segment the camera monitors — without closing it in the optimizer.

The geometry is weaker here than at the junction step, in a specific way. At a
junction, the two directions of every road lie on the same line. A camera's
bearing can therefore narrow the choice to a *line* and can never narrow it to a
*direction of travel*: a camera filming oncoming traffic and one filming the
backs of departing vehicles stand in the same place and point the same way.

## Decision

`GET /api/sites/{id}/segments` lists every directed segment incident to the
already-confirmed junction — both the incoming arms (which carry approaches) and
the outgoing ones. Where the site records `bearing_degrees` and the junctions
carry positions, each candidate also carries its compass bearing from the
junction and the angle to the camera's facing.

`POST /api/sites/{id}/binding/segment` records the operator's choice. It
requires `confirm: true`, `junction_id` **and** `segment_id`, and validates in
this order: the site exists, a junction binding exists, the named junction is the
one on record, that binding was confirmed against the loaded network *version*,
the segment exists, and the segment actually touches the junction.

The result is written back onto the **same** `ConfirmedSiteBinding`, adding
`segment_id`, `segment_direction` and a second `segment_confirmation`. The
junction confirmation and whoever made it are untouched.

`BindingConfirmationMethod` gains `aligned_with_bearing`. Confirming a segment
the camera's bearing points along records that value; confirming one it points
away from records `chose_different` and therefore requires written notes, via the
validator that already existed.

## Rationale

**Why the listing is ordered by segment id, not by bearing.** Ordering by
bearing puts one segment at the top of the operator's list, which is selecting by
another name — and it would be most confidently wrong precisely where it matters,
since the two directions of one road have near-identical offsets. Pairs within
`COLLINEAR_TOLERANCE_DEG` of each other are named in a warning instead.

**Why `aligned_with_bearing` rather than reusing `accepted_nearest`.** There is
no "nearest segment"; the word would have been false. The new value records that
the geometry *agreed*, not that it decided — and it is reachable for either
direction of the same road, which is exactly why it cannot be read as a decision.

**Why outgoing segments are offered.** A camera at an intersection may watch the
queue arriving on one arm or the traffic discharging onto another. Listing only
approaches would decide that for the operator.

**Why `approach_id` is never filled in automatically.** The lookup from segment
to approach is unambiguous, but this record carries `observed` provenance, and
every value on it should be one a person actually stated.

**Why a stale network version *blocks* a segment confirmation** when it only
produced a warning at the junction level. Junction ids usually survive a
re-survey; segment ids, directions and which arms meet which node are exactly
what gets redrawn. Refining a stale junction decision would attach a direction
chosen by looking at a different map. Re-confirming the junction clears it.

## Consequences

A segment binding still changes **no route cost**, and this is now asserted by a
test rather than implied. The reason is specific: `RouteOptimizer` is handed the
single `ObservationMapping` held by `MappingStore` and pins its `mapping_id` and
`version` into every result so a recommendation can be replayed against the
bindings that produced it. This registry is neither versioned nor immutable — it
changes the moment an operator confirms anything — so feeding it in would leave
every route result pinned to an identifier that no longer describes the bindings
used, and the replay guarantee would still be printed while being false. On top
of that, `ObservationMapping` forbids one observation area being bound twice, and
a file mapping plus a confirmation for the same area would need a precedence rule
nobody has decided.

`GET /api/sites/confirmed` therefore reports `affects_route_cost: false` with a
`route_cost_note` saying why, so a client reads the inertness rather than
inferring it from a route that did not move.

`binding_for()` now renders a refined record with `segment_id` and `approach_id`
populated — the form costing understands — so connecting it later is a wiring
change, not a modelling one.

`ConfirmedSiteBinding` gains four optional fields with a validator keeping
`segment_id`, `segment_direction` and `segment_confirmation` present-or-absent
together, and refusing an approach on an outgoing segment. Records written by
step 2 still load unchanged.

Still no mapping provider, no geocoding and no external routing.
