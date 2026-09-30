# Architecture

Design rationale for the GenAI-Assisted Traffic Management System. This document
explains *why* the structure is what it is; the code carries the detail.

---

## 1. Scope boundary

The system is **decision support over recorded and simulated data**. It analyses
traffic, recommends actions, and evaluates those recommendations in simulation.
It never actuates anything.

That boundary is enforced structurally rather than by convention:

- Observed signal state (`SignalIndication` on `JunctionState`) and recommended
  signal action (`SignalAdvisory` in `domain/decision.py`) are **different types
  in different modules**. A fact about the world and a proposal to change it
  cannot be confused for one another.
- `SignalAdvisory.advisory_only` is validated to be `True` and cannot be set
  otherwise.
- No type in the domain carries an actuation address, controller id or endpoint.

---

## 2. Three lifetimes, three types

The most consequential modelling decision. Observations, network state and
emergency events have genuinely different lifetimes, and collapsing them into one
"TrafficState" object ties the lifetime of everything to the shortest-lived thing.

| Type | Scope | Lifetime |
|---|---|---|
| `JunctionObservation` | one source, one junction, one instant | immutable at creation, never updated |
| `NetworkState` | whole network, one instant | rebuilt continuously |
| `EmergencyState` | one declared response | persists across thousands of both |

`TrafficState` survives only as a deprecation-warned alias in
`domain/compat.py`. Nothing inside `app` imports it, and a test enforces that.
The name is ambiguous between "one camera's view" and "the whole network" — which
is precisely the confusion the three-way split exists to remove.

---

## 3. Static topology vs dynamic state

`RoadNetwork` (topology) and `NetworkState` (conditions) are separate and both
immutable.

Distance, lane count and capacity are properties of the map. Density, queue
length and blockage are properties of *this instant*. Putting both on one mutable
`RoadSegment` would mean:

- two concurrent emergency decisions racing on the same object, and
- no way to replay a decision against the state that actually produced it.

`NetworkIndex` (adjacency, id lookups) is built on demand rather than memoised
onto the frozen model — a derived acceleration structure has a different lifetime
from the value it accelerates.

---

## 4. Movements are first-class

A junction modelled as a bare graph node cannot express either fact this system
exists to reason about:

- *"Junction J11, approach 1 is congested"* — an **approach**-level fact.
- *"You may not turn right here"* — a **movement**-level fact.

So `Approach` and `TurnMovement` are modelled now, with `conflicting_movement_ids`
reserved for signal conflict checking. Phase 7 can expand to a movement graph —
the standard formulation for turn-aware routing — without breaking a schema.
Retrofitting this later would have been a rewrite of every model downstream.

Banned turns are recorded explicitly (`is_permitted=False`) rather than omitted:
omission is indistinguishable from an incomplete map import, and "unknown" must
never be read as "allowed".

---

## 5. The clock

Every traffic quantity is a rate — m/s, veh/h, veh/s, seconds-to-arrival. If any
downstream code called `time.time()`, replaying a recorded video at 10× would
multiply every speed and ETA by 10, silently, with no error anywhere.

So **the data source owns the clock**:

- `WallClock` — live sources. Time advances by itself.
- `VirtualClock` — recorded sources. Time is the media timeline, advancing only
  when a sample is decoded. Replay rate is recorded for telemetry and
  deliberately does not affect `now()`.

This is what makes "recorded and live converge on one pipeline" true rather than
aspirational: the pipeline is written once against `Clock`.

`VirtualClock.advance_to` refuses to move backwards; only an explicit `seek_to`
may. An out-of-order sample is a fault, not a silent timeline reorder.

---

## 6. Recorded and live are not the same thing

Pretending they are is the tempting design and it fails in both directions. So
`SourceCapabilities` declares the differences explicitly, and
`RecordedTrafficDataSource.validate_capabilities` /
`LiveTrafficDataSource.validate_capabilities` reject incoherent declarations at
construction.

| | Mode A (recorded) | Mode B (live) |
|---|---|---|
| bounded | yes | no |
| replayable | yes | no |
| seekable | usually | never |
| realtime | no | yes |
| backpressure | not required | **required** |
| clock | `VirtualClock` | `WallClock` |

Mode A is primary because **reproducibility** is what makes engineering possible:
without deterministic re-reads, "the route changed" can never be attributed to
the change we made.

`LiveTrafficDataSource.seek()` raises rather than being left unimplemented — a
silently ignored seek means analysing the wrong point in time while believing
otherwise.

---

## 7. Staleness is a safety property

Cameras do not report in lockstep. A naive "latest observation per junction" fuse
can mix a 200 ms-old reading with a 45 s-old one, and nothing in the result
records that. Routing an ambulance across such a snapshot is routing partly on
the present and partly on the past.

So every element carries its own `FreshnessStatus`, computed against a budget
recorded *inside* the snapshot (`StateAssemblyPolicy`) rather than read from
config at decision time. `NetworkState.is_degraded` then lowers reported
confidence and raises warnings.

Stale evidence is still **usable**. Withholding a route from an ambulance because
one camera lagged would be its own kind of failure. Unknown evidence is not
usable, and is priced by `soft.stale_evidence` rather than pruned — pruning every
unobserved element would disconnect a sparsely instrumented network.

---

## 8. Two safety tiers, applied in the right order

**Hard** constraints answer *may this element be used at all?* They are
admissibility filters applied to segments, junctions and movements **before**
search. **Soft** constraints answer *how much worse is it?* and contribute
penalties.

Selecting a route and then testing it is the obvious design and it is wrong
twice: it wastes work proportional to candidate count, and when the best route
fails there is no principled fallback because the alternatives are already gone.
Filtering first means whatever search returns is admissible *by construction*.

The corollary is handled explicitly: hard filtering can disconnect the graph.
`NoAdmissibleRouteError` is a first-class outcome — never a trigger for relaxing
a hard constraint to produce *some* answer. Phase 7 separates two causes of it,
because they call for opposite responses: `disconnected_by_pruning` means an
operator can clear an incident and restore the link, while `network_disconnected`
means the map as surveyed never joined those points.

Invariants, enforced by validators:

- A hard constraint **must not** carry a penalty. If it were priced, a
  sufficiently congested alternative could outbid a fire exclusion zone.
- A soft constraint **must** carry an explicit penalty. A soft constraint with no
  cost is a silently ignored safety signal.
- A constraint may not re-tier its catalog rule (`validate_constraint`).

Tier lives on the **rule**, not the instance, so "is a crowded crossing hard or
soft?" is answered once in a versioned, reviewable artifact. Escalating a
condition is a catalog change with a version bump and an audit trail.

### Exclusion zones are resolved once, and measured to the road

A zone is declared as a centre and a radius, but pruning cannot afford a
geometric query in the inner loop of a search. So a zone is resolved at
declaration into the set of segment and junction ids it covers, and pruning
becomes a set membership test.

The resolution itself has one trap worth naming. Testing each junction's
position and excluding a segment when an endpoint falls inside is the obvious
implementation, and it fails on exactly the case that matters: a long segment
running straight through the zone with both endpoints outside it is the road
past the fire, and it would stay admissible. Each segment is therefore treated
as a polyline — its surveyed geometry when present, its endpoints otherwise —
and the *minimum* distance from the centre to that polyline decides.

An element with no coordinates cannot be tested at all. Those are counted and
named rather than silently omitted, because a zone that looks resolved and
covers less than it claims is the failure mode where a hard constraint quietly
stops constraining. Incomplete coverage is a fact the caller must be able to
see.

Blockage and degradation stay apart all the way through. An incident's
`blocks_segment_ids` prunes; its `affected_segment_ids` prices. The domain model
rejects a segment listed as both, because the ambiguity would decide whether a
road is unusable or merely slow — and getting that backwards either strands a
responder or discards a usable route.

---

## 9. Route cost in equivalent seconds

The specification's formula — travel time + congestion + incident + safety +
accessibility penalties — is dimensionally incoherent if the penalties are
unitless points. Adding points to seconds produces a quantity that is neither.

Every term is therefore in **equivalent seconds**, and the total is a
*risk-adjusted arrival time*. "Cost 412" cannot be explained to an operator;
"310 s predicted traversal, plus 60 s for a crowded crossing, plus 42 s for
incident proximity" can be argued with — which is the property that makes it
possible to validate at all.

`RouteCostBreakdown` is always retained beside the total, and
`penalty_share` surfaces when a ranking rests on policy weights rather than on
measured traffic.

Rejected alternatives are kept in `RouteComparison`, because *"why not the short
route?"* is the first question anyone asks of an emergency recommendation.

**Why not a Pareto front:** returning a non-dominated set is defensible for a
planning tool where a human browses trade-offs. Under dispatch time pressure
somebody must still collapse it to one route, and doing that implicitly at the UI
layer hides the trade-off rather than recording it.

---

## 10. Emergency ETA is modelled independently

`TravelTimeEstimate` holds `nominal_travel_time_s` and
`emergency_travel_time_s` as independent fields, and the module provides **no
function converting one into the other**.

Multiplying nominal time by a constant is the standard shortcut and it is wrong
where it matters most. An emergency vehicle's advantage comes from discrete
mechanisms — passing stopped queues, receiving signal priority, using opposing
lanes — not from a proportion of normal travel time. On a free-flowing segment
the advantage is near zero; in stop-and-go it can be several-fold. A constant
ratio is optimistic in free flow and badly pessimistic in gridlock, which is
exactly the case that decides which route an ambulance takes.

An emergency ETA is also **conditional**: it depends on operational behaviour
this system can only recommend. `emergency_assumptions` is therefore mandatory
whenever an emergency estimate is present. *"3 minutes if priority is granted at
J14"* is defensible; *"3 minutes"* is not.

---

## 11. Provenance, and the meaning of absence

Every quantity that could be invented is a `Measurement` carrying mandatory
`Provenance`. `ValueSource` is a closed set — `observed`, `derived`, `predicted`,
`simulated`, `static`, `assumed` — with deliberately **no** `mock`, `default` or
`placeholder` member.

`assumed` requires `notes`; `derived` requires `input_refs`. An unjustified
assumption is indistinguishable from a fabrication, and an underived derivation
breaks the audit chain.

**Absence is `None`, never `0`.** Zero is a legitimate traffic reading — an empty
approach at 04:00 — and using it to mean "unknown" is how a routing engine
concludes that a gridlocked road is empty.

`Provenance.is_factual` distinguishes what may be stated as fact from what must
keep its modality ("predicted to", "in simulation", "assuming").

---

## 12. Computer vision: contract first, technology second

No detector backend is named *by the interfaces*. The criteria that decide one
are properties of the data and the deployment, not preferences: recall on small,
oblique, occluded vehicles in the actual camera geometry; throughput on the
hardware actually available; class coverage against the dataset's taxonomy;
licence weight. COCO mAP is close to uninformative for traffic-camera footage.

So Phase 1 fixed the contract — `VehicleDetector`, `VehicleTracker`,
`TrafficStateExtractor` — narrow enough that swapping backends is a constructor
change, which is what made an honest comparison affordable. Phase 3 then selected
against it: `app/vision/registry.py` is the only module naming a concrete
backend, and it *probes* rather than guesses — a backend is offered only once its
imports succeed, so selecting one can never fail later with an `ImportError` at
first inference. Where no backend's dependencies are installed, detection reports
as unavailable and the service still starts.

Two details that are easy to get wrong and are handled explicitly:

- **Ground contact, not centroid.** A box centroid sits at vehicle mid-height;
  projecting it through a ground-plane homography places the vehicle metres
  beyond its true position, with error growing with distance.
- **Timestamps, not frame indices.** Live sources drop frames and recorded video
  is often variable-frame-rate. A tracker assuming uniform spacing divides by the
  wrong `dt` and every speed is wrong.

Without calibration, metric speed/density/queue-length must be `None` and
`UNCALIBRATED` must be flagged — enforced by the `JunctionObservation` validator,
so the honest behaviour cannot be forgotten. **Counts survive without
calibration; metric quantities do not**, which is what makes an uncalibrated
camera partially useful rather than useless.

---

## 13. Analytics measure; they do not interpret

`app/analytics/` (Phase 4) answers *what measurable traffic behaviour is
occurring?* and stops there. Counts, spatial distribution, trajectories,
image-space speeds, flow over time, arrival rate and queue evidence are
measurements. Congestion level, travel time, ETA and route are **interpretations**
of measurements, and they live elsewhere: congestion in `app/state/` (Phase 5),
the rest in later phases. Keeping the boundary at the module edge is what stops a
level-of-service label — a policy judgement with thresholds — from being smuggled
in beside a vehicle count as though both were observed.

`TrafficAnalytics` describes **one observation area**: the field of view of one
camera. It makes no claim about junctions, approaches or lanes, because a single
video carries no information about how its view maps onto a road network. That
mapping is supplied separately, and only then can the result be projected onto a
`JunctionObservation`.

This is why traffic analytics is Phase 4 and junction and network state is
Phase 5, rather than the reverse. The fusion layer's unit of input is precisely
what analytics produces, and building it first would have meant fusing a value
type that did not exist yet.

### Interpretation refuses rather than guesses

`app/state/` (Phase 5) turns those measurements into a `CongestionLevel`. Three
properties make the verdict arguable rather than merely asserted:

- **The verdict carries its own inputs.** Every signal, its raw value, its
  normalised score, its weight, and the exact threshold set in force travel
  inside the `CongestionAssessment`. A disagreement becomes a specific argument
  about a threshold rather than a general suspicion about the system.
- **An unavailable signal contributes nothing, not zero.** Weights are
  renormalised over the signals actually available. Scoring a missing speed as
  0.0 would drag every uncalibrated camera toward "free flow" — the one
  direction it is dangerous to be wrong in, because it makes a jammed road look
  like the fast route.
- **`UNKNOWN` is a refusal, not a band.** Poor measurement quality, a tracking
  failure, or no usable signal produce `UNKNOWN` with a stated reason. It never
  orders between `FREE_FLOW` and the rest.

The evidence basis is reported alongside the level, because a verdict from
calibrated density and speed and one from frame occupancy are not comparable:
the second is camera-specific, and the same road under the same conditions
produces different numbers from a different mounting.

`LevelOfService` is deliberately **not** assigned. At a signalized junction LOS
is a function of control delay per vehicle; nothing in this layer measures
control delay, and mapping an occupancy ratio onto an HCM letter grade would put
an authoritative-looking label on a quantity that never entered the calculation.

The one thing this layer still cannot do is bind a verdict to a road segment.
`IntersectionTrafficState` is keyed by *observation area* — one camera's field of
view — not by a surveyed junction, and it is named that way so the distinction
survives contact with a reader in a hurry. The surveyed graph arrives in Phase 6,
which is also the first phase that needs it: incidents and exclusion zones are
expressed over network elements.

---

### Evidence reaches a road only through an explicit binding

Phase 5 interprets congestion for an **observation area** — one camera's field of
view. Phase 6 loaded a graph of segments. Nothing connected them, and Phase 6 said
so rather than guessing which road a camera watched.

`app/routing/mapping.py` is that connection, and it is a *survey artifact*:
somebody has to look at the camera and record which road it sees. It is loaded
from a versioned file, validated against the loaded network, and pinned into every
route result. It is deliberately not inferrable — a congestion band attached to
the wrong segment is a specific false claim that a route would then be planned
around.

An area with no binding contributes nothing and is **named** in the result. Its
evidence is real but unattributable, and silently dropping it would leave an
operator unable to tell "we saw nothing" from "we saw something and could not
place it".

### Unknown is priced, and never as zero

The rule the cost function turns on. A segment nobody watches has no congestion
band, and charging it nothing would make the optimizer prefer exactly the roads it
knows least about — it would route an ambulance down the unmonitored street
*because* it is unmonitored. So congestion is priced only where a band genuinely
exists, and absence draws a separate `uncertainty_penalty_s` instead. "Busy" and
"unknown" are different claims and carry different weights.

### An ETA exists, or it does not

A leg's time comes from the strongest basis available — an emergency-specific
estimate, a nominal one, or length over the segment's *surveyed* free-flow speed —
and from nothing at all when none of the three is there. There is no fallback
speed, because a fallback speed is an invented measurement whatever it is called.

A route with one untimed leg reports **no** ETA rather than the sum of the rest: a
partial sum understates arrival by exactly the legs it silently omitted. When no
route has a complete time basis, ranking falls back to distance, every route
reports no ETA, and the result says so. What never happens is a seconds-per-metre
proxy bridging the two bases — that is an invented speed under a different name.

### Emergency routing differs structurally, not by a multiplier

Scaling nominal travel time by a constant is the obvious way to model an emergency
advantage, and it is wrong where it matters most: optimistic in free flow, badly
pessimistic in the stop-and-go that actually decides the route. So no such
constant exists. Emergency weights *default to the normal ones*, and any numeric
divergence is an operator decision recorded in configuration.

What genuinely differs is structural: the responder's physical exclusions become
hard constraints, and an emergency-specific travel time is preferred where one
exists. Where none exists, the fallback is stated on the route itself — times are
the same as for general traffic, no advantage is modelled, and the estimate is
therefore conservative rather than flattering.

### Why Dijkstra, and why not A*

Every edge weight is a non-negative number of equivalent seconds, which is exactly
Dijkstra's precondition. A* was considered and rejected: an admissible heuristic
needs a lower bound on remaining cost, which needs both surveyed coordinates on
every junction and a defensible maximum speed. Junction positions are optional in
the schema, and a maximum speed would be an invented number. A heuristic that is
not admissible does not merely slow the search — it silently returns the wrong
route, which is the worst available failure for an emergency recommendation.

The graph is movement-expanded: a search node is `(junction, arriving segment)`,
because turn feasibility and turn cost both depend on how you arrived. A
node-level graph would happily route a fire engine through a banned right turn,
since the junction considered alone is fine.

---

## 14. Keeping GenAI honest

"Do not invent numbers" in a prompt is a request, not a control. Language models
comply unevenly, and fail most often by producing plausible round numbers exactly
where a reader is least likely to check.

So numbers are pinned to a `FactTable`. Every quantity the narrative may mention
exists there with an id, a value and provenance. Generated text cites fact ids,
and `scan_for_unsupported_numbers()` then verifies **mechanically** that no
numeric literal in the prose lacks backing. `DecisionRecord` refuses to carry
generated narrative without its validation result.

`Fact.display_variants` carries pre-rendered forms ("187 s", "3.1 min") so the
model never performs a unit conversion — that is an arithmetic step by a
component with no arithmetic guarantees, and it is unnecessary because we already
know the answer.

The scan is a lint, not a proof: it catches invented quantities, not invented
causal claims. Those are constrained separately, by giving the model only
structured state and requiring the reason to reference the cost breakdown routing
actually produced.

---

## 15. Reproducibility

`RunManifest` pins scenario, input source, and **independently versioned** code,
schema, network, safety rules, route weights and model artifacts.

They are separate identifiers on purpose. A safety rule can be revised without
changing the wire schema; a model can be retrained without changing either.
Collapsing them into one "app version" would destroy the ability to attribute a
behaviour change to its true cause.

A recommendation without a manifest is an anecdote.

---

## 16. Error semantics

`NotAvailableYetError` maps to **501 Not Implemented** with the owning phase
attached — never a 404, and never an empty-but-successful payload.

An empty 200 renders in the dashboard as a network with no congestion and no
incidents. That is a specific, false claim about the world, and it is precisely
the claim that would be most dangerous to make accidentally in a traffic system.
A 501 cannot be misread.

The frontend mirrors this: `Resource<T>` is a four-state union — `loading`,
`success`, `unavailable`, `error` — and "unavailable" is styled as a phase
boundary, not a fault. Presenting a working-as-designed phase boundary in red
trains operators to ignore red.
