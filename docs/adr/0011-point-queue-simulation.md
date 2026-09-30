# ADR 0011 — A deterministic point-queue model, not a microsimulator

**Status:** accepted (Phase 8)

## Context

Phase 7 recommends a route. Nothing so far can answer whether the
recommendation is any *good*: the optimizer's ranking is an argument about
equivalent seconds, not a result. Without a simulator, "expected improvement" is
an assertion, and the emergency panel is right to report that no improvement
figure exists rather than showing an illustrative one.

So Phase 8 has to answer one narrow question honestly: **does the recommended
route actually beat the baseline, given the traffic we believe is there?**

The constraint that decides everything is what this project has actually
measured. It has: a surveyed graph with lengths, lane counts and free-flow
speeds; incidents that block or degrade segments; observation areas bound to
segments by an explicit survey, producing counts, arrival rates and stationary
vehicle counts. It does **not** have: an origin-destination matrix, signal
timings, capacities, jam densities, driver behaviour parameters, or a calibrated
speed-flow relationship for any road.

## Decision

Implement a **deterministic point-queue (vertical-queue) link model**.

Each directed segment has a free-flow traversal time and an exit discharge
capacity. Vehicles enter at a stated demand, run for at least the free-flow
time, and queue at the exit when demand exceeds capacity. The evaluated vehicle
is a probe: it experiences each queue as `queue / capacity` seconds of waiting,
and neither adds to the queue nor consumes capacity.

Two arms — a baseline route and the recommended one — are driven through the
*same* evolved network, so the difference between them is attributable to the
routes rather than to the conditions.

There is no random component, no sampling and no seed anywhere in the package.

## Alternatives considered

**Microscopic car-following (SUMO, IDM and relatives).** Rejected. It needs
per-vehicle acceleration, desired headway, reaction time and driver
heterogeneity. Nothing here measures any of them, so every one would be a number
invented to make the output look detailed — and the detail would be fiction with
a confidence interval attached. It would also introduce a large external
dependency whose calibration surface nobody in this project could defend.

**The cell transmission model.** Rejected, with regret. CTM propagates congestion
upstream, which a point queue cannot, but it needs a jam density and a backward
wave speed per link. Neither is surveyed. Its advantage is real, and its absence
is recorded as a stated limitation rather than hidden: a queue here is vertical,
occupies no road space, and cannot block the segment behind it.

**A BPR-style volume-delay function.** Rejected. It would produce a travel time
from a volume-to-capacity ratio in one step, but it is a steady-state equilibrium
curve with fitted exponents, answering a different question than "what happens to
this vehicle over the next fifteen minutes", with parameters nobody here has
calibrated.

**Deriving demand from the interpreted congestion band.** Rejected outright.
Turning "heavy" into a veh/h figure would be the exact fabrication the whole
system is built to prevent. Demand comes from an observed arrival rate through a
surveyed binding, or from an assumption an operator states and justifies, or it
does not come at all.

## Consequences

**Every input has three possible origins and no fourth.** Surveyed (from the
network file), observed (through an explicit observation-to-network binding), or
stated (an assumption on the request, which must carry a justification). A link
missing demand, capacity or a traversal time is reported *unmodelled* with the
reason. There is no default demand, no default capacity and no default speed
anywhere in `app/simulation/`.

**Absence propagates rather than being filled.** A route whose vehicle had not
arrived when the horizon ended reports no travel time, not a truncated one. A
comparison with an unavailable arm reports no improvement, not a one-sided
figure. A route crossing a blocked segment is refused rather than timed.

**Every output is labelled.** Each headline figure is a `Measurement` carrying
`ValueSource.SIMULATED`, so `is_factual` is false and Phase 12 cannot state it as
a fact about the world without deliberately stripping the modality. The
free-flow reference beside it carries `DERIVED`, because it is arithmetic over
surveyed values rather than an outcome of the run.

**Determinism is structural, not configured.** With no RNG there is no seed to
pin and no runs to average. The run identifier is the SHA-256 fingerprint of the
scenario, so identical inputs address the same run — and a caller can check
reproducibility rather than taking it on trust. The fingerprint deliberately
excludes the traffic-state id, which embeds the instant its snapshot was
assembled; what an observation actually contributes is already resolved onto the
links and hashed there.

**Known limits, stated in the API's own `/config` response.** Vertical queues do
not spill back. Links are evolved independently, because assigning discharged
vehicles to successors would need an OD matrix that does not exist. Demand is
constant over the horizon. A run over a sparsely surveyed network understates
congestion — which is the right direction for a figure an operator will act on,
and is named in the result rather than left to be discovered.
