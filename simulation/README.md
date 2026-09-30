# Simulation

Controlled traffic simulation. **Phase 8 — implemented.**

The model itself lives in [`backend/app/simulation/`](../backend/app/simulation/)
rather than here, because the API has to import it. This directory holds scenario
notes and example request bodies.

## Purpose

To answer one question honestly: *does the recommended route actually beat the
baseline?* Without a simulator, a recommendation can be explained but never
evaluated, and "expected outcome" would be an assertion rather than a result.

## The model

A **deterministic point-queue (vertical-queue) link model**. Each directed
segment runs vehicles at a free-flow traversal time and discharges them at a
stated exit capacity; where demand exceeds capacity a queue forms at the exit,
and the evaluated vehicle waits `queue / capacity` seconds for the traffic ahead
of it to clear.

It was chosen because it needs exactly two inputs per link — a traversal time
and a discharge capacity — and both are either surveyed facts about the road or
assumptions an operator states and justifies. A microsimulator would need
per-vehicle acceleration, headway and reaction times that nothing in this project
has measured; the cell transmission model would need a jam density and a backward
wave speed that nothing here has surveyed. The reasoning is recorded in
[ADR 0011](../docs/adr/0011-point-queue-simulation.md).

## Comparison

Both arms run against the **same** network, the same demand and the same
horizon:

- **Baseline** — the shortest route by distance over the same admissible
  subgraph: what an operator would take from a map alone. Configurable to the
  optimizer's runner-up, or to a route named explicitly on the request.
- **Recommended** — the route Phase 7 selected.

Because both meet identical conditions, a difference between them is
attributable to the routes rather than to the traffic.

## Metrics

Travel time, delay against the free-flow reference, route completion time,
per-leg queue delay, mean speed, per-link peak/mean/final queue, throughput,
total delay in vehicle-seconds, saturation ratio, and the improvement between
the two arms.

Every metric carries `ValueSource.SIMULATED` provenance. A simulated result is
the modelled outcome of a hypothetical scenario and may never be presented as a
measurement of the real world — `Measurement.is_factual` is `False` for all of
them, so the narrative layer in Phase 12 cannot state one as fact without
deliberately stripping the modality.

## No performance figure is fabricated

There is no default demand, no default capacity and no default speed anywhere in
the package. A segment lacking any of them is reported **unmodelled**, with the
reason, and contributes no queue and no delay. A vehicle that had not arrived
when the horizon ended reports **no travel time**, not a truncated one. A
comparison where either arm is unavailable reports **no improvement**, not a
one-sided number.

## Determinism

No random number generator, no sampling, no seed. Identical inputs produce
identical output, and the run identifier *is* the SHA-256 fingerprint of the
scenario — so re-requesting the same scenario addresses the same run, and a
caller can check reproducibility rather than taking it on trust.

## Stated limitations

Published by `GET /api/simulation/config` as well as here, so they travel with
the numbers:

- Queues are vertical. They occupy no road space and cannot spill back into the
  segment upstream, so a network at capacity is modelled optimistically.
- Links are evolved independently. Vehicles discharged from a link leave the
  modelled area rather than moving to a successor, because no origin-destination
  matrix has been surveyed for this network.
- Demand is constant over the horizon unless stated otherwise.
- The evaluated vehicle is a probe: it experiences queues but neither adds to
  them nor consumes exit capacity.
- A run over a sparsely surveyed network understates congestion rather than
  inventing it — and says which segments it left out.

## Validation

A simulator that has not been checked against observed reality is a source of
confident nonsense. The model's arithmetic is verified against hand-computed
cases in `backend/tests/test_simulation.py` (vehicle conservation, queue growth
exactly when demand exceeds capacity, discharge time equal to queue over
capacity). Calibration against recorded footage — comparing modelled queues to
observed ones on a segment with a bound camera — belongs to the measured
evaluation in Phase 13, and until it exists the results are labelled as the
modelled outcomes of stated assumptions, which is what they are.

## Example request

`scenarios/example-grid-peak.json` is the body of a
`POST /api/simulation/compare` against the bundled `example-grid` network: a
peak-hour demand on the southern approach against a lane closure, with off-peak
demand elsewhere. Every assumption in it carries its own justification, because
the API requires one.
