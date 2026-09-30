# ADR 0007 — Emergency travel time is modelled independently

**Status:** accepted (Phase 1)

## Context

It is tempting to derive emergency travel time from nominal travel time by a
constant factor.

## Decision

`nominal_travel_time_s` and `emergency_travel_time_s` are independent fields
with independent provenance. The module exposes **no** conversion function, and
a test asserts that none exists. `emergency_assumptions` is mandatory whenever an
emergency estimate is present.

## Rationale

An emergency vehicle's advantage does not scale with normal travel time. It
comes from discrete mechanisms: passing stopped queues, receiving signal
priority, using opposing lanes. On a free-flowing segment the advantage is near
zero; in stop-and-go conditions it can be several-fold; on a blocked segment it
is undefined for both.

A constant ratio is therefore optimistic in free flow and badly pessimistic in
gridlock — and gridlock is exactly the case that decides which route an ambulance
takes.

An emergency ETA is also *conditional*: it depends on operational behaviour this
system can only recommend, never guarantee. "3 minutes if priority is granted at
J14" is a defensible statement; "3 minutes" is not.

## Consequences

Two models to build and validate in Phase 10 rather than one, each against
its own ground truth.

`emergency_advantage_ratio` exists for reporting only, computed *from* two
independent estimates as a sanity check. A value near or above 1.0 is a
legitimate signal that a segment offers no emergency advantage.
