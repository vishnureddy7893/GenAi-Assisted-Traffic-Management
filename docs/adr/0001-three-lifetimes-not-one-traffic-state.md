# ADR 0001 — Three lifetimes, not one TrafficState

**Status:** accepted (Phase 1)

## Context

The specification permits `TrafficState` to survive as an alias for what one
camera observes at one timestamp.

## Decision

Model `JunctionObservation`, `NetworkState` and `EmergencyState` as distinct
types. Keep `TrafficState` only as a deprecation-warned alias in
`domain/compat.py`, unused inside `app` and enforced unused by a test.

## Rationale

The three have genuinely different lifetimes: an observation is immutable at
creation and never updated, a network state is rebuilt continuously, an
emergency persists across thousands of both. One object ties every lifetime to
the shortest-lived thing.

"TrafficState" is also ambiguous between one camera's view and the whole network
— the exact confusion the split exists to remove. In a greenfield repository
there is nothing to be backwards compatible with, so a second name for one
concept only guarantees that both survive in the codebase forever, and that a
reader must always check whether they mean the same thing.

## Consequences

Fusion from observations to network state becomes an explicit step with an
explicit staleness policy (ADR 0003).
