# ADR 0004 — The data source owns the clock

**Status:** accepted (Phase 1)

## Context

Every traffic quantity is a rate: m/s, veh/h, veh/s, seconds-to-arrival.
Recorded input may be replayed faster or slower than real time.

## Decision

Introduce a `Clock` abstraction supplied by the data source — `WallClock` for
live sources, `VirtualClock` (media-timeline, rate-controlled) for recorded
sources. All downstream code reads time from the clock the source hands it.

## Rationale

If any downstream component called `time.time()` directly, replaying a recorded
video at 10x would multiply every speed and every ETA by 10 — silently, with no
error raised anywhere.

This is also the mechanism that makes "recorded and live converge on the same
pipeline" true rather than aspirational: the pipeline is written once against
`Clock` and is agnostic to which implementation it received.

## Consequences

Domain objects never read a clock themselves. `elapsed_response_time_s(now)` and
`Measurement.age_seconds_at(when)` take the instant as an argument, so a domain
object never decides which time base it is being evaluated against.

`VirtualClock.advance_to` refuses to move backwards; only an explicit `seek_to`
may. An out-of-order sample is therefore a fault rather than a silent reorder of
the timeline underneath the analytics.
