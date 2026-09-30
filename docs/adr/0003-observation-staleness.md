# ADR 0003 — Staleness travels with the data

**Status:** accepted (Phase 1)

## Context

Cameras report at different rates and at different times. Fusing "the latest
observation per junction" discards how old each one is.

## Decision

Every element of a `NetworkState` carries its own `FreshnessStatus` and
`evidence_age_s`, computed against a budget recorded *inside* the snapshot
(`StateAssemblyPolicy`) rather than read from configuration at decision time.

## Rationale

A 200 ms-old reading and a 45 s-old reading are identical once the timestamp is
dropped. Routing an ambulance across such a snapshot means routing partly on the
present and partly on the past, with nothing in the result signalling that it
happened.

Recording the policy inside the snapshot means a decision can be audited against
the freshness rules that were actually in force when it was produced, not the
ones in the config file today.

## Consequences

Stale evidence remains usable with reduced confidence and an explicit warning:
withholding a route from an ambulance because one camera lagged would be its own
kind of failure.

Unknown evidence is priced by the `soft.stale_evidence` rule rather than pruned,
because pruning every unobserved element would disconnect a sparsely
instrumented network and deny a route outright — a worse failure than an
uncertain one.
