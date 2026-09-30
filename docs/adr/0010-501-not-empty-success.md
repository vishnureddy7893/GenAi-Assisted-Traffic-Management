# ADR 0010 — Unimplemented capabilities return 501, never an empty 200

**Status:** accepted (Phase 1)

## Context

During phased development many endpoints have no implementation yet, and the
dashboard must render something for them.

## Decision

`NotAvailableYetError` maps to **501 Not Implemented**, carrying the owning
phase in its details. The frontend models this as a distinct `unavailable` state
in `Resource<T>`, styled as a phase boundary rather than as a fault.

## Rationale

An empty but successful response renders in the dashboard as a network with no
congestion and no incidents. That is a specific, false claim about the world —
and it is precisely the claim that would be most dangerous to make accidentally
in a traffic system. A 501 cannot be misread.

Presenting a working-as-designed phase boundary in red would train an operator to
ignore red, so `unavailable` and `error` are visually and semantically distinct:
only `error` is announced as an alert and offers a retry.

## Consequences

Every panel handles four states — loading, success, unavailable, error — and
there is no shared "if anything failed, show nothing" path. One endpoint being
unimplemented must not blank a panel that is working.

The phase plan lives in `core/phases.py` and is served from `/api/meta/phases`,
so the error messages, the dashboard labels and the documentation all read from
one source and cannot drift apart.
